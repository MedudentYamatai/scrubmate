"""Phase 5: 器械の台帳 (どの器械が今どこにあるか) を状態機械で管理し、画像のカウントと突き合わせる。

状態 (器械ごとに 1 つ)
  table   … 器械台
  surgeon … 術者 (手に持っている / 術野で使っている)
  tray    … 返却トレイ

許される移り方 (これ以外はエラー = 帳簿の付け間違いをその場で見つける)
  table   → surgeon   渡す (ロボット)
  tray    → surgeon   返却された器械をもう一度渡す (ロボット)
  surgeon → tray      返却 (術者)
  tray    → table     器械台に戻す (ロボット)

台帳はあくまで「ロボットと術者がやったはずのこと」の記録。体内遺残は記録と実物がずれることで起きるので、
閉創前には画像で数えた実物と、初期在庫・台帳の両方を比べる (reconcile)。
この課題の器械は 5 種類 1 本ずつなので、台帳は器械の種類をそのまま器械の名前として扱う。
"""

from collections import Counter
from dataclasses import dataclass, field

from sim import scene
from sim.instruments import JA_NAMES

TABLE, SURGEON, TRAY = "table", "surgeon", "tray"
PLACE_JA = {TABLE: "器械台", SURGEON: "術者", TRAY: "返却トレイ"}
TRANSITIONS = {
    (TABLE, SURGEON): "渡す",
    (TRAY, SURGEON): "渡す",
    (SURGEON, TRAY): "返却",
    (TRAY, TABLE): "器械台に戻す",
}


class LedgerError(RuntimeError):
    """台帳の規則に合わない操作 (例: 器械台にあるはずの器械を「返却」した)。"""


@dataclass
class Event:
    seq: int
    kind: str
    src: str
    dst: str
    action: str
    by: str  # robot / surgeon / count


@dataclass
class Reconciliation:
    """照合カウントの結果。"""
    seen: Counter  # 画像で数えた器械 (器械台 + トレイ) の本数
    missing: dict[str, str]  # 器械 → 台帳上の状態。初期在庫にあるのに見つからない
    unexpected: list[str]  # 初期在庫より多く見えた器械 (1 本ずつなので、取り違えの印)
    ledger_mismatch: dict[str, tuple[str, str]]  # 器械 → (台帳の場所, 見えた場所)。見えた場所が台帳と違う
    stacked: bool = False  # 器械が重なったまま (浮いている器械がある) = 数が信用できない

    def problems(self) -> list[str]:
        """術者に伝える問題点。空なら一致。"""
        out = []
        for kind, state in self.missing.items():
            # 台帳で術者が持っているはずなら「未返却」、器械台やトレイにあるはずなのに無ければ「所在不明」
            out.append(f"{JA_NAMES[kind]}1本{'未返却' if state == SURGEON else '所在不明'}")
        out += [f"{JA_NAMES[k]}が余分" for k in self.unexpected]
        # 台帳では術者が持っているのに見えた: 返却の記録漏れか、別の器械をこの器械と取り違えたか。
        # 取り違えなら本物は術野に残っているかもしれない (開発用の場面 種 100: 重なった鑷子を「メス」と数え、
        # 術野に残ったメスを見逃しかけた) → 黙って台帳を直さず、確認してもらう
        out += [f"{JA_NAMES[k]}が台帳では術者 (画像では{PLACE_JA[seen]})"
                for k, (led, seen) in self.ledger_mismatch.items() if led == SURGEON]
        if self.stacked:
            out.append("器械の重なりで数えきれない所あり")
        return out

    def named(self) -> set[str]:
        """警告文で名前を出した器械 (足りない・余分・台帳と違う)。"""
        return set(self.missing) | set(self.unexpected) | {k for k, (led, _) in self.ledger_mismatch.items() if led == SURGEON}

    @property
    def ok(self) -> bool:
        return not self.problems()

    def alert(self) -> str | None:
        """術者への警告文。例: 「持針器1本未返却。確認してください」"""
        p = self.problems()
        return "、".join(p) + "。確認してください" if p else None


@dataclass
class Ledger:
    initial: Counter | None = None  # 初期在庫 (初期カウントで確定)。None = まだ数えていない
    state: dict[str, str] = field(default_factory=dict)
    home_slot: dict[str, int] = field(default_factory=dict)  # 器械台に戻すときのスロット
    history: list[Event] = field(default_factory=list)
    halted: bool = False  # 照合で不一致 → 確認されるまでロボットは動かない

    @property
    def established(self) -> bool:
        return self.initial is not None

    def establish(self, table_items: dict[str, tuple[float, float]]) -> None:
        """初期カウントの結果 (器械台で見えた器械 → xy) を初期在庫として確定する。
        戻し先のスロットは、見えた位置に一番近い器械台のスロット。"""
        self.initial = Counter(table_items.keys())
        self.state = {k: TABLE for k in table_items}
        for kind, (x, y) in table_items.items():
            self.home_slot[kind] = min(scene.TABLE_SLOTS, key=lambda s: (scene.TABLE_SLOTS[s][0] - x) ** 2
                                       + (scene.TABLE_SLOTS[s][1] - y) ** 2)
        self.history.append(Event(len(self.history), "*", "-", TABLE, f"初期在庫 {len(table_items)} 本", "count"))

    def move(self, kind: str, dst: str, by: str) -> Event:
        if not self.established:
            raise LedgerError("初期カウントの前に器械を動かそうとした")
        if self.halted:
            raise LedgerError("カウントの不一致で停止中。確認されるまで動かさない")
        if kind not in self.state:
            raise LedgerError(f"{JA_NAMES[kind]} は初期在庫に無い")
        src = self.state[kind]
        action = TRANSITIONS.get((src, dst))
        if action is None:
            raise LedgerError(f"{JA_NAMES[kind]}: {PLACE_JA[src]} → {PLACE_JA[dst]} は許されない移り方")
        self.state[kind] = dst
        ev = Event(len(self.history), kind, src, dst, action, by)
        self.history.append(ev)
        return ev

    def record_return(self, kind: str) -> Event:
        """術者が器械をトレイに返したのを記録する (カメラで見えたとき)。停止中でも記録する (返却は確認の一部なので)。"""
        halted, self.halted = self.halted, False
        try:
            return self.move(kind, TRAY, by="surgeon")
        finally:
            self.halted = halted

    def at(self, place: str) -> list[str]:
        return [k for k, s in self.state.items() if s == place]

    def reconcile(self, counted: list[tuple[str, str]], stacked: bool = False) -> Reconciliation:
        """画像のカウント ((器械, 見えた場所 table / tray) の一覧) を、初期在庫・台帳と突き合わせる。
        判定の基準は初期在庫 (最初に数えた実物)。台帳は「無い理由」(未返却か所在不明か) と記録漏れの指摘に使う。
        同じ器械が 2 回数えられたら「余分」になる。1 本ずつしか無いので、重なりなどで別の器械と取り違えた印。"""
        if not self.established:
            raise LedgerError("初期在庫が無いので照合できない")
        seen = Counter(k for k, _ in counted)
        missing = {k: self.state[k] for k in self.initial if seen[k] < self.initial[k]}
        unexpected = sorted(k for k in seen if seen[k] > self.initial.get(k, 0))
        mismatch = {k: (self.state[k], place) for k, place in dict(counted).items()
                    if seen[k] == 1 and k in self.state and self.state[k] != place}
        return Reconciliation(seen=seen, missing=missing, unexpected=unexpected, ledger_mismatch=mismatch,
                              stacked=stacked)

    def correct(self, kind: str, place: str) -> None:
        """記録漏れの訂正: 画像で見えた場所に台帳を合わせる (例: 術者が黙ってトレイに返していた)。"""
        self.history.append(Event(len(self.history), kind, self.state[kind], place, "記録の訂正 (カウントで確認)", "count"))
        self.state[kind] = place

    def summary_lines(self) -> list[str]:
        """表示用: 1 行 1 器械 (例: 「持針器   術者」)。"""
        return [f"{JA_NAMES[k]}: {PLACE_JA[s]}" for k, s in self.state.items()]
