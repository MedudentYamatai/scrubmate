"""Phase 5: カウントの手順 (腕の動き + 撮影 + 台帳)。Phase 6 のデモからは count_request を呼ぶ。

  count_request(...) … 「器械カウント」系の指示が来たときの入口。
                       初期在庫がまだ無ければ初期カウント、あれば照合カウント (閉創前など) にする (ユーザーと相談して決定)

初期カウント: 腕を待避 → 器械台を数える → 初期在庫として確定。
照合カウント:
  1. 腕を待避 → 器械台とトレイを数え、初期在庫・台帳と突き合わせる
  2. 合わなければ (足りない・同じ器械が 2 か所に見える・器械が浮いている)、トレイの器械と浮いている器械を
     器械台の空いているスロットへ 1 本ずつ並べ直してから数え直す (ユーザーと相談して決定。合うときは並べ直さない)。
     トレイの中で器械が重なると、下の器械を見逃したり別の器械と取り違えたりする (Phase 3: 重なりのある画像の約 15%)。
     実際の手術室でも、カウントは器械を並べてから行う
  3. それでも合わなければ警告し、待機姿勢 (待避姿勢) で止まる
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from control.motion import ArmController
from control.pick_place import pick, place, plan_transfer
from inventory.count import CountResult, Detection, InstrumentCounter
from inventory.ledger import SURGEON, TABLE, TRAY, Ledger, Reconciliation
from sim import robot, scene
from sim.instruments import JA_NAMES

MAX_ROUNDS = 2  # 並べ直し → 数え直し の上限
MAX_MOVES = 6  # 1 回の並べ直しで運ぶ本数の上限 (トレイに最大 3 本 + 器械台で重なった分)
TABLE_YAW = 0.0  # 器械台には先端を +x に揃えて置く
SLOT_FREE_DIST = 0.04  # スロットの中心からこれ以内に器械が見えていたら、そのスロットは埋まっている
# 重なりの上の器械を掴むときに、掴む高さを上げる量。指のパッドは高さ 17mm (掴む点の上下 8.5mm) あり、
# 器械 (厚さ 10mm) の中心を掴むと下の器械にも 3.5mm かかって 2 本まとめて運んでしまった (開発用の場面 種 1)。
# 3mm 上げれば上の 1 本だけ持ち上がった (3 / 4.5 / 6mm で確認) → 間を取って 4.5mm
STACK_GRASP_LIFT = 0.0045
TRIED_XY = 0.03


@dataclass
class CountReport:
    kind: str  # initial / check
    counts: list[CountResult] = field(default_factory=list)  # 撮った順 (並べ直すと複数)
    recon: Reconciliation | None = None
    moved: list[str] = field(default_factory=list)  # 並べ直しで運んだ器械 (検出したときの名前)
    message: str = ""

    @property
    def alert(self) -> bool:
        return self.recon is not None and not self.recon.ok


def park_and_count(ctrl: ArmController, counter: InstrumentCounter) -> CountResult:
    ctrl.move_joints(robot.PARK_QPOS, duration=1.5)
    ctrl.hold(0.3)
    return counter.count(ctrl.data)


def nearest_slot(xy) -> int:
    return min(scene.TABLE_SLOTS, key=lambda s: np.hypot(*(np.array(scene.TABLE_SLOTS[s]) - xy[:2])))


def initial_count(ctrl: ArmController, counter: InstrumentCounter, ledger: Ledger) -> CountReport:
    c = park_and_count(ctrl, counter)
    table = {d.kind: tuple(d.pos[:2]) for d in c.detections if d.region == "table"}
    ledger.establish(table)
    names = "・".join(JA_NAMES[k] for k in sorted(table, key=lambda k: ledger.home_slot[k]))
    return CountReport("initial", [c], message=f"器械台に {len(table)} 本を確認しました ({names})")


def _to_move(c: CountResult) -> list[Detection]:
    """並べ直す器械: トレイの器械全部と、器械台で浮いている (下に器械が挟まっている) 器械。高い順 = 上に載っている順。
    「器械台の同じスロットに 2 本見えたら重なり」の規則も試したが、開発用の場面では成績が変わらず、
    count_demo --pile では取り違えた検出まで運んでかえって崩れたのでやめた。"""
    return sorted((d for d in c.detections if d.region == "tray" or (d.region == "table" and d.elevated)),
                  key=lambda d: -d.pos[2])


def _free_slot(c: CountResult, prefer: int | None = None) -> int | None:
    """器械台の空いているスロット (どの器械も見えていない所)。prefer (その器械の元のスロット) が空いていればそこ。
    浮いている器械がいるスロットは空きにしない (その下には、隠れた器械がいる)。"""
    occupied = [d.pos[:2] for d in c.detections if d.region == "table"]
    free = [s for s, xy in scene.TABLE_SLOTS.items()
            if all(np.hypot(*(np.array(xy) - p)) > SLOT_FREE_DIST for p in occupied)]
    if prefer in free:
        return prefer
    return free[0] if free else None


def move_to_table(ctrl: ArmController, counter: InstrumentCounter, c: CountResult, d: Detection, slot: int) -> None:
    """検出 d の器械を、器械台のスロット slot に先端を揃えて置く。掴む位置と向きは、この検出の輪郭から推定する
    (器械名で探し直さない。重なりで取り違えていると、同じ名前の別の器械を掴みに行ってしまうため)。"""
    est = counter.loc.estimate_one(ctrl.data, c.depth, d.kind, d.conf, d.polygon)
    grasp = est.pos.copy() if est is not None else d.pos.copy()
    yaw = est.yaw if est is not None else 0.0
    if d.elevated:
        grasp[2] += STACK_GRASP_LIFT
    x, y = scene.TABLE_SLOTS[slot]
    dst = np.array([x, y, scene.INSTRUMENT_Z])
    plan = plan_transfer(ctrl, grasp, yaw, dst, TABLE_YAW)
    pick(ctrl, grasp, yaw, plan=plan)
    place(ctrl, dst, plan.place_yaw, yaw_from=plan.pick_yaw)
    ctrl.hold(0.4)


def rearrange(ctrl: ArmController, counter: InstrumentCounter, ledger: Ledger, rep: CountReport,
              log: Callable[[str], None]) -> None:
    """トレイの器械と浮いている器械を、上から 1 本ずつ器械台の空きスロットへ運ぶ。1 本運ぶごとに撮り直す
    (上の器械をどけると、下の器械が見えてくるため)。"""
    tried: list[np.ndarray] = []  # 運ぼうとした器械の位置
    for _ in range(MAX_MOVES):
        c = park_and_count(ctrl, counter)
        # 運べたかは、撮り直して同じ所にまだ器械が見えるかで分かる (真値は使わない)。まだ見えていたら掴み損ねたので、
        # 同じ器械を何度も掴みに行かないように飛ばす (トレイの壁際で掴めず、同じ器械を 6 回掴みに行ったことがあった)。
        # 掴み損ねると器械が 2cm ほど動くことがある (開発用の場面 種 100) ので、同じ所 = xy で TRIED_XY 以内
        # 上の器械をどけた後に下から出てきた器械は、xy はほぼ同じでも高さが約 1cm 低いので、高さで区別する
        todo = [d for d in _to_move(c)
                if all(np.linalg.norm(d.pos[:2] - t[:2]) > TRIED_XY or abs(d.pos[2] - t[2]) > 0.005 for t in tried)]
        if not todo:
            return
        d = todo[0]
        slot = _free_slot(c, prefer=ledger.home_slot.get(d.kind))
        if slot is None:
            log("器械台に空きスロットが無い")
            return
        log(f"  {JA_NAMES[d.kind]} ({'トレイ' if d.region == 'tray' else '器械台で重なり'}) → 器械台のスロット {slot}")
        move_to_table(ctrl, counter, c, d, slot)
        rep.moved.append(d.kind)
        tried.append(d.pos)


def _sync_ledger(ledger: Ledger, c: CountResult, recon: Reconciliation) -> None:
    """見えた場所に台帳を合わせる。トレイにあるはずの器械が器械台にある = ロボットが並べ直した (正規の移動)。
    台帳では術者が持っている器械が見えた場合は直さない (取り違えかもしれないので、警告して確認してもらう)。
    足りない器械も直さない (警告の対象)。"""
    for kind, (led, seen) in recon.ledger_mismatch.items():
        if led == SURGEON:
            continue
        if led == TRAY and seen == TABLE:
            ledger.move(kind, TABLE, by="robot")
        else:
            ledger.correct(kind, seen)
        if seen == TABLE:
            ledger.home_slot[kind] = nearest_slot(next(d.pos for d in c.detections if d.kind == kind))


def check_count(ctrl: ArmController, counter: InstrumentCounter, ledger: Ledger, rearrange_on_mismatch: bool = True,
                log: Callable[[str], None] = print) -> CountReport:
    rep = CountReport("check")
    for rnd in range(MAX_ROUNDS + 1):
        c = park_and_count(ctrl, counter)
        rep.counts.append(c)
        recon = ledger.reconcile(c.counted(), stacked=any(d.elevated for d in c.detections))
        rep.recon = recon
        if recon.ok and not recon.ledger_mismatch:
            break
        if not rearrange_on_mismatch or rnd == MAX_ROUNDS or not _to_move(c):
            break  # 並べ直しても変わらない (トレイが空で、重なりも無い)
        log(f"カウントが合わない ({recon.alert() or '台帳と場所が違う'}) → 並べ直して数え直す")
        rearrange(ctrl, counter, ledger, rep, log)

    _sync_ledger(ledger, rep.counts[-1], recon)
    if recon.ok:
        rep.message = f"カウント一致。{sum(recon.seen.values())} 本すべて確認しました"
    else:
        rep.message = recon.alert()
        ledger.halted = True  # 確認されるまで受け渡しをしない
        ctrl.move_joints(robot.PARK_QPOS, duration=1.5)  # 待機姿勢で停止
    return rep


def count_request(ctrl: ArmController, counter: InstrumentCounter, ledger: Ledger, **kw) -> CountReport:
    """intent=count の入口。初期在庫が無ければ初期カウント、あれば照合カウント。"""
    if not ledger.established:
        return initial_count(ctrl, counter, ledger)
    return check_count(ctrl, counter, ledger, **kw)
