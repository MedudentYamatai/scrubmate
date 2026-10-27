"""Phase 6: 器械出しロボットの頭脳。指示を聞いて渡す・返された器械を回収する・カウントする、を状態機械で切り替える。

状態 (phase)
  idle     … 指示待ち。数秒おきにトレイを撮り、返された器械があれば台帳に記録して回収する
  handoff  … 器械を渡している
  retrieve … トレイの器械を器械台に戻している
  count    … カウント中
  halted   … カウントの不一致で停止 (確認されるまで動かない)

優先順位 (ユーザーと相談して決定): 渡す・カウントの指示が先。回収の途中で指示が来たら、掴んでいる 1 本を器械台に置き終えてから応じる
(1 本ずつ回収し、1 本ごとに指示の列を確かめる)。術者の手がトレイの近くにある間は、回収に手を出さない。
"""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from control.motion import ArmController
from control.pick_place import LIFT_HEIGHT, OPEN_WIDTH, pick, plan_transfer
from demo.surgeon import DRAPE_SPOTS, Surgeon
from inventory.count import REGIONS, CountResult, InstrumentCounter
from inventory.draw import draw_count
from inventory.ledger import SURGEON, TABLE, TRAY, Ledger
from inventory.procedures import _free_slot, count_request, move_to_table
from language.interpret import Context, Interpreter, Result
from perception.locate import Locator, draw
from sim import robot
from sim.instruments import JA_NAMES

HANDOFF = np.array([0.40, 0.36, 0.95])  # 器械を差し出す点 (TCP)。術者の手の少し手前、トレイの上
TIP_YAW = -np.pi / 2  # 先端を術者と反対 (−y) に向け、持ち手を術者側にして渡す
MONITOR_EVERY = 2.0  # 指示待ちの間、トレイを撮る間隔 [s]
HAND_NEAR_MARGIN = 0.12  # トレイの箱をこれだけ広げた範囲に術者の手があれば、回収しない [m]
# 最後に手がトレイの近くに見えてから、これだけ経つまで回収しない [s]。術者が続けて返すとき、次の器械を取りに行く手は
# 検出カメラの視野の外に出る (ドレープは視野の外)。その間に回収を始めると、術者の手とロボットの腕がトレイの上でぶつかる
HAND_CLEAR_SEC = 4.0
# トレイの中身が変わってから、これだけ経つまで回収しない [s]。術者が器械を離す瞬間の手は低く (約 0.78m)、
# 重なった器械 (約 0.78m) と高さで見分けられず、手を見落として回収を始めたことがあった → 中身が落ち着くのを待つ
TRAY_SETTLE_SEC = 4.0


@dataclass
class Command:
    result: Result  # 音声認識 + 解釈の結果
    t: float  # 聞き終えた時刻


class ScrubNurse:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ctrl: ArmController, surgeon: Surgeon,
                 speak: Callable[[str], None], show: Callable[[np.ndarray, str], None],
                 note: Callable[[str], None], interpreter: Interpreter | None = None):
        self.model, self.data, self.ctrl, self.surgeon = model, data, ctrl, surgeon
        self.loc = Locator(model)
        self.counter = InstrumentCounter(self.loc)
        # 撮影のたびに検出カメラの小窓を更新する (カウント・見張り・並べ直しの途中も、ロボットが何を見ているかが分かるように)
        raw_count = self.counter.count

        def count_and_show(d: mujoco.MjData) -> CountResult:
            c = raw_count(d)
            self.show(draw_count(c, self.loc, d)[..., ::-1], "検出: 器械台・トレイの範囲で数える")
            return c

        self.counter.count = count_and_show
        self.ledger = Ledger()
        self.interp = interpreter or Interpreter()
        self.speak, self.show, self.note = speak, show, note  # 声で返事 / 検出カメラの画像を表示 / 状況の字幕
        self.queue: list[Command] = []
        self.step: str | None = None  # 今の工程
        self.with_surgeon: list[str] = []  # 術者に渡した順 (解釈の状況に使う。キャッシュの鍵になるので順番も大事)
        self.phase, self.phase_t0 = "idle", 0.0
        self.last_monitor_t = -1e9
        self.hand_seen_t = -1e9  # 術者の手を最後にトレイの近くに見た時刻
        self.tray_sig: tuple = ()  # 前回の見張りで見えたトレイの中身 (器械と位置 cm)
        self.tray_changed_t = -1e9
        self.ledger_message = ""  # 不一致の警告文
        self.ok_message = ""  # 数え直して一致したときの文

    # ---- 状態 ----
    def set_phase(self, p: str) -> None:
        self.phase, self.phase_t0 = p, self.data.time

    def context(self) -> Context:
        return Context(self.step, tuple(self.with_surgeon))

    def hear(self, heard: str) -> Result:
        """書き起こしを解釈して、指示の列に入れる (解釈はその時点の状況で行う)。"""
        r = self.interp.interpret(heard, self.context())
        if r.intent.intent != "other":
            self.queue.append(Command(r, self.data.time))
        return r

    # ---- メインループ (1 回呼ぶと、指示 1 つか、見張り 1 回か、0.1 秒の待機をする) ----
    def tick(self) -> None:
        if self.phase == "halted":
            # 停止中は受け渡しをしない。受け付けるのはカウント (数え直し) と返却の指示だけ。
            # 返された器械の記録はするが、回収はしない (数え直して一致するまで器械台に触らない)
            self.queue = [c for c in self.queue if c.result.intent.intent in ("count", "return")]
            if self.queue:
                self.execute(self.queue.pop(0))
            elif self.data.time - self.last_monitor_t >= MONITOR_EVERY:
                self.monitor(record_only=True)
            else:
                self.ctrl.hold(0.1)
            return
        if self.queue:
            self.execute(self.queue.pop(0))
        elif self.data.time - self.last_monitor_t >= MONITOR_EVERY and self.ledger.established:
            self.monitor()
        else:
            self.ctrl.hold(0.1)

    def execute(self, cmd: Command) -> None:
        it = cmd.result.intent
        if it.intent in ("request", "next") and it.instrument:
            self.step = it.step or self.step
            self.handoff(it.instrument, announce=(f"{it.step}ですね。" if it.intent == "next" and it.step else ""))
        elif it.intent == "count":
            self.count()
        elif it.intent == "return":
            self.speak("はい")  # 回収するかどうかは、見張りでトレイを見て決める

    # ---- 撮影 ----
    def _park(self) -> None:
        if np.abs(robot.arm_qpos(self.model, self.data) - robot.PARK_QPOS).max() > 0.02:
            self.ctrl.move_joints(robot.PARK_QPOS, duration=1.5)
            self.ctrl.hold(0.3)

    def hand_near_tray(self, c: CountResult) -> bool:
        """検出カメラの深度で、トレイの周りの高い所 (0.80m 以上) に背景に無かった物 = 術者の手があるかを見る。"""
        v, u = np.mgrid[0:c.depth.shape[0]:8, 0:c.depth.shape[1]:8]
        v, u = v.ravel(), u.ravel()
        o, d = self.loc.rays(self.data, u, v)
        p = o + d * c.depth[v, u][:, None]
        (x0, x1), (y0, y1) = REGIONS["tray"]
        m = HAND_NEAR_MARGIN
        near = ((p[:, 0] > x0 - m) & (p[:, 0] < x1 + m) & (p[:, 1] > y0 - m) & (p[:, 1] < y1 + m)
                & (p[:, 2] > 0.80) & self.loc._foreground(c.depth, v, u))
        return int(near.sum()) > 15

    # ---- 見張りと回収 ----
    def monitor(self, record_only: bool = False) -> None:
        """指示待ちの間の見張り: トレイを撮り、返された器械を台帳に記録し、術者の手が離れていれば 1 本回収する。
        record_only = 記録だけして回収しない (停止中)。"""
        self._park()
        c = self.counter.count(self.data)
        self.last_monitor_t = self.data.time
        tray = [d for d in c.detections if d.region == "tray"]
        for d in tray:
            if self.ledger.state.get(d.kind) == SURGEON:
                self.ledger.record_return(d.kind)
                if d.kind in self.with_surgeon:
                    self.with_surgeon.remove(d.kind)
                self.note(f"返却を確認: {JA_NAMES[d.kind]}")
                self.show(draw_count(c, self.loc, self.data, self.ledger)[..., ::-1], f"返却を確認: {JA_NAMES[d.kind]}")
        if record_only:
            return
        if self.hand_near_tray(c):
            self.hand_seen_t = self.data.time
        sig = tuple(sorted((d.kind, *np.round(d.pos[:2] * 100).astype(int)) for d in tray))
        if sig != self.tray_sig:
            self.tray_sig, self.tray_changed_t = sig, self.data.time
        if (not tray or self.data.time - self.hand_seen_t < HAND_CLEAR_SEC
                or self.data.time - self.tray_changed_t < TRAY_SETTLE_SEC):
            return
        d = max(tray, key=lambda d: d.pos[2])  # 上に載っている器械から
        slot = _free_slot(c, prefer=self.ledger.home_slot.get(d.kind))
        if slot is None:
            return
        self.set_phase("retrieve")
        self.show(draw_count(c, self.loc, self.data, self.ledger)[..., ::-1], f"回収: {JA_NAMES[d.kind]} → 器械台")
        self.note(f"回収: {JA_NAMES[d.kind]} をトレイから器械台へ")
        move_to_table(self.ctrl, self.counter, c, d, slot)
        if self.ledger.state.get(d.kind) == TRAY:
            self.ledger.move(d.kind, TABLE, by="robot")
            self.ledger.home_slot[d.kind] = slot
        self._park()
        self.set_phase("idle")
        self.last_monitor_t = -1e9  # すぐもう一度見る (トレイに残りがあれば続けて回収)

    # ---- 渡す ----
    def handoff(self, kind: str, announce: str = "") -> None:
        name = JA_NAMES[kind]
        place = self.ledger.state.get(kind)
        if place not in (TABLE, TRAY):
            self.speak(f"{name}は術者がお持ちです" if place == SURGEON else f"{name}は在庫にありません")
            return
        self.set_phase("handoff")
        self._park()
        ests, rgb = self.loc.locate(self.data)
        est = ests.get(kind)
        if est is None:
            self.speak(f"{name}が見つかりません")
            self.set_phase("idle")
            return
        self.show(draw(rgb, {kind: est}, locator=self.loc, data=self.data)[..., ::-1],
                  f"{name}: 把持点と先端の向きを推定")
        self.note(f"{name} を渡す (カメラで推定した把持点へ)")
        if est.tip_uncertain:
            self.speak(f"{name}の先端の向きが確認できません。ご注意ください")
        plan = None
        # 手首 (joint7, ±166°) の余裕は基準 0.15rad。器械台の右端では向きが数度ずれただけで基準を割ることがあるので 0.1 まで許す
        # (Phase 5 の試作: 30 回中 5 回。0.134〜0.148rad の計画で問題なく掴めた)
        for margin in (0.15, 0.10):
            try:
                plan = plan_transfer(self.ctrl, est.pos, est.yaw, HANDOFF - [0, 0, LIFT_HEIGHT], TIP_YAW, min_margin=margin)
                break
            except RuntimeError:
                continue
        if plan is None:
            self.speak(f"{name}を渡せる姿勢が見つかりません")
            self.set_phase("idle")
            return
        pick(self.ctrl, est.pos, est.yaw, plan=plan)
        self.ctrl.move_to_turning(HANDOFF, plan.pick_yaw, plan.place_yaw, 2.0)
        self.ctrl.hold(0.3)
        self.speak(announce + f"{name}です")
        spot = next(i for i in range(len(DRAPE_SPOTS)) if i not in self.surgeon.on_drape.values())
        self.surgeon.start(f"receive {kind}", self.surgeon.receive(kind, spot))
        t0 = self.data.time
        while kind not in self.surgeon.holding and self.data.time - t0 < 6.0:
            self.ctrl.hold(0.05)
        self.ctrl.set_gripper(OPEN_WIDTH)
        self.ctrl.hold(0.4)
        p, R = self.ctrl.commanded_tcp()
        self.ctrl.move_to(p + [0, 0, 0.10], R, 0.8)
        self.ledger.move(kind, SURGEON, by="robot")
        self.with_surgeon.append(kind)
        self._park()
        self.set_phase("idle")

    # ---- カウント ----
    def count(self) -> None:
        first = not self.ledger.established
        recount = self.ledger.halted
        # 不一致で止まった後の数え直し。一致すれば止まった状態を解く (一致しなければ、また止まる)
        self.ledger.halted = False
        self.set_phase("count")
        self.speak("器械カウントを始めます" if first else "もう一度数えます" if recount else "照合カウントを始めます")
        rep = count_request(self.ctrl, self.counter, self.ledger, log=self.note)
        img = draw_count(rep.counts[-1], self.loc, self.data, self.ledger, rep.message, rep.alert)
        self.show(img[..., ::-1], "初期カウント" if first else "照合カウントの結果")
        self.note(rep.message)
        if rep.alert:
            self.ledger_message = rep.message
            self.speak(rep.message)
            self.set_phase("halted")
        else:
            self.speak(rep.message.split(" (")[0])  # 初期カウントの括弧内 (器械名の一覧) は読み上げない
            if recount:
                self.ok_message = rep.message
            self.set_phase("idle")
        self.last_monitor_t = self.data.time
