"""Phase 6: 術者 (手のマーカー = mocap) のスクリプト。人体モデルは作らず、手の位置と向きを予定どおりに動かす。

術者の動きは、制御周期ごとに update() で少しずつ進める (ジェネレーターで書いた「動きの手順」を 1 歩ずつ進める)。
こうすると、ロボットが動いている最中 (pick などの関数の中) にも術者が動ける。

受け渡しの瞬間だけ weld 拘束で器械を手に固定する (術者側はスクリプトなので拘束で代用してよい: 仕様)。
受け取った器械は術野のドレープの上に置いて「使っている」ことにする。返すときはそこから取ってトレイへ。

返し方は Phase 5 の試作で決めた (learning_log の Phase 5 Q&A):
  向き   … 支持バーと直交 (先端 0° か 180°)。平行だとバーの間に落ちて縦や斜めになり、検出も把持もできなかった
  前後   … トレイの置き場の ±1cm 以内。2cm ずれるとペアンの重心がバーの外に出て傾いた
  高さ   … バーの 1.5cm 上でそっと離す。6.5cm 上から落とすと持針器などが跳ねて裏返った
"""

from collections.abc import Generator

import mujoco
import numpy as np

from control.motion import CONTROL_DT
from sim import scene

REST = np.array([0.30, 0.62, 1.02])  # 待機中の手の位置 (患者の上)
# 術野のドレープの上の置き場 (器械の重心の xy)。患者 (y 0.67〜0.93) とトレイ (y < 0.49) の間
DRAPE_SPOTS = [(0.42, 0.585), (0.30, 0.585), (0.18, 0.585), (0.06, 0.585)]
DRAPE_Z = 0.79 + 0.005 + 0.004  # ドレープの上面 + 器械の厚みの半分 + 少し浮かせて離す
TRAY_RELEASE_Z = scene.INSTRUMENT_Z + 0.015  # バーの 1.5cm 上で離す
GRIP_OFFSET = 0.06  # 器械の重心から持ち手側に何 m の所を持つか


def _yaw_of(data: mujoco.MjData, kind: str) -> float:
    ax = data.body(kind).xmat.reshape(3, 3)[:, 0]
    return float(np.arctan2(ax[1], ax[0]))


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class Surgeon:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.data.mocap_pos[0] = REST
        self.data.mocap_quat[0] = [1, 0, 0, 0]
        self._job: Generator | None = None
        self.job_name = ""
        self.holding: list[str] = []
        self.on_drape: dict[str, int] = {}  # 器械 → ドレープの置き場の番号
        self.last_release_t = -1.0  # 最後にトレイに器械を離した時刻 (シミュレーター内)
        self.log: list[tuple[float, str]] = []

    # ---- 状態 ----
    @property
    def busy(self) -> bool:
        return self._job is not None

    def update(self) -> None:
        """制御周期ごとに呼ぶ。今の動きを 1 歩進める。"""
        if self._job is None:
            return
        try:
            next(self._job)
        except StopIteration:
            self._job, self.job_name = None, ""

    def start(self, name: str, job: Generator) -> None:
        assert self._job is None, f"術者は {self.job_name} の途中"
        self._job, self.job_name = job, name
        self.log.append((self.data.time, name))

    # ---- 基本の動き (ジェネレーター) ----
    def _hand_yaw(self) -> float:
        q = self.data.mocap_quat[0]
        return float(2 * np.arctan2(q[3], q[0]))

    def _move(self, target, duration: float, yaw: float | None = None) -> Generator:
        """手を target へ、なめらかに (3 次の補間) 動かす。yaw を渡すと手首も回す。"""
        p0, y0 = self.data.mocap_pos[0].copy(), self._hand_yaw()
        y1 = y0 if yaw is None else y0 + _wrap(yaw - y0)
        n = max(1, round(duration / CONTROL_DT))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            self.data.mocap_pos[0] = p0 + s * (np.asarray(target, float) - p0)
            a = y0 + s * (y1 - y0)
            self.data.mocap_quat[0] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
            yield

    def _wait(self, duration: float) -> Generator:
        for _ in range(max(1, round(duration / CONTROL_DT))):
            yield

    def _grip_point(self, kind: str) -> np.ndarray:
        """持つ所: 器械の重心から持ち手側 (先端の反対) に GRIP_OFFSET。"""
        b = self.data.body(kind)
        return b.xpos - b.xmat.reshape(3, 3)[:, 0] * GRIP_OFFSET

    def _grab(self, kind: str) -> None:
        scene.hand_grab(self.model, self.data, kind)
        self.holding.append(kind)

    def _release(self, kind: str) -> None:
        scene.hand_release(self.model, self.data, kind)
        self.holding.remove(kind)

    # ---- 手順 ----
    def receive(self, kind: str, spot: int) -> Generator:
        """ロボットが差し出した器械の持ち手を握り、術野のドレープの置き場 spot に置いて手を戻す。
        ロボットは self.holding に kind が入ったら指を開く。"""
        g = self._grip_point(kind)
        yield from self._move(g + [0, 0, 0.03], 0.9)
        yield from self._move(g + [0, 0, 0.01], 0.4)
        self._grab(kind)
        yield from self._wait(0.8)  # ロボットが指を開いて離れるのを待つ
        yield from self._move(self.data.mocap_pos[0] + [0, 0.05, 0.08], 0.8)
        x, y = DRAPE_SPOTS[spot]
        # 持っている器械の重心がドレープの置き場の真上に来るように手を動かす (手と重心の位置関係は weld で固定されている)
        off = self.data.mocap_pos[0] - self.data.body(kind).xpos
        yield from self._move(np.array([x, y, DRAPE_Z + 0.08]) + off, 1.2)
        yield from self._move(np.array([x, y, DRAPE_Z]) + off, 0.5)
        self._release(kind)
        self.on_drape[kind] = spot
        yield from self._wait(0.3)
        yield from self._move(REST, 1.0)

    def return_to_tray(self, kind: str, slot: int, dy: float = 0.0, yaw_dev: float = 0.0,
                       drop_z: float = TRAY_RELEASE_Z, go_rest: bool = True) -> Generator:
        """ドレープの上の器械を取り、返却トレイの置き場 slot (+dy) に、支持バーと直交する向き (+yaw_dev) で置く。
        go_rest=False なら、置いた後も手をトレイの近くに残す (続けてもう 1 本返すとき。手が近くにあるとロボットは回収しない)。"""
        self.on_drape.pop(kind, None)
        g = self._grip_point(kind)
        yield from self._move(g + [0, 0, 0.06], 1.0)
        yield from self._move(g + [0, 0, 0.01], 0.4)
        self._grab(kind)
        yield from self._move(self.data.mocap_pos[0] + [0, 0, 0.10], 0.6)
        # 器械の先端を 0° か 180° (近い方) + yaw_dev に向ける: 手を回す量 = 目標 − 今の器械の向き
        cur = _yaw_of(self.data, kind)
        target = (0.0 if abs(_wrap(cur)) < np.pi / 2 else np.pi) + yaw_dev
        yield from self._move(self.data.mocap_pos[0], 0.6, yaw=self._hand_yaw() + _wrap(target - cur))
        x, y = scene.TRAY_SLOTS[slot]
        off = self.data.mocap_pos[0] - self.data.body(kind).xpos
        yield from self._move(np.array([x, y + dy, drop_z + 0.08]) + off, 1.2)
        yield from self._move(np.array([x, y + dy, drop_z]) + off, 0.5)
        self._release(kind)
        self.last_release_t = self.data.time
        yield from self._wait(0.3)
        yield from self._move(self.data.mocap_pos[0] + [0, 0.06, 0.12], 0.6, yaw=0.0)
        if go_rest:
            yield from self._move(REST, 1.0)
