"""上から挟んで持ち上げ、別の場所に置く動作 (pick & place)。

この関数群は「どこを・どの向きで掴むか」を引数で受け取るだけで、その値の出どころは知らない。
Phase 1 では真値を、Phase 3 以降はカメラからの推定値を渡す (= 知覚と制御を分離しておく)。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from control.motion import ArmController, choose_grasp_yaw, grasp_rotation

APPROACH_HEIGHT = 0.12  # 把持点の何 m 上から降りるか
LIFT_HEIGHT = 0.15  # 掴んだ後にどれだけ持ち上げるか
OPEN_WIDTH = 0.055  # 近づくときの指の開き [m]。全開 (8cm) だと隣の器械に当たりうる
GRASP_DEPTH = 0.0  # TCP を把持点からさらに何 m 下げるか (指パッドの中心を器械の厚みの中心に合わせる)
PLACE_CLEARANCE = 0.004  # 置くとき、支持バーの少し上で離す [m]


@dataclass
class GraspResult:
    closed_width: float  # 閉じた後の指の間隔 [m]。ほぼ 0 なら「何も挟めていない」
    yaw_used: float


def finger_width(ctrl: ArmController) -> float:
    d = ctrl.data
    return float(d.joint("finger_joint1").qpos[0] + d.joint("finger_joint2").qpos[0])


def pick(ctrl: ArmController, grasp_pos: np.ndarray, yaw: float) -> GraspResult:
    _, R_now = ctrl.commanded_tcp()
    yaw = choose_grasp_yaw(yaw, R_now)
    R = grasp_rotation(yaw)
    above = grasp_pos + [0, 0, APPROACH_HEIGHT]

    ctrl.set_gripper(OPEN_WIDTH)
    ctrl.move_to(above, R, duration=1.6)
    ctrl.move_to(grasp_pos + [0, 0, -GRASP_DEPTH], R, duration=1.0)
    ctrl.hold(0.2)  # 降りきった後の揺れが収まるのを待つ
    ctrl.set_gripper(0.0)
    ctrl.hold(0.6)  # 指が閉じきって把持力が立ち上がるのを待つ
    width = finger_width(ctrl)
    ctrl.move_to(grasp_pos + [0, 0, LIFT_HEIGHT], R, duration=1.2)
    return GraspResult(closed_width=width, yaw_used=yaw)


def place(ctrl: ArmController, place_pos: np.ndarray, yaw: float,
          before_release: Callable[[], None] | None = None) -> None:
    _, R_now = ctrl.commanded_tcp()
    R = grasp_rotation(choose_grasp_yaw(yaw, R_now))
    ctrl.move_to(place_pos + [0, 0, LIFT_HEIGHT], R, duration=2.0)
    ctrl.move_to(place_pos + [0, 0, PLACE_CLEARANCE], R, duration=1.0)
    ctrl.hold(0.2)
    if before_release:
        before_release()  # 評価用フック (離す直前の器械の位置を測る)
    ctrl.set_gripper(OPEN_WIDTH)
    ctrl.hold(0.5)
    ctrl.move_to(place_pos + [0, 0, APPROACH_HEIGHT], R, duration=0.8)
