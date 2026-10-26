"""上から挟んで持ち上げ、別の場所に置く動作 (pick & place)。

この関数群は「どこを・どの向きで掴むか」を引数で受け取るだけで、その値の出どころは知らない。
Phase 1 では真値を、Phase 3 以降はカメラからの推定値を渡す (= 知覚と制御を分離しておく)。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from control.motion import ArmController, choose_grasp_yaw, grasp_rotation
from sim import robot, scene

APPROACH_HEIGHT = 0.12  # 把持点の何 m 上から降りるか
LIFT_HEIGHT = 0.15  # 掴んだ後にどれだけ持ち上げるか
OPEN_WIDTH = 0.055  # 近づくときの指の開き [m]。全開 (8cm) だと隣の器械に当たりうる
GRASP_DEPTH = 0.0  # TCP を把持点からさらに何 m 下げるか (指パッドの中心を器械の厚みの中心に合わせる)
PLACE_CLEARANCE = 0.004  # 置くとき、支持バーの少し上で離す [m]


@dataclass
class TransferPlan:
    """掴んでから置くまでの手首の回し方。"""

    pick_yaw: float  # 掴むときの手の yaw
    q_above: np.ndarray  # 把持点の真上での関節角 (この姿勢へ関節空間で直接動く)
    place_yaw: float  # 置くときの手の yaw (pick_yaw との差 = 運ぶ間に回す角度。符号が回す向き)
    wrist_margin: float  # 手首 (joint7) が関節の限界から一番近づいたときの余裕 [rad]


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _wrist_reference(ctrl: ArmController, pos: np.ndarray) -> tuple[np.ndarray, float]:
    """手先を pos に真下向きで置く腕の構え q_ref と、そのときの手の yaw (y_ref) を 1 つ求める。

    y_ref は home 姿勢の手の向きを、根元 (joint1) が pos の方を向くぶんだけ回したもの。
    腕にとって一番素直な向きなので、IK が他の関節をねじらずに収束する。"""
    q0 = robot.HOME_QPOS.copy()
    _, R_home = ctrl.ik.fk(q0)
    q0[0] = np.arctan2(pos[1], pos[0])
    y_ref = float(np.arctan2(R_home[1, 0], R_home[0, 0]) + q0[0])
    q, ok, _ = ctrl.ik.solve(q0, pos, grasp_rotation(y_ref), iters=400)
    if not ok:
        raise RuntimeError(f"{np.round(pos, 3)} に真下向きで手が届かない")
    return q, y_ref


def _wrist_angle(q_ref: np.ndarray, y_ref: float, hand_yaw: float, lo: float, hi: float) -> float | None:
    """手の yaw を hand_yaw にしたときの joint7。可動範囲に入らなければ None。

    Panda の手首 (joint7) の回転軸は手先 (TCP) を真っすぐ通る。手が真下を向いていれば、
    joint7 だけを回すと、手先の位置は変わらずに手の yaw だけが回る (他の 6 関節はそのまま)。
    手の z 軸は下向きなので、手の yaw を +θ 回す = joint7 を −θ 回す。
    yaw は 360° 回せば同じ向きなので、joint7 も ±2π ずらした値のどれかが範囲に入ればよい。"""
    j = q_ref[6] - _wrap(hand_yaw - y_ref)
    for k in (0, 1, -1):
        if lo <= j + 2 * np.pi * k <= hi:
            return float(j + 2 * np.pi * k)
    return None


def plan_transfer(ctrl: ArmController, grasp_pos: np.ndarray, inst_yaw: float,
                  place_pos: np.ndarray, place_inst_yaws: float | tuple[float, ...]) -> TransferPlan:
    """器械 (先端の向き inst_yaw) を、先端が place_inst_yaws のどれかを向くように置くための手の回し方を決める。
    向きを問わない置き場 (返却トレイ) では (0, π) のように複数渡すと、手首に余裕がある方を選ぶ。

    選択肢は 2 × 2 通り:
      掴み方    … 手の yaw = inst_yaw か inst_yaw+π (グリッパは前後対称なので、どちらでも同じように挟める)
      回す向き  … 運ぶ間に器械を回すとき、右回りか左回りか
    Panda の手首 (joint7) は ±166° しか回らないので、組み合わせによっては途中で限界に当たり、
    手が回りきらずに器械が斜めに置かれる (試作で 30° ずれた)。
    掴む所・置く所それぞれで joint7 の角度を計算し、手首が限界から一番遠い組み合わせを選ぶ。
    joint7 の可動域は 332° なので、どの場所にも「手が向けない yaw」が約 28° ある。
    例えばトレイでは手の yaw ±180° 付近に向けられず、器械台の右端から先端 180° 向きでトレイに置く組み合わせは解が無かった。"""
    lo, hi = ctrl.ik.lo[6], ctrl.ik.hi[6]
    min_margin = 0.15  # 手首の限界まで最低これだけ (約 9°) 余裕がある組み合わせだけ使う
    q_pick_ref, y_pick_ref = _wrist_reference(ctrl, grasp_pos + [0, 0, APPROACH_HEIGHT])
    q_place_ref, y_place_ref = _wrist_reference(ctrl, place_pos + [0, 0, LIFT_HEIGHT])
    targets = np.atleast_1d(place_inst_yaws)
    best, notes = None, []
    for h, d in ((h, _wrap(t - inst_yaw)) for t in targets for h in (_wrap(inst_yaw), _wrap(inst_yaw + np.pi))):
        # d = 器械を回す角度 = 手を回す角度 (360° の違いを除いて)
        j_pick = _wrist_angle(q_pick_ref, y_pick_ref, h, lo, hi)
        if j_pick is None:
            notes.append(f"手の yaw {np.degrees(h):+.0f}° で掴む: 手首の範囲外")
            continue
        for rot in (d, d - 2 * np.pi, d + 2 * np.pi):
            # 運ぶ間に手を rot 回す → joint7 は −rot 動く。置く所の joint7 は、置く所の基準から計算した値と一致するはず
            j_place = _wrist_angle(q_place_ref, y_place_ref, h + rot, lo, hi)
            if j_place is None or abs((j_pick - rot) - j_place) > np.pi / 2:
                continue  # この回し方では、手首を範囲内で回しきって置く姿勢にたどり着けない
            margin = float(min(j_pick - lo, hi - j_pick, j_place - lo, hi - j_place))
            notes.append(f"手の yaw {np.degrees(h):+.0f}° で掴み {np.degrees(rot):+.0f}° 回す: "
                         f"手首 {j_pick:+.2f} → {j_place:+.2f} rad (余裕 {margin:.2f})")
            if margin >= min_margin and (best is None or margin > best.wrist_margin):
                q_above = q_pick_ref.copy()
                q_above[6] = j_pick
                best = TransferPlan(pick_yaw=h, q_above=q_above, place_yaw=h + rot, wrist_margin=margin)
    if best is None:
        raise RuntimeError("手首が回りきる掴み方が見つからない\n  " + "\n  ".join(notes))
    return best


NARROW_OPEN_WIDTH = 0.04  # 壁際で使う狭い開き。器械の把持点の幅は最大 19mm なので、片側 1cm の余裕がある
FINGER_OUTSIDE = 0.017  # 指の内側の面から外側までの厚み + 余裕 [m]。実測: 5.5cm 開きで壁まで 3.7cm だと当たり、4.2cm なら当たらない


def open_width_for(grasp_pos: np.ndarray, yaw: float) -> float:
    """返却トレイの壁際では、指が壁に当たらないように開きを狭める。指は器械の長手方向 yaw と直交する向きに開く。"""
    cx, cy = scene.TRAY_CENTER
    hx, hy = scene.TRAY_INNER_HALF
    if abs(grasp_pos[0] - cx) > hx or abs(grasp_pos[1] - cy) > hy:
        return OPEN_WIDTH  # トレイの外 (器械台)
    n = np.array([-np.sin(yaw), np.cos(yaw)])  # 指が開く向き
    for w in (OPEN_WIDTH, NARROW_OPEN_WIDTH):
        reach = w / 2 + FINGER_OUTSIDE
        tips = [grasp_pos[:2] + s * reach * n for s in (-1, 1)]
        if all(abs(t[0] - cx) <= hx and abs(t[1] - cy) <= hy for t in tips):
            return w
    return NARROW_OPEN_WIDTH


@dataclass
class GraspResult:
    closed_width: float  # 閉じた後の指の間隔 [m]。ほぼ 0 なら「何も挟めていない」
    yaw_used: float


def finger_width(ctrl: ArmController) -> float:
    d = ctrl.data
    return float(d.joint("finger_joint1").qpos[0] + d.joint("finger_joint2").qpos[0])


def pick(ctrl: ArmController, grasp_pos: np.ndarray, yaw: float, plan: TransferPlan | None = None,
         open_width: float | None = None) -> GraspResult:
    """yaw = 器械の長手方向。plan を渡すと、その掴み方 (手の yaw と真上での関節角) を使う。
    渡さなければ yaw と yaw+π のうち今の手に近い方で掴む。open_width = 近づくときの指の開き (省略すると、壁際だけ狭める)。"""
    _, R_now = ctrl.commanded_tcp()
    yaw = plan.pick_yaw if plan else choose_grasp_yaw(yaw, R_now)
    R = grasp_rotation(yaw)
    above = grasp_pos + [0, 0, APPROACH_HEIGHT]

    ctrl.set_gripper(open_width if open_width is not None else open_width_for(grasp_pos, yaw))
    if plan:
        # 計画した腕の構えへ関節空間で直接動く (手先の直線移動だと、途中の IK が別の構えに流れることがある)
        ctrl.move_joints(plan.q_above, duration=1.6)
    else:
        ctrl.move_to(above, R, duration=1.6)
    ctrl.move_to(grasp_pos + [0, 0, -GRASP_DEPTH], R, duration=1.0)
    ctrl.hold(0.2)  # 降りきった後の揺れが収まるのを待つ
    ctrl.set_gripper(0.0)
    ctrl.hold(0.6)  # 指が閉じきって把持力が立ち上がるのを待つ
    width = finger_width(ctrl)
    ctrl.move_to(grasp_pos + [0, 0, LIFT_HEIGHT], R, duration=1.2)
    return GraspResult(closed_width=width, yaw_used=yaw)


def place(ctrl: ArmController, place_pos: np.ndarray, yaw: float,
          before_release: Callable[[], None] | None = None, yaw_from: float | None = None) -> None:
    """yaw = 置くときの手の yaw。
    yaw_from を渡すと (plan_transfer の結果)、運ぶ間に手を yaw_from → yaw へ、数値どおりの量・向きで回す。
    渡さなければ yaw と yaw+π のうち今の手に近い方を選ぶ (器械の前後を気にしない場合)。"""
    _, R_now = ctrl.commanded_tcp()
    if yaw_from is None:
        yaw = choose_grasp_yaw(yaw, R_now)
        ctrl.move_to(place_pos + [0, 0, LIFT_HEIGHT], grasp_rotation(yaw), duration=2.0)
    else:
        ctrl.move_to_turning(place_pos + [0, 0, LIFT_HEIGHT], yaw_from, yaw, duration=2.0)
    R = grasp_rotation(yaw)
    ctrl.move_to(place_pos + [0, 0, PLACE_CLEARANCE], R, duration=1.0)
    ctrl.hold(0.2)
    if before_release:
        before_release()  # 評価用フック (離す直前の器械の位置を測る)
    ctrl.set_gripper(OPEN_WIDTH)
    ctrl.hold(0.5)
    ctrl.move_to(place_pos + [0, 0, APPROACH_HEIGHT], R, duration=0.8)
