"""動作計画: 「手先をここからそこへ、こう動かす」をシミュレーションの各ステップの関節指令に変える。

構成
- grasp_rotation(): 器械の向き (yaw) → 上から掴むときの手の姿勢 (回転行列)
- ArmController: 手先の目標を受け取り、IK で関節指令 q_cmd を更新して ctrl に書く
- move_to / gripper などのプリミティブ: 直線補間した手先の軌道をたどる
"""

from collections.abc import Callable

import mujoco
import numpy as np

from control.ik import DiffIK, IKConfig
from sim import robot

CONTROL_DT = 0.01  # 制御周期 [s] (100 Hz)。物理は 2ms なので 5 ステップごとに指令を更新する


def grasp_rotation(yaw: float) -> np.ndarray:
    """器械の長手方向 yaw に対して、真上から挟むための手の回転行列 (列 = 手の x, y, z 軸)。

    Panda の指は手の y 軸方向に開閉する。器械を幅方向に挟みたいので:
      手の z 軸 = 真下 (0, 0, -1)        … 指が下を向く
      手の x 軸 = 器械の長手方向           … 指の板が器械に沿う
      手の y 軸 = z × x                  … 指が器械の幅方向に閉じる
    """
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def choose_grasp_yaw(yaw: float, current_rot: np.ndarray) -> float:
    """グリッパは 180° 回しても同じ形なので、yaw と yaw+π のうち今の手の向きに近い方を選ぶ。
    (手首 joint7 を無駄に大回転させない / 関節限界に当たりにくくする)"""
    best, best_ang = yaw, np.inf
    for cand in (yaw, yaw + np.pi):
        R = grasp_rotation(cand)
        ang = np.arccos(np.clip((np.trace(current_rot.T @ R) - 1) / 2, -1, 1))
        if ang < best_ang:
            best, best_ang = cand, ang
    return float(np.arctan2(np.sin(best), np.cos(best)))


def min_jerk(s: float) -> float:
    """0→1 の時間比 s を、始点・終点で速度も加速度も 0 になる滑らかな比率に変換する。
    急発進・急停止すると、掴んだ器械が慣性で滑るので、それを避けるため。"""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def slerp_rot(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """回転行列の球面線形補間 (向きを等角速度で回す)。"""
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.flatten())
    mujoco.mju_mat2Quat(q1, R1.flatten())
    if np.dot(q0, q1) < 0:
        q1 = -q1
    # 相対回転を回転ベクトルにして s 倍する
    q0inv, dq = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(q0inv, q0)
    mujoco.mju_mulQuat(dq, q1, q0inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    qs, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qs, vel / (np.linalg.norm(vel) + 1e-12), np.linalg.norm(vel) * s)
    mujoco.mju_mulQuat(out, qs, q0)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, out)
    return R.reshape(3, 3)


class ArmController:
    """手先目標 → IK → 関節指令。

    IK の初期値には「実際の関節角 qpos」ではなく「前回の指令 q_cmd」を使う。
    実際の角度はサーボの追従遅れでわずかに遅れているので、そこから IK を始めると
    目標に届いたことにならず、指令がずっと揺れ続けるため。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ik_cfg: IKConfig | None = None,
                 on_step: Callable[[], None] | None = None):
        self.model, self.data = model, data
        self.ik = DiffIK(model, ik_cfg)
        self.arm_act = np.array([model.actuator(a).id for a in robot.ARM_ACTUATORS])
        self.grip_act = model.actuator(robot.GRIPPER_ACTUATOR).id
        self.q_cmd = robot.arm_qpos(model, data)
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self.on_step = on_step  # 制御周期ごとに呼ばれるフック (描画・録画・ログ用)
        self.ik_iters = 3  # 制御周期ごとの IK 反復回数 (前回の解から始めるので少なくて良い)

    # --- 低レベル ---
    def tick(self, target_pos=None, target_rot=None) -> None:
        """1 制御周期: (目標があれば) IK で q_cmd を更新 → ctrl に書く → 物理を substeps 回進める。"""
        if target_pos is not None:
            for _ in range(self.ik_iters):
                self.q_cmd, _ = self.ik.step(self.q_cmd, target_pos, target_rot)
        self.data.ctrl[self.arm_act] = self.q_cmd
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.on_step:
            self.on_step()

    def commanded_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        """指令値 q_cmd での TCP (= IK 上の手先)。実際の手先とはサーボの誤差ぶん違う。"""
        return self.ik.fk(self.q_cmd)

    # --- プリミティブ ---
    def move_to(self, pos, rot, duration: float) -> None:
        """今の (指令上の) 手先から目標まで、直線 + min-jerk で動かす。"""
        p0, R0 = self.commanded_tcp()
        pos = np.asarray(pos, dtype=float)
        n = max(1, round(duration / CONTROL_DT))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            self.tick(p0 + s * (pos - p0), slerp_rot(R0, rot, s))

    def hold(self, duration: float, pos=None, rot=None) -> None:
        if pos is None:
            pos, rot = self.commanded_tcp()
        for _ in range(max(1, round(duration / CONTROL_DT))):
            self.tick(pos, rot)

    def set_gripper(self, opening: float) -> None:
        """両指の間隔 [m] (0〜0.08) を指令する。"""
        self.data.ctrl[self.grip_act] = np.clip(opening / 0.08, 0, 1) * robot.GRIPPER_OPEN

    def move_joints(self, q_target, duration: float) -> None:
        """関節空間で補間して動かす (待機姿勢への移動など、手先の経路を気にしない大移動用)。"""
        q0 = self.q_cmd.copy()
        n = max(1, round(duration / CONTROL_DT))
        for i in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(i / n) * (np.asarray(q_target) - q0)
            self.tick()
