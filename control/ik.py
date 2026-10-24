"""微分 IK (damped least squares, DLS)。

やりたいこと: 手先 (TCP) の目標位置・姿勢 → 7 関節の角度 q。

一発で解く式は無いので、「今の q で手先がどれだけ目標からずれているか (誤差 e)」を測り、
ヤコビアン J (関節を少し動かしたときの手先の動き, 6×7) を使って
誤差が減る方向に q を少しずつ動かす、を繰り返す:

    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e          … damped least squares
         └─ λ=0 なら普通の擬似逆行列 J⁺。特異姿勢 (腕が伸びきる等) で J Jᵀ が
            ほぼ特異になると Δq が爆発するので、λ² I を足して「ブレーキ」をかける。

Panda は 7 関節で、手先の自由度 (6) より 1 つ多い (冗長)。同じ手先姿勢を作る q が無数にあるので、
手先を動かさない方向 (零空間) を使って「なるべく home 姿勢に近い q」を選ぶ (posture 項)。
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sim import robot


@dataclass
class IKConfig:
    damping: float = 0.05  # λ。大きいほど特異点に強いが、収束が遅く誤差が残りやすい
    rot_weight: float = 0.5  # 姿勢誤差 [rad] を位置誤差 [m] に対してどれだけ重視するか
    max_step: float = 0.15  # 1 反復あたりの関節変化の上限 [rad] (大きな飛びを防ぐ)
    posture_gain: float = 0.05  # 零空間で home 姿勢に引き戻す強さ
    pos_tol: float = 5e-4  # 収束判定 [m]
    rot_tol: float = np.radians(0.5)  # 収束判定 [rad]


class DiffIK:
    def __init__(self, model: mujoco.MjModel, cfg: IKConfig | None = None, q_nominal: np.ndarray | None = None):
        self.model = model
        self.cfg = cfg or IKConfig()
        # IK 専用の MjData。シミュレーション本体の data を汚さずに「仮に q にしたら手先はどこか」を計算する
        self.data = mujoco.MjData(model)
        self.qadr = np.array([model.joint(j).qposadr[0] for j in robot.ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in robot.ARM_JOINTS])
        self.lo = np.array([model.joint(j).range[0] for j in robot.ARM_JOINTS])
        self.hi = np.array([model.joint(j).range[1] for j in robot.ARM_JOINTS])
        self.hand_id = model.body("hand").id
        self.q_nominal = robot.HOME_QPOS.copy() if q_nominal is None else q_nominal
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """順運動学: 関節角 q → TCP の (位置, 回転行列)。"""
        self.data.qpos[self.qadr] = q
        # mj_kinematics: 位置姿勢だけ計算 (mj_forward より軽い)。mj_comPos: ヤコビアン計算に必要な量
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return robot.tcp_pose(self.model, self.data)

    def jacobian(self, tcp_pos: np.ndarray) -> np.ndarray:
        """TCP 点のヤコビアン (6×7)。上 3 行 = 並進速度、下 3 行 = 角速度。fk() の直後に呼ぶ。"""
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, tcp_pos, self.hand_id)
        return np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])

    def error(self, pos, rot, target_pos, target_rot) -> np.ndarray:
        """6 次元の誤差 [位置 (m); 姿勢 (rad)]。姿勢誤差は「今の姿勢から目標姿勢への回転」の回転ベクトル。"""
        e_pos = target_pos - pos
        # R_err = R_target R_currentᵀ を回転ベクトル (軸 × 角度) にする
        q_cur, q_tgt, q_err = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q_cur, rot.flatten())
        mujoco.mju_mat2Quat(q_tgt, target_rot.flatten())
        mujoco.mju_negQuat(q_cur, q_cur)
        mujoco.mju_mulQuat(q_err, q_tgt, q_cur)
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)
        return np.concatenate([e_pos, e_rot])

    def step(self, q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """DLS を 1 反復。新しい q と、更新前の誤差 e を返す。"""
        c = self.cfg
        pos, rot = self.fk(q)
        e = self.error(pos, rot, target_pos, target_rot)
        J = self.jacobian(pos)

        W = np.diag([1, 1, 1, c.rot_weight, c.rot_weight, c.rot_weight])
        Jw, ew = W @ J, W @ e
        # Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e  (6×6 の連立方程式を解くだけなので軽い)
        JJt = Jw @ Jw.T + (c.damping**2) * np.eye(6)
        dq = Jw.T @ np.linalg.solve(JJt, ew)

        # 零空間射影 N = I − J⁺J: この方向に関節を動かしても手先は動かない。
        # そこで home 姿勢へ引き戻す成分だけを足す → 肘が変な向きに流れていくのを防ぐ。
        # J⁺ はダンピング無しの擬似逆行列を使う。ダンピング付きだと N が「正確な零空間」にならず、
        # 姿勢の引き戻しが手先にも漏れて、目標に 0.7mm ほど届かないまま止まってしまった。
        N = np.eye(7) - np.linalg.pinv(Jw, rcond=1e-4) @ Jw
        dq += N @ (c.posture_gain * (self.q_nominal - q))

        # 1 反復の変化量を制限 (誤差が大きいときに関節が一気に飛ばないように)
        norm = np.linalg.norm(dq)
        if norm > c.max_step:
            dq *= c.max_step / norm
        q_new = np.clip(q + dq, self.lo, self.hi)
        return q_new, e

    def solve(self, q_init, target_pos, target_rot, iters: int = 200) -> tuple[np.ndarray, bool, np.ndarray]:
        """収束するまで反復する (ウェイポイントを事前に解くとき用)。戻り値: (q, 収束したか, 最終誤差)"""
        q = np.array(q_init, dtype=float)
        for _ in range(iters):
            q, e = self.step(q, target_pos, target_rot)
            if np.linalg.norm(e[:3]) < self.cfg.pos_tol and np.linalg.norm(e[3:]) < self.cfg.rot_tol:
                return q, True, e
        pos, rot = self.fk(q)
        e = self.error(pos, rot, target_pos, target_rot)
        ok = np.linalg.norm(e[:3]) < self.cfg.pos_tol and np.linalg.norm(e[3:]) < self.cfg.rot_tol
        return q, ok, e
