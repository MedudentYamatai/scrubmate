"""手術室シーン = or_scene.xml (固定物) + Panda (Menagerie) + 器械 5 本 を MjSpec で組み立てる。

Menagerie の panda.xml は書き換えずに、プログラムから「部品として取り付ける (attach)」。
こうすると Menagerie 側の更新と自分のシーンが混ざらず、どこを変えたかがこのファイルに集約される。
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sim import robot
from sim.instruments import INSTRUMENT_NAMES, add_instrument

ASSET_DIR = Path(__file__).parent / "assets"
ROBOT_BASE_POS = (0.0, 0.0, 0.6)  # 台座の上面

# 支持バー上面の高さ。器械は厚さ 1cm なので、重心 (= 把持点) はその 5mm 上
BAR_TOP_Z = 0.75
INSTRUMENT_Z = BAR_TOP_Z + 0.005

# 器械台の固定 5 スロット (把持点の xy)。器械の長手方向は world の x 軸 (yaw = 0)
TABLE_SLOTS = {i: (0.48, y) for i, y in enumerate([0.18, 0.09, 0.0, -0.09, -0.18])}
# 初期配置: スロット番号 → 器械名
DEFAULT_LAYOUT = {0: "scalpel", 1: "needle_holder", 2: "forceps", 3: "scissors", 4: "pean"}
# 返却トレイ (器械台の術野側の端) の中心と、トレイ内の置き場 (3 箇所)
TRAY_CENTER = (0.48, 0.38)
TRAY_INNER_HALF = (0.112, 0.102)  # 壁の内側の半分の寸法 (x, y) [m]
# 置き場の間隔は 6cm。最初は 7.5cm だったが、端の置き場は壁まで 2.7cm しかなく、指が壁に当たって 5 種類とも掴めなかった
# (Phase 5 で発見。Phase 3 の pick & place は中央の置き場しか使っていなかった)。指 5.5cm 開きで掴めるのは壁まで 4.2cm 以上
TRAY_SLOTS = {i: (0.48, TRAY_CENTER[1] + dy) for i, dy in enumerate([-0.06, 0.0, 0.06])}

# アームのリンク (重力補償をかける対象)
ARM_BODIES = [f"link{i}" for i in range(1, 8)] + ["hand", "left_finger", "right_finger"]

# グリッパの把持力の倍率。Menagerie の設定のままだと、幅 2cm の物を挟んだときの
# 指の力が約 0.5N しか出ず、60g の器械を摩擦で持ち上げるのにほぼ足りない。
# 実機の Franka Hand は最大 70N なので、20 倍 (≈10N/指) にしても非現実的ではない。
GRIPPER_FORCE_SCALE = 20.0


@dataclass
class SceneOptions:
    gravcomp: bool = True
    gripper_force_scale: float = GRIPPER_FORCE_SCALE
    layout: dict | None = None  # None なら DEFAULT_LAYOUT
    teleop_target: bool = False  # 遠隔操作用の目標マーカー (mocap) を置くか


def build_spec(opts: SceneOptions | None = None) -> mujoco.MjSpec:
    opts = opts or SceneOptions()
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "or_scene.xml"))

    # --- Panda を取り付ける ---
    panda = mujoco.MjSpec.from_file(str(robot.panda_dir() / "panda.xml"))
    # Panda の keyframe は「Panda だけの qpos (9 次元)」なので、器械の freejoint を足すと
    # 次元が合わなくなる。消しておき、初期姿勢は robot.reset_to_home で関節名指定で入れる。
    for key in list(panda.keys):
        panda.delete(key)
    frame = spec.worldbody.add_frame(pos=list(ROBOT_BASE_POS))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attaching a spec with an empty prefix")
        spec.attach(panda, frame=frame, prefix="")

    for name in ARM_BODIES:
        # gravcomp=1: その body にかかる重力と同じ大きさの上向きの力を加える (= 重力補償)。
        # Phase 0 の実験で見た「サーボが重力に負けて垂れる誤差」がほぼ消える。
        spec.body(name).gravcomp = 1.0 if opts.gravcomp else 0.0

    grip = spec.actuator(robot.GRIPPER_ACTUATOR)
    k = opts.gripper_force_scale
    # force = gain*ctrl + bias1*length + bias2*velocity。3 つ同じ倍率なので目標開き幅は変わらず、硬さだけ k 倍
    grip.gainprm[0] *= k
    grip.biasprm[1] *= k
    grip.biasprm[2] *= k

    # --- 器械 ---
    layout = opts.layout if opts.layout is not None else DEFAULT_LAYOUT
    for slot, kind in layout.items():
        x, y = TABLE_SLOTS[slot]
        add_instrument(spec, kind, pos=(x, y, INSTRUMENT_Z + 0.001), yaw=0.0)

    # --- 術者の手と器械をつなぐ weld (受け渡し用、初期は無効) ---
    for kind in layout.values():
        spec.add_equality(
            name=f"handoff_{kind}",
            type=mujoco.mjtEq.mjEQ_WELD,
            name1="surgeon_hand",
            name2=kind,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=False,
        )

    if opts.teleop_target:
        tgt = spec.worldbody.add_body(name="teleop_target", mocap=True, pos=[0.5, 0, 1.0])
        vis = dict(contype=0, conaffinity=0, group=1)
        tgt.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], rgba=[1, 0.6, 0.1, 0.5], **vis)
        # 目標姿勢の軸を色で示す (赤 = x: 指の幅方向と直交 / 緑 = y: 指が閉じる方向 / 青 = z: 指が伸びる方向)
        for axis, rgba in zip(np.eye(3), ([1, 0, 0, 0.8], [0, 1, 0, 0.8], [0, 0, 1, 0.8])):
            tgt.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *(axis * 0.06)],
                         size=[0.003, 0, 0], rgba=rgba, **vis)
    return spec


def build(opts: SceneOptions | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_spec(opts)
    with warnings.catch_warnings():
        # prefix="" で attach すると「XML 保存時に名前が衝突しうる」警告が出る。保存はしないので無視
        warnings.filterwarnings("ignore", message="Attaching a spec with an empty prefix")
        model = spec.compile()
    data = mujoco.MjData(model)
    robot.reset_to_home(model, data)
    return model, data


def hand_grab(model: mujoco.MjModel, data: mujoco.MjData, kind: str) -> None:
    """術者の手が器械を持つ: 今の手と器械の位置関係のまま weld で固定する (術者はスクリプトなので拘束で代用してよい)。
    weld の eq_data は [anchor (3), 器械の位置 (手の座標系, 3), 器械の向き (手から見た quat, 4), torquescale, …]。
    anchor は MjSpec の既定値が (0, 1, 0) になっていて、そのままだと器械が 1m ずれた所へ引っ張られた → 0 にする。"""
    eq = model.equality(f"handoff_{kind}").id
    hand, inst = data.body("surgeon_hand"), data.body(kind)
    R = hand.xmat.reshape(3, 3)
    q_inv, rel_q = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(q_inv, hand.xquat)
    mujoco.mju_mulQuat(rel_q, q_inv, inst.xquat)
    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = R.T @ (inst.xpos - hand.xpos)
    model.eq_data[eq, 6:10] = rel_q
    data.eq_active[eq] = 1


def hand_release(model: mujoco.MjModel, data: mujoco.MjData, kind: str) -> None:
    data.eq_active[model.equality(f"handoff_{kind}").id] = 0


def instrument_pose(model: mujoco.MjModel, data: mujoco.MjData, kind: str) -> tuple[np.ndarray, float]:
    """器械の真値 (把持点の world 座標, 長手方向の yaw [rad])。ログ・評価・Phase 1 の試作専用。"""
    body = data.body(kind)
    axis = body.xmat.reshape(3, 3)[:, 0]
    return body.xpos.copy(), float(np.arctan2(axis[1], axis[0]))
