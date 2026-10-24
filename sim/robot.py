"""Franka Panda (MuJoCo Menagerie) の読み込みと、よく使う名前・定数をまとめるモジュール。

Menagerie の MJCF は robot_descriptions が初回に git clone して
~/.cache/robot_descriptions/ にキャッシュする。取得する commit は
robot_descriptions のバージョンごとに固定されているので、uv.lock で
robot-descriptions のバージョンを固定すれば Panda のモデルも固定される。
"""

from pathlib import Path

import mujoco
import numpy as np

# 7 自由度アームの関節名 (根元 → 手先の順)
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
# アーム用アクチュエータ。Menagerie の Panda は「位置サーボ」型:
#   force = kp * (ctrl - q) - kd * qdot   (gainprm / biasprm がこの式を表す)
# つまり ctrl には「目標関節角 [rad]」を入れる。トルクを直接入れるのではない。
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
# グリッパ用アクチュエータ。ctrl 0 (閉) 〜 255 (開, 指 1 本あたり 4cm) にスケールされている。
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

# 手首フランジ (link7 の先) に付いた "hand" body の原点から、
# 指先の中心 (TCP: Tool Center Point) までの距離。Franka 公式値 0.1034 m。
# Menagerie の hand には TCP の site が無いので、ここで定数として持つ。
HAND_TO_TCP = 0.1034


def panda_dir() -> Path:
    """Menagerie の franka_emika_panda ディレクトリ (初回は clone が走る)。"""
    from robot_descriptions import panda_mj_description

    return Path(panda_mj_description.PACKAGE_PATH)


def load_panda_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """床とライト付きの scene.xml (panda.xml を include) を読み込む。

    Phase 1 以降は器械台などを足した独自シーンに置き換える。
    """
    model = mujoco.MjModel.from_xml_path(str(panda_dir() / "scene.xml"))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    return model, data


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """keyframe "home" (肘を曲げて手先を下に向けた姿勢) に qpos と ctrl を合わせる。"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # mj_forward: qpos から全 body の位置姿勢 (順運動学) や接触などを計算する。
    # 時間は進めない。描画や姿勢の読み出しの前に必要。
    mujoco.mj_forward(model, data)


def tcp_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """TCP の world 座標 (3,) と回転行列 (3, 3) を返す。

    hand の回転行列の第 3 列 (= hand の z 軸) が指の伸びる方向なので、
    そちらに HAND_TO_TCP だけ進んだ点を TCP とする。
    """
    hand = data.body("hand")
    rot = hand.xmat.reshape(3, 3)
    pos = hand.xpos + rot[:, 2] * HAND_TO_TCP
    return pos.copy(), rot.copy()


def arm_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """アーム 7 関節の現在角 [rad]。"""
    return np.array([data.joint(j).qpos[0] for j in ARM_JOINTS])
