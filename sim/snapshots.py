"""シーンの確認用の静止画を outputs/phase1/ に書き出す (配置を変えたら撮り直す)。

    uv run python -m sim.snapshots

  overview.png      … 動画用の全体カメラ (器械台・ロボット・術野・術者の手)
  detect_home.png   … 検出カメラ、腕が home 姿勢 (腕が器械を隠す)
  detect_parked.png … 検出カメラ、腕が待避姿勢 (器械台と返却トレイが全部見える。検出・カウントはこの姿勢で撮る)
"""

from pathlib import Path

import mujoco

from sim import robot, scene
from sim.render import snapshot

OUT_DIR = Path("outputs/phase1")


def main() -> None:
    model, data = scene.build()
    for _ in range(300):  # 器械を支持バーの上に落ち着かせる
        mujoco.mj_step(model, data)
    snapshot(model, data, OUT_DIR / "overview.png", "overview_cam")
    snapshot(model, data, OUT_DIR / "detect_home.png", "detect_cam", width=1024, height=768)
    robot.set_arm_qpos(model, data, robot.PARK_QPOS)
    mujoco.mj_forward(model, data)
    snapshot(model, data, OUT_DIR / "detect_parked.png", "detect_cam", width=1024, height=768)
    print(f"saved {OUT_DIR}/overview.png, detect_home.png, detect_parked.png")


if __name__ == "__main__":
    main()
