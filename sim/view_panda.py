"""Phase 0: Panda をビューアで表示し、関節を正弦波で動かす。

Mac では passive viewer (自前のループで物理を回しつつ描画する方式) は
macOS の「GUI はメインスレッドで」という制約のため mjpython でしか動かない。
uv の Python では libpython の参照を補う必要があるので、ラッパ経由で起動する:
    ./scripts/mjpython.sh -m sim.view_panda
    ./scripts/mjpython.sh -m sim.view_panda --static   # 動かさずに眺めるだけ

操作: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム /
      ダブルクリックで body 選択 → Ctrl+右ドラッグで力を加える / Space=一時停止
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim import robot

FREQ_HZ = 0.25  # 正弦波の周波数 (4 秒で 1 往復)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="関節指令を変えず、home 姿勢を保持するだけ")
    parser.add_argument("--amp", type=float, default=0.4, help="正弦波の振幅 [rad]")
    parser.add_argument("--duration", type=float, default=0.0, help="秒数で自動終了 (0 なら手動で閉じるまで)")
    parser.add_argument("--kp-scale", type=float, default=1.0, help="アームのサーボゲイン (kp, kd) の倍率")
    args = parser.parse_args()

    model, data = robot.load_panda_scene()
    arm_act = [model.actuator(name).id for name in robot.ARM_ACTUATORS]
    # gainprm[0] = kp、biasprm[1] = -kp、biasprm[2] = -kd。3 つ同時に倍率をかけて PD ゲインを変える。
    model.actuator_gainprm[arm_act, 0] *= args.kp_scale
    model.actuator_biasprm[arm_act, 1:3] *= args.kp_scale
    grip_act = model.actuator(robot.GRIPPER_ACTUATOR).id
    q_home = data.ctrl[arm_act].copy()
    # 関節ごとに位相をずらして、動きが分かりやすいようにする
    phases = np.linspace(0, np.pi, len(arm_act))

    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        # Mac で普通の python から呼ぶと「mjpython で実行せよ」という RuntimeError になる
        sys.exit(f"{e}\n→ `./scripts/mjpython.sh -m sim.view_panda` で起動してください")

    with viewer_ctx as viewer:
        t0 = time.time()
        sq_err, n_err = 0.0, 0
        while viewer.is_running():
            step_start = time.time()
            t = data.time
            if not args.static:
                # ctrl は「目標関節角」。物理エンジン側の PD サーボがそこへ追従するトルクを出す。
                # 指令 (ctrl) と実際の角度 (qpos) の差が、サーボの追従遅れとして見える。
                data.ctrl[arm_act] = q_home + args.amp * np.sin(2 * np.pi * FREQ_HZ * t + phases)
                # グリッパは 2 秒ごとに開閉
                data.ctrl[grip_act] = robot.GRIPPER_OPEN if int(t / 2) % 2 == 0 else robot.GRIPPER_CLOSED

            # 1 ステップ = model.opt.timestep (0.002 s) だけ時間を進める
            mujoco.mj_step(model, data)
            viewer.sync()
            # 追従誤差 (指令角 − 実際の角) を記録。最初の 1 秒は過渡応答なので除く
            if data.time > 1.0:
                sq_err += float(np.sum((data.ctrl[arm_act] - robot.arm_qpos(model, data)) ** 2))
                n_err += len(arm_act)

            if args.duration and time.time() - t0 > args.duration:
                break
            # 実時間に合わせて待つ (シミュレーションの方が速いので)
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    if n_err:
        print(f"kp_scale={args.kp_scale}  追従誤差 RMS = {np.degrees(np.sqrt(sq_err / n_err)):.3f} deg")


if __name__ == "__main__":
    main()
