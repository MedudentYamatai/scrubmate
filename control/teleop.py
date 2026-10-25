"""遠隔操作: 手先の「目標」を動かすと、IK が 7 関節を自動で計算して腕がついてくる。

    ./scripts/mjpython.sh -m control.teleop

目標はオレンジの球 + 3 色の軸 (赤 x / 緑 y = 指が閉じる方向 / 青 z = 指の向き)。
動かし方は 2 通り:
  (1) ターミナルでキー入力 (このスクリプトを起動したターミナルをアクティブにして押す)
        w/s: 前/後 (x)   a/d: 左/右 (y)   r/f: 上/下 (z)   q/e: 手首を左/右に回す (yaw)
        space: グリッパ開閉   1/2/3: 移動量 1mm/5mm/20mm   h: home に戻る   z: シーンを初期化   x: 終了
  (2) ビューアでマウス: オレンジの球をダブルクリックで選択 → Ctrl+右ドラッグで移動 / Ctrl+左ドラッグで回転
※ ビューア側のキーは英字すべてに表示切替が割り当てられているため、キー操作はターミナルで受ける。
   ビューアの Backspace (リセット) は使わず、ターミナルの z を使うこと。

ターミナルには 目標と実際の手先の差 (IK が届いているか) と、特異点への近さを表示する。
"""

import queue
import shutil
import sys
import termios
import threading
import time
import tty
import unicodedata

import mujoco
import mujoco.viewer
import numpy as np

from control.motion import CONTROL_DT, ArmController
from sim import robot, scene

STEP_SIZES = {"1": 0.001, "2": 0.005, "3": 0.02}
YAW_STEP = np.radians(5)
MOVE_KEYS = {
    "w": (1, 0, 0), "s": (-1, 0, 0),
    "a": (0, 1, 0), "d": (0, -1, 0),
    "r": (0, 0, 1), "f": (0, 0, -1),
}


def key_reader(q: queue.Queue, stop: threading.Event) -> None:
    """ターミナルから 1 文字ずつ読む (端末は main で cbreak モード = Enter 不要 にしてある)。"""
    while not stop.is_set():
        ch = sys.stdin.read(1)
        if not ch:
            break
        q.put(ch)
        if ch == "x":
            break


def quat_from_rot(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def fit_line(text: str) -> str:
    """端末の幅に収まるよう切り詰める。はみ出すと折り返され、\r で行頭に戻れず表示が重なっていくため。
    日本語や記号は 2 マスぶん使うので、文字数ではなく表示幅で数える。"""
    width = shutil.get_terminal_size().columns - 1
    out, used = "", 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
        if used + w > width:
            break
        out, used = out + ch, used + w
    return out


def finger_gap_mm(data: mujoco.MjData) -> float:
    return 1000 * float(data.joint("finger_joint1").qpos[0] + data.joint("finger_joint2").qpos[0])


def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("ターミナルから起動してください (キー入力を受け取るため)")
    model, data = scene.build(scene.SceneOptions(teleop_target=True))
    for _ in range(300):  # 器械を落ち着かせる
        mujoco.mj_step(model, data)
    mocap_id = model.body("teleop_target").mocapid[0]

    def target_to_tcp():
        pos, R = robot.tcp_pose(model, data)
        data.mocap_pos[mocap_id] = pos
        data.mocap_quat[mocap_id] = quat_from_rot(R)

    target_to_tcp()
    mujoco.mj_forward(model, data)
    init_state = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                  data.mocap_pos.copy(), data.mocap_quat.copy())

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        sys.exit(f"{e}\n→ `./scripts/mjpython.sh -m control.teleop` で起動してください")

    keys: queue.Queue = queue.Queue()
    stop = threading.Event()
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    threading.Thread(target=key_reader, args=(keys, stop), daemon=True).start()
    try:
        run_loop(model, data, viewer, keys, stop, mocap_id, init_state, target_to_tcp)
    finally:
        # 読み取りスレッドは daemon なので途中で消える。端末の設定は必ずここで元に戻す
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        stop.set()
        print()


def run_loop(model, data, viewer, keys, stop, mocap_id, init_state, target_to_tcp) -> None:
    ctrl = ArmController(model, data)
    step = STEP_SIZES["2"]
    gripper_open = True
    last_print = 0.0
    print(__doc__.split("\n\n")[2])  # 操作方法を表示

    with viewer:
        while viewer.is_running() and not stop.is_set():
            t_start = time.time()
            # --- キー入力を目標に反映 ---
            while not keys.empty():
                ch = keys.get()
                if ch in MOVE_KEYS:
                    data.mocap_pos[mocap_id] += step * np.array(MOVE_KEYS[ch])
                elif ch in ("q", "e"):
                    # world の z 軸まわりに目標の向きを回す (指の向き = 真下 は保ったまま)
                    dq = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(dq, np.array([0, 0, 1.0]), YAW_STEP if ch == "q" else -YAW_STEP)
                    out = np.zeros(4)
                    mujoco.mju_mulQuat(out, dq, data.mocap_quat[mocap_id])
                    data.mocap_quat[mocap_id] = out
                elif ch == " ":
                    gripper_open = not gripper_open
                    ctrl.set_gripper(0.08 if gripper_open else 0.0)
                elif ch in STEP_SIZES:
                    step = STEP_SIZES[ch]
                elif ch == "h":
                    ctrl.on_step = lambda: (viewer.sync(), time.sleep(CONTROL_DT))
                    ctrl.move_joints(robot.HOME_QPOS, duration=1.5)
                    ctrl.on_step = None
                    target_to_tcp()
                elif ch == "z":
                    qpos, qvel, c, mp, mq = init_state
                    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, c
                    data.mocap_pos[:], data.mocap_quat[:] = mp, mq
                    mujoco.mj_forward(model, data)
                    ctrl.q_cmd = robot.arm_qpos(model, data)
                    gripper_open = True
                elif ch == "x":
                    stop.set()

            # --- 目標 (キー or マウスで動いた mocap) に向けて IK → 関節指令 ---
            tgt_pos = data.mocap_pos[mocap_id].copy()
            tgt_rot = rot_from_quat(data.mocap_quat[mocap_id])
            with viewer.lock():
                ctrl.tick(tgt_pos, tgt_rot)
            viewer.sync()

            # --- 状態表示 ---
            if time.time() - last_print > 0.2:
                last_print = time.time()
                cmd_pos, cmd_rot = ctrl.commanded_tcp()
                e = ctrl.ik.error(cmd_pos, cmd_rot, tgt_pos, tgt_rot)
                # ヤコビアンの最小特異値: 0 に近いほど「その方向に手先を動かせない」= 特異点に近い
                J = ctrl.ik.jacobian(cmd_pos)
                sv_min = np.linalg.svd(J[:3], compute_uv=False).min()
                warn = "  ⚠届いていない" if np.linalg.norm(e[:3]) > 0.01 else ""
                line = (f"目標 {np.round(tgt_pos, 3)}  IK誤差 {np.linalg.norm(e[:3]) * 1000:.1f}mm "
                        f"{np.degrees(np.linalg.norm(e[3:])):.1f}°  特異値min {sv_min:.3f}  "
                        f"指 {finger_gap_mm(data):.0f}mm  step {step * 1000:.0f}mm{warn}")
                # \033[K = カーソルから行末までを消す (前回の表示の残りを消すため)
                sys.stdout.write("\r" + fit_line(line) + "\033[K")
                sys.stdout.flush()

            dt = CONTROL_DT - (time.time() - t_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
