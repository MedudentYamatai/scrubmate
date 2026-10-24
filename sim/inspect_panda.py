"""Phase 0: Panda モデルの中身を表示し、オフスクリーン描画で PNG を保存する。

ビューア (GUI) 無しで動くので、通常の python で実行できる:
    uv run python -m sim.inspect_panda
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim import robot

OUT_DIR = Path("outputs/phase0")


def print_structure(model: mujoco.MjModel) -> None:
    # nq: 一般化座標 (qpos) の次元, nv: 速度 (qvel) の次元, nu: アクチュエータ数。
    # ヒンジ関節だけなら nq == nv。free joint (6 自由度の物体) があると nq = nv + 1 (クォータニオン分)。
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  ngeom={model.ngeom}")
    print(f"timestep={model.opt.timestep} s  integrator={mujoco.mjtIntegrator(model.opt.integrator).name}")

    print("\n[joints]  name / type / range")
    for j in range(model.njnt):
        jnt = model.joint(j)
        jtype = mujoco.mjtJoint(jnt.type[0]).name
        rng = jnt.range if model.jnt_limited[j] else "unlimited"
        print(f"  {jnt.name:14s} {jtype:12s} {rng}")

    print("\n[actuators]  name / target / ctrlrange / kp")
    for a in range(model.nu):
        act = model.actuator(a)
        trn_type = mujoco.mjtTrn(act.trntype[0])
        if trn_type == mujoco.mjtTrn.mjTRN_TENDON:
            target = "tendon:" + model.tendon(act.trnid[0]).name
        else:
            target = model.joint(act.trnid[0]).name
        print(f"  {act.name:10s} {target:14s} {act.ctrlrange}  kp={act.gainprm[0]:.4g}")

    print("\n[bodies] (親 → 子)")
    for b in range(1, model.nbody):
        body = model.body(b)
        parent = model.body(body.parentid[0]).name or "world"
        print(f"  {parent:12s} → {body.name}")


def print_fk(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    np.set_printoptions(precision=4, suppress=True)
    print("\n[forward kinematics @ home]")
    print("  arm qpos [rad] :", robot.arm_qpos(model, data))
    pos, rot = robot.tcp_pose(model, data)
    print("  TCP position [m]:", pos)
    # rot の各列 = TCP 座標系の x, y, z 軸を world で表したもの。
    # z 列が (0, 0, -1) なら「指先が真下を向いている」。
    print("  TCP z-axis      :", rot[:, 2], "(真下なら [0 0 -1])")
    finger_gap = data.joint("finger_joint1").qpos[0] + data.joint("finger_joint2").qpos[0]
    print(f"  finger gap [m]  : {finger_gap:.4f}")


def render_png(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    # Renderer はウィンドウを作らずに OpenGL コンテキストを確保して描画する (オフスクリーン)。
    # Mac では CGL が使われるので mjpython は不要。動画書き出しもこの仕組みを使う。
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation, cam.distance = 135, -25, 1.8
        cam.lookat[:] = [0.3, 0.0, 0.4]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"\nsaved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    model, data = robot.load_panda_scene()
    print("MJCF:", robot.panda_dir() / "scene.xml")
    print_structure(model)
    print_fk(model, data)
    if not args.no_render:
        render_png(model, data, OUT_DIR / "panda_home.png")


if __name__ == "__main__":
    main()
