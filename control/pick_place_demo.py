"""Phase 1: 真値を使った pick & place の試作と、把持の安定性の計測。

各器械について 器械台 → 返却トレイ → 器械台 (元のスロット) と往復させ、
持ち上げ成功・把持中の滑り・置き位置の誤差・隣の器械を乱したか を測る。

    uv run python -m control.pick_place_demo                      # 5 本全部、計測のみ (速い)
    uv run python -m control.pick_place_demo --video              # outputs/phase1/pick_place.mp4 も書き出す
    uv run python -m control.pick_place_demo --only scalpel --video
    ./scripts/mjpython.sh -m control.pick_place_demo --viewer     # ビューアで実時間再生

注意: ここでは把持目標に「シミュレーターの真値」を使っている (Phase 1 だけの例外)。
Phase 3 でカメラ画像からの推定に置き換える。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from control.motion import ArmController
from control.pick_place import finger_width, pick, place
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES
from sim.render import Recorder

OUT_DIR = Path("outputs/phase1")


def rel_pose_in_hand(model, data, kind) -> tuple[np.ndarray, float]:
    """器械の重心を TCP 座標系で表した位置と、TCP から見た器械の yaw。
    把持中にこれが変化したら「手の中で器械が滑った/回った」ということ。"""
    tcp, R = robot.tcp_pose(model, data)
    body = data.body(kind)
    p = R.T @ (body.xpos - tcp)
    ax = R.T @ body.xmat.reshape(3, 3)[:, 0]
    return p, float(np.arctan2(ax[1], ax[0]))


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def run_cycle(ctrl: ArmController, kind: str, src, dst, log: list) -> dict:
    model, data = ctrl.model, ctrl.data
    others = [k for k in INSTRUMENT_NAMES if k != kind]
    others_before = {k: data.body(k).xpos.copy() for k in others}

    # ★ 真値から把持目標を取る (Phase 3 で知覚に置き換える箇所)
    grasp_pos, yaw = scene.instrument_pose(model, data, kind)

    g = pick(ctrl, grasp_pos, yaw)
    ctrl.hold(0.3)  # 動作直後はサーボが追いついていない (動いている最中は最大 2cm 遅れる) ので、静止させてから測る
    lifted_z = float(data.body(kind).xpos[2])
    lifted = lifted_z > scene.INSTRUMENT_Z + 0.10
    rel_after_lift = rel_pose_in_hand(model, data, kind)
    # 掴んだ瞬間に手の中心から器械の重心がどれだけずれているか (xy のみ)
    grasp_offset_mm = float(np.linalg.norm(rel_after_lift[0][:2]) * 1000)
    # 静止時の 実際の TCP と 指令上の TCP (IK の結果) の差 = サーボの定常誤差 (重力補償があればほぼ 0)
    tcp_real, _ = robot.tcp_pose(model, data)
    tcp_cmd, _ = ctrl.commanded_tcp()
    tcp_err_mm = float(np.linalg.norm(tcp_real - tcp_cmd) * 1000)

    rel_before_release = {}

    def measure():
        rel_before_release["v"] = rel_pose_in_hand(model, data, kind)

    place_pos = np.array([dst[0], dst[1], scene.INSTRUMENT_Z])
    place(ctrl, place_pos, 0.0, before_release=measure)
    ctrl.hold(0.6)  # 置いた器械が落ち着くまで

    p_rel0, y_rel0 = rel_after_lift
    p_rel1, y_rel1 = rel_before_release["v"]
    final_pos, final_yaw = scene.instrument_pose(model, data, kind)
    res = {
        "instrument": kind,
        "from": src,
        "to": dst,
        "closed_width_mm": round(g.closed_width * 1000, 2),
        "lifted": bool(lifted),
        "grasp_offset_mm": round(grasp_offset_mm, 2),
        "tcp_static_err_mm": round(tcp_err_mm, 2),
        "slip_mm": round(float(np.linalg.norm(p_rel1 - p_rel0) * 1000), 2),
        "slip_deg": round(float(np.degrees(abs(wrap(y_rel1 - y_rel0)))), 2),
        "place_err_mm": round(float(np.linalg.norm(final_pos[:2] - place_pos[:2]) * 1000), 2),
        "place_yaw_err_deg": round(float(np.degrees(abs(wrap(final_yaw - 0.0)))), 2),
        "neighbor_disturb_mm": round(max((float(np.linalg.norm(data.body(k).xpos - others_before[k]) * 1000)
                                          for k in others), default=0.0), 2),
    }
    # 成功の定義: 持ち上がった・置き位置誤差 1cm 以内・向きの誤差 10° 以内・隣を 5mm 以上動かしていない
    res["success"] = bool(res["lifted"] and res["place_err_mm"] < 10 and res["place_yaw_err_deg"] < 10
                          and res["neighbor_disturb_mm"] < 5)
    # 失敗理由 (成功しなかったときだけ表示)。持ち上がらなかった場合、slip / place_err は
    # 「手と、器械台に残された器械との距離」になるので、滑りや置き誤差としては意味を持たない
    reasons = []
    if not res["lifted"]:
        reasons.append("持ち上がらず器械は元の場所に残った (slip/place_err は無意味)")
    elif res["place_err_mm"] >= 10:
        reasons.append("置き位置が 1cm 以上ずれた")
    if res["lifted"] and res["place_yaw_err_deg"] >= 10:
        reasons.append("向きが 10° 以上ずれた")
    if res["neighbor_disturb_mm"] >= 5:
        reasons.append("隣の器械を動かした")
    log.append(res)
    flag = "OK " if res["success"] else "NG "
    print(f"{flag}{kind:14s} {src}→{dst}  lifted={res['lifted']}  width={res['closed_width_mm']:.1f}mm  "
          f"offset={res['grasp_offset_mm']:.1f}mm  slip={res['slip_mm']:.1f}mm/{res['slip_deg']:.1f}°  "
          f"place_err={res['place_err_mm']:.1f}mm/{res['place_yaw_err_deg']:.1f}°  "
          f"neighbor={res['neighbor_disturb_mm']:.1f}mm  tcp_err={res['tcp_static_err_mm']:.1f}mm", flush=True)
    for r in reasons:
        print(f"     └ 失敗理由: {r}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=INSTRUMENT_NAMES, help="1 本だけ試す")
    ap.add_argument("--video", action="store_true", help="overview_cam で mp4 を書き出す")
    ap.add_argument("--viewer", action="store_true", help="ビューアで実時間表示 (mjpython 必須)")
    ap.add_argument("--no-gravcomp", action="store_true", help="重力補償を切る (比較実験用)")
    ap.add_argument("--grip-scale", type=float, default=scene.GRIPPER_FORCE_SCALE, help="把持力の倍率")
    args = ap.parse_args()

    model, data = scene.build(scene.SceneOptions(gravcomp=not args.no_gravcomp,
                                                 gripper_force_scale=args.grip_scale))
    # 器械を支持バーの上に落ち着かせる
    for _ in range(300):
        mujoco.mj_step(model, data)

    hooks = []
    rec = None
    if args.video:
        name = f"pick_place_{args.only}.mp4" if args.only else "pick_place.mp4"
        rec = Recorder(model, OUT_DIR / name)
        hooks.append(lambda: rec.maybe_capture(data))
    viewer = None
    if args.viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except RuntimeError as e:
            sys.exit(f"{e}\n→ `./scripts/mjpython.sh -m control.pick_place_demo --viewer` で起動してください")
        wall0, sim0 = time.time(), data.time

        def sync():
            viewer.sync()
            # 実時間に合わせる
            lag = (data.time - sim0) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
            if not viewer.is_running():
                raise SystemExit

        hooks.append(sync)

    ctrl = ArmController(model, data, on_step=lambda: [h() for h in hooks])
    kinds = [args.only] if args.only else [scene.DEFAULT_LAYOUT[s] for s in sorted(scene.DEFAULT_LAYOUT)]
    slot_of = {v: k for k, v in scene.DEFAULT_LAYOUT.items()}

    log: list = []
    t0 = time.time()
    try:
        for kind in kinds:
            table_xy = scene.TABLE_SLOTS[slot_of[kind]]
            tray_xy = scene.TRAY_SLOTS[1]
            run_cycle(ctrl, kind, f"table{slot_of[kind]}", tray_xy, log)
            run_cycle(ctrl, kind, "tray1", table_xy, log)
        ctrl.move_joints(robot.HOME_QPOS, duration=1.5)
    finally:
        if rec:
            rec.close()
        if viewer:
            viewer.close()

    n_ok = sum(r["success"] for r in log)
    print(f"\n成功 {n_ok}/{len(log)}  (sim {data.time:.1f}s / wall {time.time() - t0:.1f}s)")
    for key in ("grasp_offset_mm", "slip_mm", "slip_deg", "place_err_mm", "place_yaw_err_deg", "tcp_static_err_mm"):
        v = np.array([r[key] for r in log])
        print(f"  {key:20s} mean={v.mean():6.2f}  max={v.max():6.2f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("pick_place_metrics.json" if not args.only else f"pick_place_metrics_{args.only}.json")
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
