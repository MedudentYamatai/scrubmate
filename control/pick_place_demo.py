"""pick & place の実行と、把持の安定性の計測 (Phase 1 で作り、Phase 3 で知覚ベースにした)。

各器械について 器械台 → 返却トレイ → 器械台 (元のスロット) と往復させ、
持ち上げ成功・把持中の滑り・置き位置の誤差・隣の器械を乱したか を測る。
器械台には先端を +x (yaw = 0) に揃えて置く (先端の向きの推定が正しいかの確認を兼ねる)。
返却トレイの中の向きは問わない (実際のシナリオでトレイに置くのは術者で、向きはばらばら)。

    uv run python -m control.pick_place_demo                      # 知覚ベース、5 本全部、計測のみ
    uv run python -m control.pick_place_demo --random 1           # 配置をランダムに (乱数の種 1)。先端が逆向きの器械も混ざる
    uv run python -m control.pick_place_demo --trials 30          # ランダム配置 30 通り (300 回の pick & place) をまとめて集計
    uv run python -m control.pick_place_demo --video              # outputs/phase3/pick_place.mp4 も書き出す
    uv run python -m control.pick_place_demo --gt                 # Phase 1 と同じく真値で掴む (比較用)
    ./scripts/mjpython.sh -m control.pick_place_demo --viewer     # ビューアで実時間再生

知覚ベース (既定) では、掴む前に毎回 腕を待避 → 検出カメラで撮影 → 把持点と向きを推定 (perception/locate.py)。
真値 (scene.instrument_pose) は --gt のときの把持目標と、結果の評価にだけ使う。
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
from control.pick_place import pick, place, plan_transfer
from perception.gen_dataset import yaw_quat
from perception.locate import Locator
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES
from sim.render import Recorder

OUT_DIR = Path("outputs/phase3")
TABLE_YAW = 0.0  # 器械台に置くときの器械の先端の向き
TRAY_YAWS = (0.0, np.pi)  # 返却トレイではどちら向きでもよい


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


def run_cycle(ctrl: ArmController, kind: str, src, dst, target_yaw: float | tuple[float, ...], log: list,
              locator: Locator | None) -> dict:
    model, data = ctrl.model, ctrl.data
    gt_pos, gt_yaw = scene.instrument_pose(model, data, kind)  # 評価用
    res = {"instrument": kind, "from": src, "to": dst, "start_yaw_deg": round(float(np.degrees(gt_yaw)), 1)}

    # 腕を待避させてから撮る (腕が器械を隠すとカウント・検出が崩れる: Phase 2 の評価)。
    # --gt でも同じ動きにして、違いを「把持目標の出どころ」だけにする (公平な比較のため)
    ctrl.move_joints(robot.PARK_QPOS, duration=1.5)
    ctrl.hold(0.3)
    if locator is None:
        grasp_pos, yaw = gt_pos, gt_yaw  # --gt: 真値で掴む
    else:
        ests, _ = locator.locate(data)
        if kind not in ests:
            res.update(success=False, lifted=False, detected=False)
            log.append(res)
            print(f"NG {kind:14s} {src}→{dst}  検出できず (見えている器械: {sorted(ests)})", flush=True)
            return res
        est = ests[kind]
        grasp_pos, yaw = est.pos, est.yaw
        res.update(detected=True, est_xy_err_mm=round(float(np.linalg.norm(grasp_pos[:2] - gt_pos[:2]) * 1000), 2),
                   est_yaw_err_deg=round(float(np.degrees(abs(wrap(yaw - gt_yaw)))), 2))

    others = [k for k in INSTRUMENT_NAMES if k != kind]
    others_before = {k: data.body(k).xpos.copy() for k in others}
    place_pos = np.array([dst[0], dst[1], scene.INSTRUMENT_Z])
    # 先端が target_yaw を向くように置くための、掴み方と手首の回し方を先に決める
    plan = plan_transfer(ctrl, grasp_pos, yaw, place_pos, target_yaw)
    g = pick(ctrl, grasp_pos, yaw, plan=plan)
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

    place(ctrl, place_pos, plan.place_yaw, before_release=measure, yaw_from=plan.pick_yaw)
    ctrl.hold(0.6)  # 置いた器械が落ち着くまで

    p_rel0, y_rel0 = rel_after_lift
    p_rel1, y_rel1 = rel_before_release["v"]
    final_pos, final_yaw = scene.instrument_pose(model, data, kind)
    res |= {
        "closed_width_mm": round(g.closed_width * 1000, 2),
        "lifted": bool(lifted),
        "grasp_offset_mm": round(grasp_offset_mm, 2),
        "tcp_static_err_mm": round(tcp_err_mm, 2),
        "slip_mm": round(float(np.linalg.norm(p_rel1 - p_rel0) * 1000), 2),
        "slip_deg": round(float(np.degrees(abs(wrap(y_rel1 - y_rel0)))), 2),
        "final_yaw_deg": round(float(np.degrees(final_yaw)), 1),
        "place_err_mm": round(float(np.linalg.norm(final_pos[:2] - place_pos[:2]) * 1000), 2),
        # 置き向きの誤差。向きを問わない置き場では、許される向きのうち近い方との差
        "place_yaw_err_deg": round(float(np.degrees(min(abs(wrap(final_yaw - t)) for t in np.atleast_1d(target_yaw)))), 2),
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
    est = f"est_err={res['est_xy_err_mm']:.1f}mm/{res['est_yaw_err_deg']:.1f}°  " if "est_xy_err_mm" in res else ""
    print(f"{flag}{kind:14s} {src}→{dst}  {est}lifted={res['lifted']}  width={res['closed_width_mm']:.1f}mm  "
          f"offset={res['grasp_offset_mm']:.1f}mm  slip={res['slip_mm']:.1f}mm/{res['slip_deg']:.1f}°  "
          f"place_err={res['place_err_mm']:.1f}mm/{res['place_yaw_err_deg']:.1f}°  "
          f"neighbor={res['neighbor_disturb_mm']:.1f}mm  tcp_err={res['tcp_static_err_mm']:.1f}mm", flush=True)
    for r in reasons:
        print(f"     └ 失敗理由: {r}", flush=True)
    return res


def random_table_layout(model, data, rng: np.random.Generator) -> dict[str, int]:
    """器械台の 5 スロットへの割り当てをシャッフルし、位置 ±1cm・向き ±6° (3 割は先端が逆向き) でずらす。
    評価用 (eval_locate と同じ分布)。器械 → スロット番号 を返す。"""
    slot_of = dict(zip(INSTRUMENT_NAMES, rng.permutation(len(scene.TABLE_SLOTS)).tolist()))
    for kind, slot in slot_of.items():
        x, y = scene.TABLE_SLOTS[slot]
        adr = model.joint(f"{kind}_free").qposadr[0]
        data.qpos[adr:adr + 3] = [x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01), scene.INSTRUMENT_Z + 0.001]
        yaw = (np.pi if rng.random() < 0.3 else 0.0) + rng.normal(0, np.radians(6))
        data.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
    mujoco.mj_forward(model, data)
    return slot_of


def run_layout(args, seed: int | None) -> list:
    """1 つの配置で、器械を 1 本ずつ 器械台 → トレイ → 器械台 と往復させる。"""
    model, data = scene.build(scene.SceneOptions(gravcomp=not args.no_gravcomp,
                                                 gripper_force_scale=args.grip_scale))
    slot_of = {v: k for k, v in scene.DEFAULT_LAYOUT.items()}
    if seed is not None:
        slot_of = random_table_layout(model, data, np.random.default_rng(seed))
    # 器械を支持バーの上に落ち着かせる
    for _ in range(300):
        mujoco.mj_step(model, data)

    hooks = []
    rec = None
    if args.video:
        rec = Recorder(model, OUT_DIR / f"pick_place{output_tag(args)}.mp4")
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
    kinds = [args.only] if args.only else sorted(slot_of, key=slot_of.get)
    locator = None if args.gt else Locator(model)

    log: list = []
    try:
        for kind in kinds:
            table_xy = scene.TABLE_SLOTS[slot_of[kind]]
            tray_xy = scene.TRAY_SLOTS[1]
            run_cycle(ctrl, kind, f"table{slot_of[kind]}", tray_xy, TRAY_YAWS, log, locator)
            run_cycle(ctrl, kind, "tray1", table_xy, TABLE_YAW, log, locator)
        ctrl.move_joints(robot.HOME_QPOS, duration=1.5)
    finally:
        if locator:
            locator.close()
        if rec:
            rec.close()
        if viewer:
            viewer.close()
    for r in log:
        r["seed"] = seed
    return log


def output_tag(args) -> str:
    return ("_gt" if args.gt else "") + (f"_rand{args.random}" if args.random is not None else "") + \
        (f"_trials{args.trials}" if args.trials else "") + (f"_{args.only}" if args.only else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=INSTRUMENT_NAMES, help="1 本だけ試す")
    ap.add_argument("--video", action="store_true", help="overview_cam で mp4 を書き出す")
    ap.add_argument("--viewer", action="store_true", help="ビューアで実時間表示 (mjpython 必須)")
    ap.add_argument("--no-gravcomp", action="store_true", help="重力補償を切る (比較実験用)")
    ap.add_argument("--grip-scale", type=float, default=scene.GRIPPER_FORCE_SCALE, help="把持力の倍率")
    ap.add_argument("--gt", action="store_true", help="知覚を使わず真値で掴む (Phase 1 の動作。比較用)")
    ap.add_argument("--random", type=int, metavar="SEED", help="器械台の配置をランダムにする (乱数の種)")
    ap.add_argument("--trials", type=int, metavar="N", help="ランダム配置 (種 1〜N) を N 通り続けて試し、まとめて集計する")
    args = ap.parse_args()
    if args.trials and (args.video or args.viewer):
        sys.exit("--trials と --video / --viewer は同時に使えません")

    t0 = time.time()
    seeds = range(1, args.trials + 1) if args.trials else [args.random]
    log = [r for sd in seeds for r in run_layout(args, sd)]

    n_ok = sum(r["success"] for r in log)
    print(f"\n成功 {n_ok}/{len(log)}  (wall {time.time() - t0:.1f}s)")
    keys = ("est_xy_err_mm", "est_yaw_err_deg") if not args.gt else ()
    done = [r for r in log if "slip_mm" in r]  # 検出できずに打ち切った回は除く
    for key in keys + ("grasp_offset_mm", "slip_mm", "slip_deg", "place_err_mm", "place_yaw_err_deg", "tcp_static_err_mm"):
        v = np.array([r[key] for r in done])
        print(f"  {key:20s} mean={v.mean():6.2f}  max={v.max():6.2f}")
    # 先端の向きの扱い: 掴む前に先端が器械台の向き (+x) と逆だった回を、正しい向きで器械台に戻せたか
    back = [r for r in done if r["to"] in [tuple(v) for v in scene.TABLE_SLOTS.values()]]
    flipped = [r for r in back if abs(r["start_yaw_deg"]) > 90]
    if flipped:
        print(f"  トレイで先端が逆向きだった器械: {len(flipped)}/{len(back)} 回 → 器械台に正しい向きで戻せた "
              f"{sum(r['place_yaw_err_deg'] < 10 for r in flipped)}/{len(flipped)} 回")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pick_place_metrics{output_tag(args)}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
