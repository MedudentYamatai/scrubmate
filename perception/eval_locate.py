"""Phase 3: 位置・向きの推定 (perception/locate.py) を、ランダムな置き方で真値と比べて評価する。

    uv run python -m perception.eval_locate            # 50 配置 × 5 本 = 250 回の推定
    uv run python -m perception.eval_locate -n 10 --spread 2   # 置き方のばらつきを学習データの 2 倍にする (実験用)

置き方は学習データと同じ分布: 器械台 5 + 返却トレイ 3 の場所のどれか、位置 ±1cm、向き ±6° (3 割は先端が逆向き)。
腕は待避姿勢 (Phase 2 でカウント一致率 100% だった条件)。物理を回して器械をバーの上に落ち着かせてから撮る。

出力: outputs/phase3/
  locate_metrics.json  … 器械ごとの 位置誤差 (平均・最大)・高さ誤差・向き誤差・先端方向の正解率・見逃し数
  locate_examples.jpg  … 推定 (色付きの矢印 = 把持点と先端方向) と真値 (白い矢印) を重ねた画像
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from perception.gen_dataset import PLACES, yaw_quat
from perception.locate import Locator, draw
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES

OUT_DIR = Path("outputs/phase3")


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def random_layout(model, data, rng, spread: float = 1.0) -> None:
    """5 本を、8 か所の置き場のうちランダムな 5 か所に置き、物理で落ち着かせる。"""
    places = rng.permutation(len(PLACES))[:len(INSTRUMENT_NAMES)]
    for kind, pi in zip(INSTRUMENT_NAMES, places):
        _, _, (x, y) = PLACES[pi]
        yaw = (np.pi if rng.random() < 0.3 else 0.0) + rng.normal(0, np.radians(6 * spread))
        adr = model.joint(f"{kind}_free").qposadr[0]
        data.qpos[adr:adr + 3] = [x + rng.uniform(-0.01, 0.01) * spread, y + rng.uniform(-0.01, 0.01) * spread,
                                  scene.INSTRUMENT_Z + 0.001]
        data.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
    data.qvel[:] = 0
    robot.set_arm_qpos(model, data, robot.PARK_QPOS)
    data.ctrl[:7] = robot.PARK_QPOS
    for _ in range(300):
        mujoco.mj_step(model, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50, help="配置の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spread", type=float, default=1.0, help="位置・向きのばらつきの倍率 (1 = 学習データと同じ)")
    ap.add_argument("--tag", default="", help="出力ファイル名の接尾辞")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, data = scene.build()
    loc = Locator(model)
    rows, tiles = [], []
    for i in range(args.n):
        random_layout(model, data, rng, args.spread)
        ests, rgb = loc.locate(data)
        truth = {k: scene.instrument_pose(model, data, k) for k in INSTRUMENT_NAMES}
        for kind in INSTRUMENT_NAMES:
            tp, ty = truth[kind]
            e = ests.get(kind)
            if e is None:
                rows.append({"layout": i, "kind": kind, "found": False})
                continue
            yaw_err = abs(wrap(e.yaw - ty))  # 先端方向込みの誤差 (0〜180°)
            rows.append({
                "layout": i, "kind": kind, "found": True, "conf": round(e.conf, 3),
                "xy_err_mm": round(float(np.linalg.norm(e.pos[:2] - tp[:2]) * 1000), 2),
                "z_err_mm": round(float((e.pos[2] - tp[2]) * 1000), 2),
                # 長手方向の軸の誤差 (前後は問わない、0〜90°)
                "axis_err_deg": round(float(np.degrees(min(yaw_err, np.pi - yaw_err))), 2),
                "tip_ok": bool(yaw_err < np.pi / 2),
                "tip_margin": round(e.tip_margin, 3),
            })
        if i < 4:
            tiles.append(cv2.resize(draw(rgb, ests, truth, loc, data), (768, 576)))
    loc.close()

    summary = {}
    print(f"\n{'':14s} 見逃し  位置誤差 平均/最大[mm]  高さ誤差 最大[mm]  軸の誤差 平均/最大[°]  先端方向の正解率")
    for kind in INSTRUMENT_NAMES + ["all"]:
        rs = [r for r in rows if kind in ("all", r["kind"])]
        f = [r for r in rs if r["found"]]
        xy = np.array([r["xy_err_mm"] for r in f])
        ax = np.array([r["axis_err_deg"] for r in f])
        s = {"n": len(rs), "missed": len(rs) - len(f),
             "xy_err_mean": round(float(xy.mean()), 2), "xy_err_max": round(float(xy.max()), 2),
             "z_err_absmax": round(float(np.abs([r["z_err_mm"] for r in f]).max()), 2),
             "axis_err_mean": round(float(ax.mean()), 2), "axis_err_max": round(float(ax.max()), 2),
             "tip_acc": round(float(np.mean([r["tip_ok"] for r in f])), 4)}
        summary[kind] = s
        print(f"{kind:14s} {s['missed']:4d}   {s['xy_err_mean']:8.1f} / {s['xy_err_max']:5.1f}       "
              f"{s['z_err_absmax']:8.1f}          {s['axis_err_mean']:6.2f} / {s['axis_err_max']:6.2f}       "
              f"{s['tip_acc'] * 100:6.1f}%")
    wrong = [r for r in rows if r["found"] and not r["tip_ok"]]
    for r in wrong:
        print(f"  先端を逆に判定: layout {r['layout']} {r['kind']} (margin {r['tip_margin']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    (OUT_DIR / f"locate_metrics{tag}.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    cv2.imwrite(str(OUT_DIR / f"locate_examples{tag}.jpg"),
                np.vstack([np.hstack(tiles[j:j + 2]) for j in range(0, len(tiles) - 1, 2)]))
    print(f"saved {OUT_DIR}/locate_metrics{tag}.json, locate_examples{tag}.jpg")


if __name__ == "__main__":
    main()
