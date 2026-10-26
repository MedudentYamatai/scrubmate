"""Phase 5: デモシナリオに沿って、台帳とカウントだけを通しで動かす (音声と受け渡しは Phase 6 で足す)。

    uv run python -m inventory.count_demo              # 返却トレイに並べて返す
    uv run python -m inventory.count_demo --pile       # 返却トレイに重ねて返す (並べ直しが起きる)
    uv run python -m inventory.count_demo --video      # ロボットの動き (メスを戻す・並べ直し) を mp4 に

流れ (デモシナリオの 1〜8 のうち、カウントと台帳に関わる所)
  1. 「器械カウントお願いします」→ 初期カウント (5 本)
  2〜3. メス・鑷子を術者へ (台帳だけ。器械は術野へ移す = 術者のスクリプト)
  4. 術者がメスをトレイに返す → ロボットが器械台に戻す (カメラで推定して掴む)
  5〜6. ペアン・持針器を術者へ
  7. 術者が鑷子とペアンをトレイに返す。持針器は返さない (術野に残った想定)
  8. 「閉創前カウント」→ 照合 → 持針器 1 本未返却の警告で停止

出力: outputs/phase5/demo{_pile}_<番号>.jpg (カウントごとの検出カメラの画像)、demo{_pile}.mp4 (--video)
"""

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from control.motion import ArmController
from inventory.count import InstrumentCounter
from inventory.draw import draw_count
from inventory.eval_count import settle, set_pose
from inventory.ledger import SURGEON, TABLE, TRAY, Ledger
from inventory.procedures import count_request, move_to_table, park_and_count
from perception.gen_dataset import HIDDEN_POS
from perception.locate import Locator
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES, JA_NAMES
from sim.render import Recorder

OUT_DIR = Path("outputs/phase5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pile", action="store_true", help="術者がトレイに器械を重ねて返す")
    ap.add_argument("--video", action="store_true", help="overview_cam で mp4 を書き出す")
    args = ap.parse_args()
    tag = "_pile" if args.pile else ""

    model, data = scene.build()
    robot.set_arm_qpos(model, data, robot.PARK_QPOS)
    settle(model, data)
    rec = Recorder(model, OUT_DIR / f"demo{tag}.mp4") if args.video else None
    ctrl = ArmController(model, data, on_step=(lambda: rec.maybe_capture(data)) if rec else None)
    loc = Locator(model)
    counter = InstrumentCounter(loc)
    ledger = Ledger()
    n_img = [0]

    def save(rep, title: str) -> None:
        for i, c in enumerate(rep.counts):
            # 台帳と判定は最後の撮影の時点のもの。並べ直す前の画像には「数え直し」とだけ書く
            last = i == len(rep.counts) - 1
            img = draw_count(c, loc, data, ledger if last else None,
                             f"{title}: {rep.message}" if last else f"{title} ({i + 1} 回目): 合わない → 並べ直して数え直す",
                             rep.alert or not last)
            path = OUT_DIR / f"demo{tag}_{n_img[0]}.jpg"
            cv2.imwrite(str(path), img)
            n_img[0] += 1
            print(f"   画像 {path}")

    def to_surgeon(kind: str, i: int) -> None:
        ledger.move(kind, SURGEON, by="robot")
        set_pose(model, data, kind, HIDDEN_POS + [0.3 * i, 0, 0], 0.0)  # 術野へ (検出カメラの外)
        print(f"{JA_NAMES[kind]} → 術者   台帳: {ledger.summary_lines()}")

    def return_to_tray(kinds: list[str]) -> None:
        rng = np.random.default_rng(0)
        for j, kind in enumerate(kinds):
            ledger.move(kind, TRAY, by="surgeon")
            if args.pile:
                cx, cy = scene.TRAY_CENTER
                pos = [cx + rng.uniform(-0.01, 0.01), cy + 0.02 * (j - 0.5), scene.INSTRUMENT_Z + 0.015 * (j + 1)]
                yaw = np.pi * j + rng.normal(0, np.radians(12))
            else:
                pos, yaw = [*scene.TRAY_SLOTS[[1, 0, 2][j]], scene.INSTRUMENT_Z + 0.001], np.pi * j
            set_pose(model, data, kind, pos, yaw)
        ctrl.hold(1.0)  # 器械がトレイに落ち着くまで (物理で)
        print(f"術者が {'・'.join(JA_NAMES[k] for k in kinds)} をトレイに返却   台帳: {ledger.summary_lines()}")

    print("1. 「器械カウントお願いします」")
    rep = count_request(ctrl, counter, ledger)
    print(f"   → {rep.message}")
    save(rep, "初期カウント")

    to_surgeon("scalpel", 0)
    to_surgeon("forceps", 1)
    return_to_tray(["scalpel"])
    # 4. 返却されたメスを器械台に戻す (カメラで見つけて掴む)
    c = park_and_count(ctrl, counter)
    d = next(d for d in c.detections if d.kind == "scalpel" and d.region == "tray")
    move_to_table(ctrl, counter, c, d, ledger.home_slot["scalpel"])
    ledger.move("scalpel", TABLE, by="robot")
    print(f"ロボットがメスを器械台に戻した   台帳: {ledger.summary_lines()}")
    to_surgeon("pean", 2)
    to_surgeon("needle_holder", 3)
    return_to_tray(["forceps", "pean"])

    print("8. 「閉創前カウント」")
    rep = count_request(ctrl, counter, ledger, log=lambda s: print("   " + s))
    print(f"   → {'⚠ ' if rep.alert else ''}{rep.message}   (撮影 {len(rep.counts)} 回、並べ直し {len(rep.moved)} 本)")
    print(f"   台帳: {ledger.summary_lines()}  停止中: {ledger.halted}")
    save(rep, "閉創前カウント")
    ctrl.hold(1.0)
    loc.close()
    if rec:
        rec.close()
    print("台帳の履歴:")
    for ev in ledger.history:
        name = JA_NAMES.get(ev.kind, ev.kind)
        print(f"  {ev.seq:2d}. {name:4s} {ev.src:8s} → {ev.dst:8s} {ev.action} ({ev.by})")
    truth = {k: data.body(k).xpos.round(3).tolist() for k in INSTRUMENT_NAMES}
    print(f"(真値: {truth})")


if __name__ == "__main__":
    main()
