"""Phase 5: カウントと照合 (警告) の評価。器械の行き先をランダムに決めた場面で、正しく数えて正しく警告できるかを測る。

    uv run python -m inventory.eval_count                 # 本番の評価 (乱数の種 1000〜、60 場面)
    uv run python -m inventory.eval_count -n 5 --seed 0   # 開発用 (種 0〜)。ルールを直すときはこちらだけを見る

1 場面の作り方
  1. 5 本を器械台にランダムに置く (位置 ±1cm・向き ±6°、3 割は先端が逆向き) → 初期カウント
  2. 各器械の行き先を決め、台帳もそのとおりに付ける (術者のスクリプト)
       器械台に残る / 返却トレイ (重なりあり・なし) / 術者 (手に持っている = カメラに写る / 術野の中 = 写らない)
     一部の場面では「台帳では返却済みなのに実物が無い」(記録の誤り) も混ぜる → 画像で見つけて「所在不明」と言えるか
  3. 照合カウント。同じ場面を「並べ直しなし」「並べ直しあり」の 2 通りで実行して比べる

指標
  初期カウント一致率   … 器械台の 5 本を種類まで正しく数えた割合
  最終カウント一致率   … 最後に撮った画像で、器械台・トレイのそれぞれにある器械が真値と一致した割合
  照合の正解率         … 警告の内容が完全に正しかった割合 (足りない器械が無ければ警告なし、あれば その器械名だけで警告)
  誤警告               … 足りない器械が無いのに警告した (人が確認すれば済む、安全側の誤り)
  危険な誤り           … 本当に足りない器械の名前が警告文に出ず、「重なりで数えきれない」の注意も出なかった
                         (体内遺残の見逃しにつながる)。0 でなければならない
"""

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from control.motion import ArmController
from control.pick_place_demo import random_table_layout
from inventory.count import InstrumentCounter, region_of
from inventory.draw import draw_count
from inventory.ledger import SURGEON, TABLE, TRAY, Ledger
from inventory.procedures import check_count, initial_count
from perception.gen_dataset import HIDDEN_POS, yaw_quat
from perception.locate import Locator
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES

OUT_DIR = Path("outputs/phase5")


def settle(model, data, seconds: float = 0.6) -> None:
    data.ctrl[:7] = robot.arm_qpos(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def set_pose(model, data, kind: str, pos, yaw: float) -> None:
    adr = model.joint(f"{kind}_free").qposadr[0]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
    data.qvel[model.joint(f"{kind}_free").dofadr[0]:][:6] = 0


def make_scenario(model, data, ledger: Ledger, rng: np.random.Generator, crowd: float) -> dict:
    """器械の行き先を決めて、実物 (シミュレーター) と台帳の両方を動かす。真値 (足りない器械など) を返す。"""
    order = [str(k) for k in rng.permutation(INSTRUMENT_NAMES)]
    n_missing = int(rng.choice([0, 0, 1, 1, 2]))  # 警告すべき場面とそうでない場面が半々くらいになるように
    missing, rest = order[:n_missing], order[n_missing:]
    n_tray = int(rng.integers(0, min(3, len(rest)) + 1))
    tray = rest[:n_tray]
    piled = n_tray >= 2 and rng.random() < crowd

    in_hand = None
    for i, kind in enumerate(missing):
        ledger.move(kind, SURGEON, by="robot")
        if i == 0 and rng.random() < 0.5:
            # 術者が手に持っている: 検出カメラに写るが、数えてはいけない
            in_hand = kind
            hand = data.body("surgeon_hand").xpos
            set_pose(model, data, kind, hand + [0, 0, -0.02], rng.uniform(-np.pi, np.pi))
            mujoco.mj_forward(model, data)
            scene.hand_grab(model, data, kind)
        else:
            set_pose(model, data, kind, HIDDEN_POS + [0.3 * i, 0, 0], 0.0)  # 術野の中 (カメラの外)
    for j, kind in enumerate(tray):
        ledger.move(kind, SURGEON, by="robot")
        ledger.move(kind, TRAY, by="surgeon")
        yaw = (np.pi if rng.random() < 0.5 else 0.0) + rng.normal(0, np.radians(6))
        if piled:
            # 術者が無造作に返した: トレイの中央付近に少しずつ高さを変えて落とす (重なりは物理で作る。Phase 3 と同じ)
            cx, cy = scene.TRAY_CENTER
            pos = [rng.uniform(cx - 0.02, cx + 0.02), rng.uniform(cy - 0.06, cy + 0.06), scene.INSTRUMENT_Z + 0.015 * (j + 1)]
            yaw += rng.normal(0, np.radians(15))
        else:
            x, y = scene.TRAY_SLOTS[[1, 0, 2][j]]
            pos = [x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01), scene.INSTRUMENT_Z + 0.001]
        set_pose(model, data, kind, pos, yaw)

    # 記録の誤り: 術野に残っている器械を、台帳では「返却済み (トレイ)」にしてしまった
    ledger_error = None
    hidden = [k for k in missing if k != in_hand]
    if hidden and rng.random() < 0.3:
        ledger_error = hidden[0]
        ledger.move(ledger_error, TRAY, by="surgeon")
    settle(model, data, 1.0)
    return {"missing": sorted(missing), "tray": sorted(tray), "piled": bool(piled), "in_hand": in_hand,
            "ledger_error": ledger_error}


def true_where(model, data) -> dict[str, str]:
    """真値: 各器械の重心がどの箱に入っているか (評価専用)。"""
    out = {}
    for kind in INSTRUMENT_NAMES:
        r = region_of(data.body(kind).xpos)
        if r != "outside":
            out[kind] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60, help="場面の数")
    ap.add_argument("--seed", type=int, default=1000, help="最初の乱数の種 (開発は 0〜、本番の評価は 1000〜)")
    ap.add_argument("--crowd", type=float, default=0.5, help="トレイに 2 本以上返すとき、重ねて返す割合")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model, data = scene.build()
    loc = Locator(model)
    counter = InstrumentCounter(loc)
    rows, examples = [], []
    t0 = time.time()
    for i in range(args.n):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(model, data)
        robot.reset_to_home(model, data)
        robot.set_arm_qpos(model, data, robot.PARK_QPOS)
        random_table_layout(model, data, rng)
        settle(model, data)

        ledger = Ledger()
        rep0 = initial_count(ArmController(model, data), counter, ledger)
        init_ok = sorted(rep0.counts[0].counted()) == [(k, "table") for k in sorted(INSTRUMENT_NAMES)]
        if not init_ok:
            print(f"[{seed}] 初期カウントの誤り: {rep0.counts[0].at('table')}")
            # 初期在庫が間違っていると以降の照合の意味が変わるので、この場面は照合を評価しない
            rows.append({"seed": seed, "init_ok": False})
            continue
        truth = make_scenario(model, data, ledger, rng, args.crowd)

        # 同じ場面を 2 通りで実行する。シミュレーターの状態と台帳を保存しておき、戻してからもう一度
        saved = mujoco.MjData(model)
        mujoco.mj_copyData(saved, model, data)
        ledger0 = copy.deepcopy(ledger)
        row = {"seed": seed, "init_ok": True, **truth}
        for mode in ("plain", "rearrange"):
            mujoco.mj_copyData(data, model, saved)
            led = copy.deepcopy(ledger0)
            ts = time.time()
            rep = check_count(ArmController(model, data), counter, led, rearrange_on_mismatch=(mode == "rearrange"),
                              log=lambda s: None)
            last = rep.counts[-1]
            predicted_missing = sorted(rep.recon.missing)
            row[mode] = {
                # 数えた (器械, 場所) の一覧が真値と完全に一致したか。取り違え・重複・見逃しのどれでも不一致になる
                "count_ok": sorted(last.counted()) == sorted(true_where(model, data).items()),
                "first_count_ok": sorted(rep.counts[0].counted()) == sorted(true_where(model, saved).items()),
                "missing_pred": predicted_missing,
                "problems": rep.recon.problems(),
                # 完全に正しい: 足りない器械の集合が真値どおりで、それ以外の問題 (余分・台帳と違う・重なり) を言っていない
                "recon_ok": predicted_missing == truth["missing"] and len(rep.recon.problems()) == len(truth["missing"]),
                # 危険: 本当に足りない器械の名前が警告文に出ず、重なりの注意 (人が全部数え直す) も出なかった
                "dangerous": bool(set(truth["missing"]) - rep.recon.named()) and not rep.recon.stacked,
                "alert": rep.message if rep.alert else None,
                "rounds": len(rep.counts),
                "moved": rep.moved,
                "sec": round(time.time() - ts, 1),
                "sim_sec": round(float(data.time - saved.time), 1),
            }
            if mode == "rearrange" and (len(examples) < 4 or (not row["plain"]["recon_ok"] and len(examples) < 8)):
                examples.append(cv2.resize(draw_count(rep.counts[0], loc, saved, ledger0, f"最終判定: {rep.message}", rep.alert), (640, 480)))
        rows.append(row)
        p, r = row["plain"], row["rearrange"]
        print(f"[{seed}] 足りない={truth['missing'] or '-'} トレイ={truth['tray'] or '-'}{' (重なり)' if truth['piled'] else ''}"
              f"{' 手に=' + truth['in_hand'] if truth['in_hand'] else ''}{' 記録誤り=' + truth['ledger_error'] if truth['ledger_error'] else ''}"
              f"  | なし: {'OK' if p['recon_ok'] else '危険' if p['dangerous'] else 'NG'} {p['problems']}"
              f"  | あり: {'OK' if r['recon_ok'] else '危険' if r['dangerous'] else 'NG'} {r['problems']}"
              f" (撮影 {r['rounds']} 回, 運んだ {len(r['moved'])} 本)", flush=True)
    loc.close()

    done = [r for r in rows if r["init_ok"]]
    summary = {"n": len(rows), "init_count_acc": round(len(done) / len(rows), 4)}
    for mode in ("plain", "rearrange"):
        m = [r[mode] for r in done]
        need = [r for r in done if r["missing"]]
        noneed = [r for r in done if not r["missing"]]
        summary[mode] = {
            "first_count_acc": round(np.mean([x["first_count_ok"] for x in m]), 4),
            "final_count_acc": round(np.mean([x["count_ok"] for x in m]), 4),
            "recon_acc": round(np.mean([x["recon_ok"] for x in m]), 4),
            "false_alarm": f"{sum(r[mode]['alert'] is not None for r in noneed)}/{len(noneed)}",
            "dangerous": f"{sum(r[mode]['dangerous'] for r in need)}/{len(need)}",
            "piled_recon_acc": round(np.mean([r[mode]["recon_ok"] for r in done if r["piled"]] or [np.nan]), 4),
            "ledger_error_caught": f"{sum(r[mode]['recon_ok'] for r in done if r['ledger_error'])}/"
                                   f"{sum(1 for r in done if r['ledger_error'])}",
            "mean_sim_sec": round(float(np.mean([x["sim_sec"] for x in m])), 1),
        }
    print(f"\n初期カウント一致: {summary['init_count_acc'] * 100:.1f}% ({len(done)}/{len(rows)})   (wall {time.time() - t0:.0f}s)")
    print(f"{'':10s} 1回目の一致 最後の一致 照合の正解 誤警告 危険な誤り 重なりでの照合 記録誤りの発見 所要(シム秒)")
    for mode, ja in (("plain", "並べ直しなし"), ("rearrange", "並べ直しあり")):
        s = summary[mode]
        print(f"{ja:8s} {s['first_count_acc'] * 100:8.1f}% {s['final_count_acc'] * 100:8.1f}% {s['recon_acc'] * 100:8.1f}% "
              f"{s['false_alarm']:>6s} {s['dangerous']:>8s} {s['piled_recon_acc'] * 100:10.1f}% {s['ledger_error_caught']:>10s} "
              f"{s['mean_sim_sec']:10.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    (OUT_DIR / f"count_metrics{tag}.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=2, ensure_ascii=False, default=str))
    if examples:
        while len(examples) % 2:
            examples.append(np.zeros_like(examples[0]))
        cv2.imwrite(str(OUT_DIR / f"count_examples{tag}.jpg"),
                    np.vstack([np.hstack(examples[j:j + 2]) for j in range(0, len(examples), 2)]))
    print(f"saved {OUT_DIR}/count_metrics{tag}.json, count_examples{tag}.jpg")


if __name__ == "__main__":
    main()
