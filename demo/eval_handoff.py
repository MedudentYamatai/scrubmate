"""Phase 7: 受け渡しと回収の評価。デモで使っているロボットの頭脳 (demo/agent.py) と術者のスクリプト (demo/surgeon.py) で、
「器械台から術者へ渡す → 術者が使ってトレイに返す → ロボットが気づいて器械台に戻す」を繰り返す。

    uv run python -m demo.eval_handoff                 # 5 本 × 6 周 = 30 回 (乱数の種 0)
    uv run python -m demo.eval_handoff --rounds 2 --seed 1

術者はトレイの 3 か所の置き場のどこかに、前後 ±1cm・左右 ±1cm・向き ±20° のばらつきで返す
(返し方の条件は Phase 5 の試作で決めたもの: バーと直交する向きで、バーの 1.5cm 上でそっと置く)。

成功の定義
  渡す … 術者の手が器械を持ち、ロボットが離れた後も器械が手に付いている
  回収 … 器械が器械台のスロットに、支持バーの上 (高さの誤差 5mm 以内)・スロットの中心から 1cm 以内で置かれている
真値 (シミュレーターの器械の位置) は判定にだけ使い、ロボットは知らない。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from control.motion import ArmController
from demo.agent import ScrubNurse
from demo.surgeon import Surgeon
from inventory.count import region_of
from inventory.eval_count import settle
from inventory.ledger import TABLE
from language.interpret import Interpreter
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES, JA_NAMES

OUT_DIR = Path("outputs/phase7")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model, data = scene.build()
    robot.set_arm_qpos(model, data, robot.PARK_QPOS)
    settle(model, data)
    surgeon = Surgeon(model, data)
    ctrl = ArmController(model, data, on_step=surgeon.update)
    agent = ScrubNurse(model, data, ctrl, surgeon, speak=lambda s: None, show=lambda img, cap: None,
                       note=lambda s: None, interpreter=Interpreter(use_llm=False))
    agent.count()  # 初期カウント (台帳の初期在庫と、器械台に戻すときのスロットを決める)

    def wait(cond, limit: float) -> bool:
        t0 = data.time
        while not cond() and data.time - t0 < limit:
            agent.tick()
        return cond()

    rows = []
    for rnd in range(args.rounds):
        for kind in INSTRUMENT_NAMES:
            # 渡す
            agent.handoff(kind)
            handed = wait(lambda: not surgeon.busy, 15.0) and agent.ledger.state[kind] == "surgeon"
            in_hand = kind in surgeon.on_drape  # 術者が受け取ってドレープに置けた = 手に付いて運べた
            # 術者がトレイに返す (置き場とばらつきは乱数)
            slot = int(rng.integers(0, 3))
            surgeon.start(f"return {kind}", surgeon.return_to_tray(
                kind, slot=slot, dy=float(rng.uniform(-0.01, 0.01)), yaw_dev=float(np.radians(rng.uniform(-20, 20)))))
            # ロボットが気づいて器械台に戻すまで (見張りは数秒おき)
            # 回収に失敗しても、もう 1 回試すか人に知らせるまで待つ
            back = wait(lambda: (not surgeon.busy and agent.phase == "idle"
                                 and (agent.ledger.state[kind] == TABLE or kind in agent.gave_up)), 80.0)
            x, y = scene.TABLE_SLOTS[agent.ledger.home_slot[kind]]
            p = data.body(kind).xpos
            placed = bool(abs(p[2] - scene.INSTRUMENT_Z) < 0.005 and np.hypot(p[0] - x, p[1] - y) < 0.01)
            told = kind in agent.gave_up  # 掴めずに人に知らせたか
            rows.append({"round": rnd, "kind": kind, "tray_slot": slot, "handed": bool(handed and in_hand), "told": told,
                         # 台帳の場所が、真値で器械がある場所 (器械台 / トレイの箱) と合っているか
                         "ledger_ok": agent.ledger.state[kind] == region_of(data.body(kind).xpos),
                         "retrieved": bool(back and placed),
                         "place_err_mm": round(float(np.hypot(p[0] - x, p[1] - y) * 1000), 2)})
            r = rows[-1]
            print(f"[{rnd}] {JA_NAMES[kind]:4s} 渡す {'OK' if r['handed'] else 'NG'}  トレイの置き場 {slot} → 回収 "
                  f"{'OK' if r['retrieved'] else 'NG'} (器械台のスロットからのずれ {r['place_err_mm']:.1f}mm)", flush=True)
            if not r["retrieved"]:
                # 次の回に持ち越さないよう、失敗した器械を元のスロットに戻す (評価の後片付け)
                from inventory.eval_count import set_pose
                set_pose(model, data, kind, [x, y, scene.INSTRUMENT_Z + 0.001], 0.0)
                agent.ledger.state[kind] = TABLE
                agent.gave_up.discard(kind)  # 頭脳側の「あきらめた器械」の記録も消す (消さないと次の周の回収が連鎖して失敗した)
                agent.retry.pop(kind, None)
                settle(model, data, 0.5)

    n = len(rows)
    summary = {"n": n, "handed": sum(r["handed"] for r in rows), "retrieved": sum(r["retrieved"] for r in rows),
               "failed_and_told": sum((not r["retrieved"]) and r["told"] for r in rows),
               "ledger_ok": sum(r["ledger_ok"] for r in rows),
               "place_err_mean_mm": round(float(np.mean([r["place_err_mm"] for r in rows])), 2),
               "place_err_max_mm": round(float(np.max([r["place_err_mm"] for r in rows])), 2),
               "sim_seconds": round(float(data.time), 1)}
    print(f"\n渡す {summary['handed']}/{n}  回収 {summary['retrieved']}/{n} (失敗して人に知らせた {summary['failed_and_told']})  "
          f"台帳が実物と合っていた {summary['ledger_ok']}/{n}  "
          f"戻した位置のずれ 平均 {summary['place_err_mean_mm']}mm / 最大 {summary['place_err_max_mm']}mm")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"handoff_metrics_seed{args.seed}.json"
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
