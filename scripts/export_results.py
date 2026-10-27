"""Phase 7: 各 Phase の評価結果 (outputs/ の JSON) を集めて docs/results.md に表としてまとめ、
README・レポートで使う図を docs/media/ に書き出す。

    uv run python -m scripts.export_results

数字は必ずここを通す (README・レポート・動画のスライドの数字の出どころを 1 か所にするため)。
outputs/ は再生成できる生成物なので git に入れていない。各評価のやり直し方は README を参照。
"""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

OUT = Path("outputs")
DOCS = Path("docs")
MEDIA = DOCS / "media"


def load(path: str):
    return json.loads((OUT / path).read_text())


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def detection() -> list[str]:
    m = load("phase2/eval_crowd_on_main/metrics.json")  # 本番の重み × 元の検証データ (300 枚)
    c = load("phase2/eval_crowd_on_crowd/metrics.json")  # 本番の重み × 重なりを含む検証データ
    lines = ["## 検出 (YOLO11n-seg、本番の重み `perception/weights/instruments_seg.pt`)", "",
             "| 検証データ | box mAP50 | mask mAP50 | mask mAP50-95 | カウント一致 (腕を待避) | カウント一致 (腕が器械台の上) |",
             "|---|---|---|---|---|---|",
             f"| 元の検証画像 300 枚 | {m['box_mAP50']:.3f} | {m['mask_mAP50']:.3f} | {m['mask_mAP50-95']:.3f} | "
             f"{pct(m['count_accuracy']['park'])} ({m['count_accuracy']['n_park']} 枚) | "
             f"{pct(m['count_accuracy']['over_table'])} ({m['count_accuracy']['n_over_table']} 枚) |",
             f"| 重なりを含む検証画像 120 枚 | {c['box_mAP50']:.3f} | {c['mask_mAP50']:.3f} | {c['mask_mAP50-95']:.3f} | "
             f"{pct(c['count_accuracy']['park'])} ({c['count_accuracy']['n_park']} 枚) | "
             f"{pct(c['count_accuracy']['over_table'])} ({c['count_accuracy']['n_over_table']} 枚) |", "",
             "器械ごとの mask mAP50-95 (元の検証画像): " +
             " / ".join(f"{k} {v['mask_mAP50-95']:.3f}" for k, v in m["per_class"].items()), ""]
    return lines


def locate() -> list[str]:
    s = load("phase3/locate_metrics.json")["summary"]["all"]
    return ["## 把持点・向きの推定 (ランダムな置き方 50 通り × 5 本、真値と比較)", "",
            "| 見逃し | 位置誤差 平均 / 最大 | 長手方向の誤差 平均 / 最大 | 先端の向きの正解率 |", "|---|---|---|---|",
            f"| {s['missed']} / {s['n']} | {s['xy_err_mean']:.1f} / {s['xy_err_max']:.1f} mm | "
            f"{s['axis_err_mean']:.2f} / {s['axis_err_max']:.2f}° | {pct(s['tip_acc'])} |", ""]


def pick_place() -> list[str]:
    rows = load("phase3/pick_place_metrics_trials30.json")
    ok = sum(r["success"] for r in rows)
    done = [r for r in rows if "place_err_mm" in r]
    pe = np.array([r["place_err_mm"] for r in done])
    gt = load("phase3/pick_place_metrics_gt_trials10.json")
    gpe = np.array([r["place_err_mm"] for r in gt if "place_err_mm" in r])
    return ["## pick & place (器械台 → 返却トレイ → 器械台、ランダムな置き方)", "",
            "| 把持目標 | 成功 | 置き位置の誤差 平均 / 最大 |", "|---|---|---|",
            f"| カメラの推定 (30 通り) | {ok} / {len(rows)} | {pe.mean():.1f} / {pe.max():.1f} mm |",
            f"| シミュレーターの真値 (比較用、10 通り) | {sum(r['success'] for r in gt)} / {len(gt)} | "
            f"{gpe.mean():.1f} / {gpe.max():.1f} mm |", "",
            "成功の定義: 持ち上がった・置き位置の誤差 1cm 以内・向きの誤差 10° 以内・隣の器械を 5mm 以上動かしていない。", ""]


def language() -> list[str]:
    rows = load("phase4/language_eval.json")
    from language.eval_language import names_in
    named = [r for r in rows if names_in(r["text"])]  # 器械名か工程名が入った台詞だけ (eval_language と同じ数え方)
    lines = ["## 音声認識と指示の解釈 (台詞 23 個)", "",
             "| 声 | 文字誤り率 | 器械名・工程名の聞き取り 補正前 → 補正後 | 予備ルールでの解釈 |", "|---|---|---|---|"]
    for v in ("kyoko", "reed", "eddy", "grandpa", "sandy"):
        cer = np.mean([r["voices"][v]["cer"] for r in rows])
        nb = sum(bool(r["voices"][v]["names_before"]) for r in named)
        na = sum(bool(r["voices"][v]["names_after"]) for r in named)
        fb = sum(tuple(r["voices"][v]["fallback"]) == tuple(r["expect"]) for r in rows)
        lines.append(f"| {v}{' (デモ用)' if v == 'kyoko' else ''} | {cer * 100:.1f}% | {nb} / {len(named)} → {na} / {len(named)} | "
                     f"{fb} / {len(rows)} |")
    gem = [r for r in rows if r.get("gemini_kyoko") is not None]
    gok = sum(tuple(r["gemini_kyoko"]) == tuple(r["expect"]) for r in gem)
    lines += ["", f"Gemini による解釈 (Kyoko の書き起こし): {gok} / {len(gem)}", ""]
    return lines


def counting() -> list[str]:
    lines = ["## 器械カウントと照合", "",
             "| 評価 | 照合の正解 (並べ直しなし → あり) | 重なりのある場面 | 誤警告 | 足りない器械の見逃し (危険な誤り) |",
             "|---|---|---|---|---|"]
    init = []
    for name, f in (("ランダム 60 場面", "phase5/count_metrics.json"), ("重なり多め 30 場面", "phase5/count_metrics_crowd.json")):
        s = load(f)["summary"]
        init.append(f"{name} {pct(s['init_count_acc'])}")
        p, r = s["plain"], s["rearrange"]
        lines.append(f"| {name} | {pct(p['recon_acc'])} → {pct(r['recon_acc'])} | {pct(p['piled_recon_acc'])} → "
                     f"{pct(r['piled_recon_acc'])} | {p['false_alarm']} → {r['false_alarm']} | "
                     f"{p['dangerous']} → {r['dangerous']} |")
    lines += ["", f"初期カウントの一致: {' / '.join(init)}。規則は別の開発用の場面 (乱数の種 0〜29、100〜129) だけを見て決めた。", ""]
    return lines


def handoff() -> list[str]:
    runs = [json.loads(f.read_text()) for f in sorted((OUT / "phase7").glob("handoff_metrics_seed*.json"))]
    rows = [r for run in runs for r in run["rows"]]
    fails = [r for r in rows if not r["retrieved"]]
    lines = ["## 受け渡しと回収 (デモと同じロボットの頭脳と術者のスクリプト、`demo/eval_handoff.py`)", "",
             "| 回数 | 渡す | トレイからの回収 | 回収に失敗して人に知らせた | 台帳が実物と合っていた |", "|---|---|---|---|---|",
             f"| {len(rows)} (乱数の種 {len(runs)} 通り × 30) | {sum(r['handed'] for r in rows)} / {len(rows)} | "
             f"{sum(r['retrieved'] for r in rows)} / {len(rows)} | {sum(r['told'] for r in fails)} / {len(fails)} | "
             f"{sum(r['ledger_ok'] for r in rows)} / {len(rows)} |", ""]
    if fails:
        lines += [f"回収に失敗した回は、トレイの置き場 {sorted({r['tray_slot'] for r in fails})} (術者側の壁際) に置かれた器械。"
                  "確かめた例 (持針器) では壁まで約 2cm しかなく、指が入らなかった。どの回も 2 回試したあと人に知らせ、台帳はトレイのまま (実物と同じ)。", ""]
    return lines


def figures() -> None:
    """README・レポート用の図。大きすぎない大きさにして docs/media/ に置く。"""
    MEDIA.mkdir(parents=True, exist_ok=True)

    def save(src: Path, name: str, width: int) -> None:
        img = cv2.imread(str(src))
        h = int(img.shape[0] * width / img.shape[1])
        cv2.imwrite(str(MEDIA / name), cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    save(OUT / "phase2/eval_crowd_on_main/predictions.jpg", "detect_predictions.jpg", 1200)
    save(OUT / "phase2/eval_crowd_on_main/confusion_matrix_normalized.png", "confusion_matrix.jpg", 900)
    save(OUT / "phase2/labels_train.jpg", "synthetic_labels.jpg", 1200)
    save(OUT / "phase3/locate_examples.jpg", "locate_examples.jpg", 1200)
    save(OUT / "phase5/count_examples_crowd.jpg", "count_examples.jpg", 1200)

    # デモ動画のコマ: 受け渡し・並べ直し・警告・カウント一致 (本編 main.mp4 の、ログに残った時刻から選ぶ)
    from demo.video import SPEED
    log = json.loads((OUT / "phase6/demo_log.json").read_text())
    rd = imageio.get_reader(OUT / "phase6/main.mp4")

    def frame_at(sim_t: float) -> np.ndarray:
        return rd.get_data(int(sim_t / SPEED * 30))[..., ::-1]

    t_mesu = next(x["t"] for x in log if x.get("robot") == "メスです")
    t_rearr = next(x["t"] for x in log if "note" in x and "器械台のスロット" in x["note"])
    t_alert = next(x["t"] for x in log if x.get("robot", "").startswith("持針器1本未返却"))
    t_ok = next(x["t"] for x in log if x.get("robot", "").startswith("カウント一致"))
    shots = {"demo_handoff.jpg": t_mesu + 0.5, "demo_rearrange.jpg": t_rearr + 3.0,
             "demo_alert.jpg": t_alert + 2.0, "demo_ok.jpg": t_ok + 1.5}
    for name, t in shots.items():
        cv2.imwrite(str(MEDIA / name), cv2.resize(frame_at(t), (1280, 720), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    # README の先頭に置く短い GIF (受け渡しの場面、640px・10fps)
    # 全体カメラの部分だけを 480px に縮め、シミュレーター内 0.3 秒ごとのコマを 0.2 秒ずつ見せる (1.5 倍速)。大きさは 3MB 程度
    frames = [cv2.resize(rd.get_data(int(t / SPEED * 30))[:810, :1440], (480, 270), interpolation=cv2.INTER_AREA)
              for t in np.arange(t_mesu - 7.0, t_mesu + 5.0, 0.3)]
    imageio.mimsave(MEDIA / "demo_handoff.gif", frames, duration=0.2, loop=0)


def main() -> None:
    parts = ["# 評価結果のまとめ", "",
             "すべてシミュレーター (MuJoCo) 内で計測。`uv run python -m scripts.export_results` で outputs/ の評価結果から作り直せる。", ""]
    for f in (detection, locate, pick_place, language, counting, handoff):
        parts += f()
    (DOCS / "results.md").write_text("\n".join(parts))
    figures()
    print((DOCS / "results.md").read_text())
    for p in sorted(MEDIA.iterdir()):
        print(f"{p}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
