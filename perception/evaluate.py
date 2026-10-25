"""Phase 2: 学習した検出器を検証データで評価する (mAP・器械ごとの成績・混同行列・予測の見本画像)。

    uv run python -m perception.evaluate                                         # 本番の重み × 本番の検証データ
    uv run python -m perception.evaluate --weights runs/segment/norand/weights/best.pt --tag norand

出力: outputs/phase2/eval_<tag>/
  metrics.json             … mAP (box / mask) と器械ごとの値
  confusion_matrix*.png    … 行 = 予測、列 = 正解。対角線以外が「取り違え」、background 行 = 見逃し、
                             background 列 = 誤検出 (何も無い所に器械があると言った)。確信度 0.25 以上の予測で数える
  count_accuracy           … 画像ごとに「どの器械が何本あるか」が正解と完全に一致した割合 (器械カウントに直結する指標)
  predictions.jpg          … 検証画像への予測結果 (色付きの輪郭 + 器械名 + 確信度)

mAP の読み方:
  mAP50     … 予測と正解の重なり (IoU) が 50% 以上なら当たり、としたときの平均精度。「見つけられたか」
  mAP50-95  … IoU 50%〜95% の厳しさで平均。「どれだけ正確な形・位置で見つけられたか」
  box = 枠の精度 / mask = 輪郭の精度。Phase 3 では mask から向きを出すので mask の値が大事。
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from perception.show_labels import COLORS
from sim.instruments import INSTRUMENT_NAMES

DEFAULT_WEIGHTS = "perception/weights/instruments_seg.pt"
CONF = 0.25  # 実際に使うときの確信度しきい値


def count_accuracy(model: YOLO, data_yaml: Path, imgsz: int) -> dict[str, float]:
    """各検証画像で、予測した器械の種類と本数が正解ラベルと完全に一致した割合。
    腕の状態 (待避中 park / 器械台の上 over_table) 別にも出す。腕に隠れると成績が落ちるかを見るため。"""
    root = data_yaml.parent
    arm_of = {}
    meta = root / "meta" / "val.jsonl"
    if meta.exists():
        for line in meta.read_text().splitlines():
            d = json.loads(line)
            arm_of[d["image"]] = d["arm"]
    hits: dict[str, list[int]] = {}
    for p in sorted((root / "images" / "val").glob("*.jpg")):
        lab = (root / "labels" / "val" / f"{p.stem}.txt").read_text().split("\n")
        truth = sorted(int(line.split()[0]) for line in lab if line.strip())
        r = model.predict(str(p), imgsz=imgsz, conf=CONF, verbose=False)[0]
        ok = sorted(r.boxes.cls.int().tolist()) == truth
        for key in ("all", arm_of.get(p.name, "unknown")):
            hits.setdefault(key, []).append(ok)
    return {k: round(sum(v) / len(v), 4) for k, v in hits.items()} | {f"n_{k}": len(v) for k, v in hits.items()}


def prediction_grid(model: YOLO, images: list[Path], out: Path, cols: int = 4) -> None:
    tiles = []
    for p in images:
        img = cv2.imread(str(p))
        r = model.predict(img, verbose=False, conf=CONF)[0]
        over = img.copy()
        if r.masks is not None:
            for poly, cls, conf in zip(r.masks.xy, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                pts = poly.astype(np.int32)
                cv2.fillPoly(over, [pts], COLORS[cls])
                cv2.polylines(img, [pts], True, COLORS[cls], 2)
                x, y = pts.min(0)
                cv2.putText(img, f"{INSTRUMENT_NAMES[cls]} {conf:.2f}", (int(x), int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
        tiles.append(cv2.resize(cv2.addWeighted(over, 0.35, img, 0.65, 0), (512, 384)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", default="datasets/instruments/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    out = Path("outputs/phase2") / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val",
                  project=str(Path("runs/segment").resolve()), name=f"val_{args.tag}", exist_ok=True, plots=True, verbose=False)

    # mAP は確信度の低い予測まで全部使って計算する (しきい値を動かしたときの総合成績なので)。
    # 一方、混同行列は実際に使うしきい値 (0.25) で数えないと、確信度 0.001 のゴミ予測が「誤検出」として大量に入ってしまう
    cm = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val", conf=CONF,
                   project=str(Path("runs/segment").resolve()), name=f"val_{args.tag}_cm", exist_ok=True,
                   plots=True, verbose=False)

    per_class = {}
    for i, cls in enumerate(m.box.ap_class_index):
        name = INSTRUMENT_NAMES[int(cls)]
        per_class[name] = {"box_mAP50": round(float(m.box.ap50[i]), 4), "mask_mAP50": round(float(m.seg.ap50[i]), 4),
                           "mask_mAP50-95": round(float(m.seg.ap[i]), 4)}
    res = {
        "weights": args.weights, "data": args.data,
        "box_mAP50": round(float(m.box.map50), 4), "box_mAP50-95": round(float(m.box.map), 4),
        "mask_mAP50": round(float(m.seg.map50), 4), "mask_mAP50-95": round(float(m.seg.map), 4),
        "count_accuracy": count_accuracy(model, Path(args.data), args.imgsz),
        "per_class": per_class,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(cm.save_dir) / f
        if src.exists():
            shutil.copy(src, out / f)

    val_dir = Path(args.data).parent / "images" / "val"
    prediction_grid(model, sorted(val_dir.glob("*.jpg"))[:12], out / "predictions.jpg")

    print(f"\n{'':14s} box mAP50  mask mAP50  mask mAP50-95")
    for name, v in per_class.items():
        print(f"{name:14s} {v['box_mAP50']:9.3f}  {v['mask_mAP50']:10.3f}  {v['mask_mAP50-95']:13.3f}")
    print(f"{'全体':12s} {res['box_mAP50']:9.3f}  {res['mask_mAP50']:10.3f}  {res['mask_mAP50-95']:13.3f}")
    ca = res["count_accuracy"]
    print("画像ごとの器械の種類・本数が完全一致: " + "  ".join(
        f"{k} {v * 100:.1f}% ({ca['n_' + k]}枚)" for k, v in ca.items() if not k.startswith("n_")))
    print(f"saved {out}/")


if __name__ == "__main__":
    main()
