"""正解ラベル (輪郭) を画像に重ねて並べ、自動ラベリングが正しいかを目で確かめる。

    uv run python -m perception.show_labels datasets/instruments --split train -k 12
    → outputs/phase2/labels_<split>.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sim.instruments import INSTRUMENT_NAMES

# 器械ごとの表示色 (BGR)。表示用なので色付きにしている (シーン内の器械は全部同じ金属色)
COLORS = [(60, 60, 255), (60, 200, 60), (255, 150, 40), (40, 220, 240), (220, 80, 220)]


def draw_labels(img: np.ndarray, label_file: Path) -> np.ndarray:
    h, w = img.shape[:2]
    over = img.copy()
    for line in label_file.read_text().splitlines():
        v = line.split()
        cls = int(v[0])
        pts = (np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(over, [pts], COLORS[cls])
        cv2.polylines(img, [pts], True, COLORS[cls], 2)
        x, y = pts.min(0)
        cv2.putText(img, INSTRUMENT_NAMES[cls], (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
    return cv2.addWeighted(over, 0.35, img, 0.65, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("-k", type=int, default=12, help="並べる枚数")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    imgs = sorted((args.root / "images" / args.split).glob("*.jpg"))[: args.k]
    tiles = []
    for p in imgs:
        img = cv2.imread(str(p))
        img = draw_labels(img, args.root / "labels" / args.split / f"{p.stem}.txt")
        tiles.append(cv2.resize(img, (512, 384)))
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    out = Path("outputs/phase2") / f"labels_{args.split}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
