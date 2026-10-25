"""Phase 2: YOLO-seg (yolo11n-seg) を合成データで学習する (Mac の MPS = Apple GPU を使う)。

    uv run python -m perception.train                                  # 本番
    uv run python -m perception.train --epochs 5 --name quick          # 動作確認
    uv run python -m perception.train --data datasets/norand/data.yaml --name norand   # 実験: ランダム化なし

出発点は COCO (日常物体 80 種) で学習済みの重み。輪郭やエッジを拾う前半の層はそのまま使えるので、
ゼロから学習するより少ない枚数・短い時間で済む (転移学習)。
学習結果は runs/segment/<name>/ に出る。一番良かった重みは weights/best.pt。
"""

import argparse
import pickle
import re
from pathlib import Path

import torch
from ultralytics import YOLO

WEIGHTS_DIR = Path("perception/weights")
# ultralytics は相対パスの project を自分の runs フォルダの下に作ってしまう (runs/segment/runs/segment/...) ので絶対パスで渡す
RUNS_DIR = Path("runs/segment").resolve()


def export_weights(src: Path, dst: Path) -> None:
    """公開リポジトリに同梱する重みを書き出す。
    ultralytics の checkpoint には学習時の絶対パス (= PC のユーザー名入り) と git 情報が入っているので消す。
    推論に要らない optimizer の状態も落とす。"""
    ck = torch.load(src, weights_only=False, map_location="cpu")
    ck["optimizer"] = None
    ck["git"] = None
    home = str(Path.home())

    def scrub(d: dict) -> dict:
        return {k: (v.replace(str(Path.cwd()) + "/", "").replace(home, "~") if isinstance(v, str) else v)
                for k, v in d.items()}

    ck["train_args"] = scrub(ck["train_args"])
    for key in ("model", "ema"):  # モデル本体も学習時の設定 (args) を持っている
        if ck.get(key) is not None and isinstance(getattr(ck[key], "args", None), dict):
            ck[key].args = scrub(ck[key].args)
    leaked = set(re.findall(re.escape(home).encode() + rb"[^\x00 ]{0,40}", pickle.dumps(ck)))
    if leaked:
        raise RuntimeError(f"重みにまだ絶対パスが残っている: {leaked}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"exported → {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/instruments/data.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="instruments")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0 (CUDA)")
    ap.add_argument("--export", action="store_true", help="best.pt を perception/weights/ に書き出す (本番用)")
    ap.add_argument("--export-only", action="store_true", help="学習せず、既存の best.pt を書き出すだけ")
    args = ap.parse_args()
    if args.export_only:
        export_weights(RUNS_DIR / args.name / "weights" / "best.pt", WEIGHTS_DIR / "instruments_seg.pt")
        return

    model = YOLO("yolo11n-seg.pt")  # 初回は ultralytics の公式リリースから自動ダウンロード (約 6MB)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        seed=0,
        workers=4,
        patience=10,  # 検証 mAP が 10 エポック改善しなければ打ち切る
        # データ拡張 (学習時に画像をその場で加工する)。合成データ側のランダム化と役割を分けている:
        #   合成側 = シーンそのもの (照明・色・配置) / ここ = 画像としての加工
        degrees=10.0,  # 回転 ±10°: 器械の置き向きのばらつき
        scale=0.2,  # 拡大縮小 ±20%: カメラは固定なので、写る大きさはあまり変わらない
        flipud=0.5,  # 上下反転: 真上からのカメラなので、器械の先端が上向きでも下向きでも同じ物
        fliplr=0.5,
        mosaic=1.0,  # 4 枚を貼り合わせる (1 枚に写る器械の数や組み合わせを増やす)
        close_mosaic=10,  # 最後の 10 エポックは貼り合わせをやめ、本物の構図で仕上げる
        plots=True,
    )
    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")
    if args.export:
        export_weights(best, WEIGHTS_DIR / "instruments_seg.pt")


if __name__ == "__main__":
    main()
