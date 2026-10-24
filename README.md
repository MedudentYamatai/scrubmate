# Scrub Nurse Robot (MuJoCo)

手術室の「器械出し看護師」役をするロボットアームの MuJoCo シミュレーション。
音声指示で器械を受け渡しし、画像ベースの器械カウントで遺残疑いを検出する。

> 開発中 (Phase 1: シーン構築・IK・真値での pick & place)。完成時に構成図・評価結果・制限事項を追記する。

## セットアップ

前提: macOS (Apple Silicon), [uv](https://docs.astral.sh/uv/)

```bash
uv sync                 # Python 3.12 + 依存を .venv に入れる
cp .env.example .env    # API キー等 (Phase 4 以降)
```

初回実行時に `robot_descriptions` が MuJoCo Menagerie を `~/.cache/robot_descriptions/` に clone する。

## 実行

```bash
# Panda の構造・順運動学を表示し、オフスクリーン描画を outputs/phase0/ に保存
uv run python -m sim.inspect_panda

# ビューア表示 (Mac は mjpython 必須。ラッパ経由で起動)
./scripts/mjpython.sh -m sim.view_panda           # 関節を正弦波で動かす
./scripts/mjpython.sh -m sim.view_panda --static  # home 姿勢で静止
```

### Phase 1: シーン・IK・pick & place

```bash
# 真値を使った pick & place (5 本 × 器械台⇄トレイ往復) の計測。--video で outputs/phase1/pick_place.mp4
uv run python -m control.pick_place_demo [--video] [--only scalpel] [--no-gravcomp] [--grip-scale 20]
./scripts/mjpython.sh -m control.pick_place_demo --viewer   # ビューアで実時間再生

# 遠隔操作 (目標マーカーを動かすと IK で腕が追従)。キー操作は起動したターミナルで
./scripts/mjpython.sh -m control.teleop
```

| キー (ターミナル) | 動作 |
|---|---|
| `w`/`s`, `a`/`d`, `r`/`f` | 目標を 前後 / 左右 / 上下 |
| `q`/`e` | 手首を回す (yaw) |
| `space` | グリッパ開閉 |
| `1`/`2`/`3` | 移動量 1 / 5 / 20 mm |
| `h` / `z` / `x` | home へ / シーン初期化 / 終了 |

ビューア上ではオレンジの球をダブルクリック → Ctrl+右ドラッグ (移動) / Ctrl+左ドラッグ (回転) でも動かせる。

`scripts/mjpython.sh` が必要な理由: uv の Python は `libpython` を `@rpath` で参照しており、
`mjpython` の .app バンドル経由だと解決できないため、`DYLD_LIBRARY_PATH` を補っている。

## ディレクトリ構成

| ディレクトリ | 役割 |
|---|---|
| `sim/` | シーン XML (`assets/or_scene.xml`)・器械の形状 (`instruments.py`)・シーン組み立て (`scene.py`)・描画 |
| `control/` | 微分 IK (`ik.py`)・軌道生成 (`motion.py`)・pick & place・遠隔操作 |
| `perception/` | 合成データ生成・YOLO-seg 学習・推論 |
| `language/` | Whisper・LLM による指示解釈 |
| `inventory/` | 器械台帳・カウント |
| `demo/` | シナリオ実行・動画出力 |
| `assets/` | 音声などの素材 |
| `docs/` | 学習ログ・レポート素材 |

## クレジット

- Franka Emika Panda モデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0)
