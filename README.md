# Scrub Nurse Robot (MuJoCo)

手術室の「器械出し看護師」役をするロボットアームの MuJoCo シミュレーション。
音声指示で器械を受け渡しし、画像ベースの器械カウントで遺残疑いを検出する。

> 開発中 (Phase 0: 環境構築)。完成時に構成図・評価結果・制限事項を追記する。

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

`scripts/mjpython.sh` が必要な理由: uv の Python は `libpython` を `@rpath` で参照しており、
`mjpython` の .app バンドル経由だと解決できないため、`DYLD_LIBRARY_PATH` を補っている。

## ディレクトリ構成

| ディレクトリ | 役割 |
|---|---|
| `sim/` | シーン XML・環境・ロボット読み込み |
| `control/` | IK・動作計画 |
| `perception/` | 合成データ生成・YOLO-seg 学習・推論 |
| `language/` | Whisper・LLM による指示解釈 |
| `inventory/` | 器械台帳・カウント |
| `demo/` | シナリオ実行・動画出力 |
| `assets/` | 音声などの素材 |
| `docs/` | 学習ログ・レポート素材 |

## クレジット

- Franka Emika Panda モデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0)
