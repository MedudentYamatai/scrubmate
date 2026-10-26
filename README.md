# Scrub Nurse Robot (MuJoCo)

手術室の「器械出し看護師」役をするロボットアームの MuJoCo シミュレーション。
音声指示で器械を受け渡しし、画像ベースの器械カウントで遺残疑いを検出する。

> 開発中 (Phase 4: 音声認識と指示の解釈 まで)。完成時に構成図・評価結果・制限事項を追記する。

## セットアップ

前提: macOS (Apple Silicon), [uv](https://docs.astral.sh/uv/)

```bash
uv sync                 # Python 3.12 + 依存を .venv に入れる
cp .env.example .env    # GEMINI_API_KEY を入れる (無くてもキャッシュと予備のルールで動く)
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
# pick & place (5 本 × 器械台⇄トレイ往復) の計測。Phase 3 以降はカメラの推定で掴む (--gt で真値)
uv run python -m control.pick_place_demo [--video] [--only scalpel] [--gt] [--no-gravcomp] [--grip-scale 20]
./scripts/mjpython.sh -m control.pick_place_demo --viewer   # ビューアで実時間再生

# シーンの確認用の静止画 (全体カメラ・検出カメラ) → outputs/phase1/
uv run python -m sim.snapshots

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

### Phase 2: 合成データ生成と YOLO-seg 学習

```bash
uv run python -m perception.gen_dataset                 # 1500 枚 (train 1200 / val 300) → datasets/instruments/ (約 8 分)
uv run python -m perception.show_labels datasets/instruments --split train   # 正解ラベルを重ねた確認画像
uv run python -m perception.train --export              # yolo11n-seg を MPS で 30 エポック (約 45 分)
uv run python -m perception.evaluate                    # mAP・混同行列・カウント一致率 → outputs/phase2/eval_main/
```

同梱の学習済み重み `perception/weights/instruments_seg.pt` (6MB) は、Phase 3 で作り直したもの。
「ずれ 2 倍 + 返却トレイで器械が重なる画像」で学習している (下の Phase 3 の追加実験を参照)。
再現するには、次を実行する:

```bash
uv run python -m perception.gen_dataset -n 600 --spread 2 --crowd 0.5 --out datasets/crowd
uv run python -m perception.train --data datasets/crowd/data.yaml --name crowd --export
```

この重みがあれば、学習をやり直さなくても評価・推論できる。学習をやり直さなくても評価・推論できる。
書き出し時に、学習時の絶対パスと git 情報を checkpoint から取り除いている (`perception/train.py` の `export_weights`)。

### Phase 3: カメラの推定で掴む

```bash
uv run python -m perception.eval_locate                     # 推定精度 (ランダム配置 50 通り) → outputs/phase3/locate_*.{json,jpg}
uv run python -m control.pick_place_demo --random 1 --video # ランダム配置で往復 → outputs/phase3/pick_place_rand1.mp4
uv run python -m control.pick_place_demo --trials 30        # ランダム配置 30 通り (300 回) をまとめて集計 (約 3.5 分)
```

掴む前に毎回、腕を待避させて検出カメラで RGB と深度を撮り、`perception/locate.py` で器械ごとに推定する。

1. YOLO-seg でマスクを得る。深度を使って、マスクから欠けた器械の縁を補う。
2. 各画素を深度で 3D に戻し、真上から見た地図 (1mm 方眼) に描き直す。こうすると透視による歪みが消える。
3. 地図上で求める:
   - 把持点: 重心
   - 向き: 主成分分析 (PCA)
   - 先端側: 外形の幅のプロファイルを形状モデルと照合
4. 掴み方と手首の回し方を `control/pick_place.py` の `plan_transfer` で決める。Panda の手首は ±166° しか回らないので、限界に当たらない組み合わせを選ぶ。

先端の判定に自信が無いときは、YOLO を使わずに深度だけで作ったシルエットで判定し直す。
深度だけのシルエットは、器械が無いときの深度画像 (背景) より手前にある所を器械とみなして作る。
それでも自信が無ければ「先端不明」と報告する。

シミュレーターの真値は `--gt` での比較と評価にだけ使う。

結果:

- 推定精度 (250 回): 位置誤差は平均 2.2mm、先端方向の正解率は 100%
- pick & place (ランダム配置 30 通り): 300/300 回成功

返却トレイは器械台の術野側の端にある (術者の手から約 0.2m)。
最初は反対側に置いていたが、術者の手が届かないので移した。数字はすべて移した後に測り直したもの。

### Phase 4: 音声認識と指示の解釈

```bash
uv run python -m language.make_audio --voices all     # 台詞の音声を macOS の読み上げで作る (デモ用は同梱済み)
uv run python -m language.interpret --wav assets/audio/demo/d6_are.wav --step 止血 --with forceps pean
uv run python -m language.interpret --text "次" --step 剥離 --with forceps     # 文字で入力 (デバッグ用)
uv run python -m language.eval_language                # 5 種類の声で評価 → outputs/phase4/
```

処理は次の流れで進む。

1. **音声 → 文字**: mlx-whisper (large-v3-turbo) で、Mac の中だけで書き起こす。
   初回はモデル (約 1.6GB) をダウンロードする。
2. **読みでの補正**: 書き起こしをひらがなに直し、器械名 5 つ (メス・ハサミ・ペアン・鑷子・持針器) と工程名 4 つの読みに近い部分を、正しい名前に置き換える。
   例: 接し → 鑷子、自信機 → 持針器、方向 → 縫合。
3. **意図の解釈**: Gemini (`.env` の `GEMINI_MODEL`) に、決まった形の JSON で答えさせる。
   形は `{intent, instrument, step, reason}`。
   「次」「縫合するのであれ取って」は、術式の工程表と、術者が今持っている器械から推定させる。

無料枠で止まらないように、2 つの仕組みを入れている。

- **キャッシュ**: 同じ文・同じ状況なら、前回の答えを `language/cache/` から返す。
  キャッシュは同梱しているので、API キーが無くてもデモは同じ結果で動く。
- **予備**: 回数制限 (429)・通信エラー・20 秒の時間切れのときは、キーワードのルールで解釈する。

術者の声は、macOS の読み上げ音声 (Kyoko) で作った。
声は個人を特定できる情報になりうるので、自分の声は録音していない。

結果 (23 台詞):

- 器械名の聞き取り: 補正前 9/18 → 補正後 18/18
- 解釈: Gemini 23/23、予備 23/23
- 他の 4 種類の声では、補正後 11〜16/18 にとどまった (`docs/learning_log.md`)

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
