# 学習ログ

各フェーズの「学んだ概念 / 設計判断と理由 / 実験と結果 / Q&A の要点」を簡潔に残す。
最終レポートの「工夫した点」「考察」の素材にする。

## Phase 0: 環境構築・Panda 読み込み・ビューア (2026-10-24)

### 学んだ概念
- **MJCF の構造**: `<worldbody>` 以下の body の入れ子が運動学ツリーそのもの。関節 (`<joint>`) は
  「親 body に対して子 body がどう動けるか」を定義する。Panda は link0→…→link7→hand→指 の直列鎖。
- **qpos / ctrl**: `qpos` は一般化座標 (Panda はヒンジ 7 + 指スライド 2 = 9 次元)。
  `ctrl` はアクチュエータへの入力。Menagerie の Panda は**位置サーボ**
  (`force = kp(ctrl − q) − kd·q̇`) なので、ctrl には目標関節角を入れる。
- **順運動学 (FK)**: 関節角 → 手先の位置姿勢。`mj_forward` で計算され、`data.xpos/xmat` に入る。
  home 姿勢で TCP = (0.554, 0, 0.521) m、TCP z 軸 = (0, 0, −1) = 真下向き。
- **mj_forward と mj_step**: forward は時間を進めずに状態から派生量 (姿勢・接触・力) を計算、
  step は timestep (2 ms) だけ積分して時間を進める。
- **オフスクリーン描画**: `mujoco.Renderer` はウィンドウ無しで描画できる。動画・合成データ生成はこちらを使う。

### 設計判断と理由
- Python 3.12 に固定: mujoco / ultralytics / torch(MPS) / mlx-whisper の wheel が揃っていて最も安全。
- Panda は `robot_descriptions` 経由で取得: commit がパッケージのバージョンで固定される
  (現在の Menagerie commit: `feadf76`)。自前でコピーするより再現性が高い。
- TCP 用の site が Menagerie の hand に無いので、hand 原点から z 方向 0.1034 m (Franka 公式) を TCP と定義 (`sim/robot.py`)。

### 詰まった点と対処
- `mjpython` が `Library not loaded: @rpath/libpython3.12.dylib` で落ちる。
  uv の Python (python-build-standalone) は libpython を rpath で引くが、mjpython の .app 経由では解決できない。
  → `DYLD_LIBRARY_PATH` に本体 Python の `lib/` を足すラッパ `scripts/mjpython.sh` で回避。

### 実験と結果
- **PD ゲインと追従誤差** (`--kp-scale 1 / 0.1 / 0.01`): 1 倍でも RMS 約 2.5°、0.01 倍では腕が持ち上がらず垂れた。
  位置サーボは「バネ」なので、重力トルクを支えるには ズレ = 重力トルク / kp が必要。kp が小さいほど大きく垂れる。
  → 把持精度に効く。実機は重力補償で打ち消す (Phase 1 で導入)。
- **肘だけ動かす FK**: 肘 −1.57→−2.4 (≈48°) で TCP は手前・下へ移動し、指の向きも真下から ≈47° 傾いた。
  home 姿勢では joint2/4/6 の軸が平行なので、肘の回転がそのまま手先の傾きになる。
  手首 (joint6) で向きを戻すと今度は TCP が 18cm 前へずれる → 位置と姿勢は複数関節で同時に合わせる必要がある = IK が要る理由。
- 外力 (Ctrl+右ドラッグ) で押すと、kp を下げた腕はバネのようにたわんで戻る。

### Q&A
- Q. 物を取るたびに関節角を人が指示する? → No。音声→LLM→検出→動作計画→**IK**→ctrl と上位から翻訳される。
  ctrl (関節角) は最下層のインタフェース。人間が考えやすい「手先の位置」とロボットが受け付ける「関節角」をつなぐのが IK。
- 公開用のコミット作者は GitHub の noreply アドレスにし、個人メールを履歴に残さない。
