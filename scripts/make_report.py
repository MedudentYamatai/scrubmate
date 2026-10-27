"""Phase 7: 説明資料 (レポート) を、動画のスライドと同じ見た目で作り、PDF にまとめる。

    uv run python -m scripts.make_report                       # → outputs/report/MedudentYamatai.pdf
    uv run python -m scripts.make_report --video-url https://youtu.be/xxxx   # 動画の URL を変える

数字は docs/results.md と同じ評価結果 (scripts/export_results.py が outputs/ から集めたもの) を使う。
文字も画像として描いてから PDF にする (提出先で「フォントが原因で正しく表示されない」ことを避けるため)。
1 枚目は講座から配られた表紙 (assets/branding/slide_cover.jpg)。公開リポジトリには含めないので、無ければ省く。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from demo.slides import (BG, CARD, INK, INK2, INK3, OLIVE, OLIVE_DARK, Slide, _bands, _fit, _font, _heading,
                         background_slide, results_slide)

OUT = Path("outputs/report")
MEDIA = Path("docs/media")
COVER = Path("assets/branding/slide_cover.jpg")
REPO_URL = "https://github.com/MedudentYamatai/scrubmate"
ACCOUNT = "MedudentYamatai"
VIDEO_URL = "https://youtu.be/QuQ808LdvgI"  # 提出した動画 (YouTube)


def img(name: str) -> np.ndarray:
    im = cv2.imread(str(MEDIA / name))[..., ::-1]
    if name.startswith("demo_"):
        im = im[:int(im.shape[0] * 810 / 1080)]  # 動画のコマは、下の字幕の段 (黒い帯) を切り落とす
    return im


def bullets(s: Slide, xy, items: list[str], width: int, size: int = 30, gap: float = 0.6) -> int:
    """箇条書き (行頭に差し色の小さな四角)。最後の y を返す。"""
    x, y = xy
    for it in items:
        s.rect(0, (x, y + size // 2 - 5, x + 10, y + size // 2 + 5), OLIVE_DARK, dx=0)
        n = s.para(0, (x + 26, y), it, width=width - 26, size=size, color=INK, dx=0)
        y += int(n * size * 1.45 + size * gap)
    return y


def picture(s: Slide, xy, image: np.ndarray, w: int, h: int, caption: str | None = None) -> None:
    x, y = xy
    s.rect(0, (x - 2, y - 2, x + w + 2, y + h + 2), (225, 225, 220), dx=0)
    s.image(0, (x, y), _fit(image, w, h), dx=0)
    if caption:
        s.para(0, (x, y + h + 10), caption, width=w, size=20, color=INK3, dx=0)


def stat(s: Slide, xy, num: str, label: str, w: int = 330) -> None:
    x, y = xy
    s.rect(0, (x, y, x + w, y + 150), CARD, radius=14, outline=(225, 225, 220), dx=0)
    s.rect(0, (x, y, x + 8, y + 150), OLIVE, dx=0)
    s.text(0, (x + 28, y + 14), num, size=58, bold=True, dx=0)
    s.para(0, (x + 28, y + 92), label, width=w - 40, size=21, color=INK2, dx=0)


def page(title_en: str, title_ja: str) -> Slide:
    s = Slide(1.0)
    _bands(s)
    _heading(s, 0, title_en, title_ja)
    return s


# ---- 各ページ ----
def p_title(video_url: str | None) -> Slide:
    s = Slide(1.0)
    _bands(s, big=True)
    s.text(0, (130, 220), "PHYSICAL AI 応用編 1  最終課題", size=30, bold=True, color=OLIVE_DARK)
    s.text(0, (122, 270), "ScrubMate", size=170, bold=True)
    s.text(0, (132, 485), "声で渡し、目で数える器械出しロボット", size=48, bold=True, color=INK2)
    s.text(0, (134, 600), "MuJoCo シミュレーション / Franka Emika Panda", size=30, color=INK2)
    s.text(0, (134, 700), f"GitHub: {REPO_URL}", size=28, color=INK)
    s.text(0, (134, 750), f"動画: {video_url or '(YouTube の URL は提出フォームを参照)'}", size=28, color=INK)
    s.text(0, (134, 860), ACCOUNT, size=28, color=INK3)
    return s


def p_system() -> Slide:
    s = page("SYSTEM", "システム構成: 5 つの部品を、状態機械の頭脳がつなぐ")
    boxes = [("聞く", "Whisper で書き起こし\n読みで器械名を補正"), ("解釈する", "Gemini が意図を\nJSON で返す"),
             ("頭脳", "状態機械\n渡す / 回収 / カウント"), ("見つける", "YOLO-seg + 深度\n把持点・向き・先端"),
             ("動かす", "微分 IK\n物理接触で掴む")]
    x0, y0, w, h, g = 130, 280, 290, 200, 40
    for i, (t, d) in enumerate(boxes):
        x = x0 + i * (w + g)
        col = OLIVE if t == "頭脳" else CARD
        s.rect(0, (x, y0, x + w, y0 + h), col, radius=16, outline=(210, 210, 200), dx=0)
        s.text(0, (x + 24, y0 + 22), t, size=40, bold=True, dx=0)
        for j, ln in enumerate(d.split("\n")):
            s.text(0, (x + 24, y0 + 92 + j * 36), ln, size=24, color=INK if t == "頭脳" else INK2, dx=0)
        if i < len(boxes) - 1:
            ax = x + w + 6
            s.poly(0, [(ax, y0 + h / 2 - 12), (ax + g - 12, y0 + h / 2), (ax, y0 + h / 2 + 12)], OLIVE_DARK, dx=0)
    s.rect(0, (x0 + 2 * (w + g), y0 + h + 30, x0 + 3 * (w + g) - g, y0 + h + 130), CARD, radius=14,
           outline=(210, 210, 200), dx=0)
    s.text(0, (x0 + 2 * (w + g) + 24, y0 + h + 44), "台帳", size=34, bold=True, dx=0)
    s.text(0, (x0 + 2 * (w + g) + 24, y0 + h + 92), "器械台 / 術者 / トレイ", size=22, color=INK2, dx=0)
    bullets(s, (130, 660), [
        "把持の目標はすべてカメラの推定から得る。シミュレーターの真値は評価にだけ使う。",
        "ロボットの把持は物理接触 (摩擦)。weld 拘束は、術者の手が器械を持つ瞬間 (術者はスクリプト) だけに使う。",
        "器械 5 種は全部同じ金属色で、識別マーカーは付けない。上から見た形だけで見分ける。",
        "Gemini 以外 (音声認識・検出・制御) は、すべて Mac (Apple Silicon) の上で動く。",
    ], width=1560, size=28)
    return s


def p_scene() -> Slide:
    s = page("SCENE", "シーンの設計: 掴みやすく、数えやすい手術室")
    picture(s, (130, 250), img("demo_handoff.jpg"), 900, 506, "全体カメラ。右の小窓は真上の検出カメラ (RGB + 深度)")
    bullets(s, (1080, 260), [
        "器械台は 2 本の浮いた支持バーで器械を支える。指が器械の下に回り込める",
        "返却トレイにも同じバーを入れ、術者の手から約 20cm の位置に置いた (最初は 89cm 離れていて手が届かなかった)",
        "トレイの置き場の間隔は 6cm。7.5cm では端の置き場が壁に近すぎ、5 種類とも掴めなかった",
        "器械 5 種 (メス・持針器・鑷子・ハサミ・ペアン) はプリミティブの組み合わせで、厚さ約 1cm",
    ], width=660, size=26)
    return s


def p_perception() -> Slide:
    s = page("PERCEPTION", "知覚: 合成データで学習し、深度で把持点を出す")
    picture(s, (130, 250), img("synthetic_labels.jpg"), 760, 380, "合成データ (自動ラベル。置き方・照明・色・カメラ位置を毎枚変える)")
    picture(s, (930, 250), img("locate_examples.jpg"), 760, 380, "推定 (色付きの矢印) と真値 (白い矢印)")
    bullets(s, (130, 700), [
        "MuJoCo の segmentation 描画で正解ラベルを自動で作る。ランダム化を切ると mask mAP50-95 が 0.855 → 0.783",
        "マスクを深度で 3D に戻し、真上から見た地図にしてから重心・向き (PCA)・先端 (幅のプロファイル) を出す",
    ], width=1560, size=26)
    for i, (n, l) in enumerate([("0.995", "検出 mask mAP50"), ("2.2mm", "把持点の誤差 (平均)"), ("100%", "先端の向き")]):
        stat(s, (130 + i * 380, 880), n, l, w=350)
    return s


def p_grasp() -> Slide:
    s = page("GRASP & HANDOFF", "把持と受け渡し: 手首の可動域まで考えて掴む")
    picture(s, (130, 250), img("demo_rearrange.jpg"), 820, 461, "閉創前の並べ直し (重なった鑷子とペアンを 1 本ずつ器械台へ)")
    bullets(s, (1000, 260), [
        "Panda の手首は ±166° しか回らない。掴み方 2 通り × 回す向き 2 通りから、限界に当たらない組み合わせを選ぶ",
        "器械台には先端をそろえて戻す。トレイで逆向きだった 45 回も全部正しい向きで戻した",
        "術者には持ち手を向けて差し出す",
        "重なりの上の器械は掴む高さを 4.5mm 上げる。壁際は指を狭める",
    ], width=740, size=26)
    for i, (n, l) in enumerate([("300/300", "pick & place 成功 (カメラの推定)"), ("90/90", "術者に渡す"),
                                ("87/90", "トレイから回収 (失敗 3 回は人に知らせた)")]):
        stat(s, (130 + i * 380, 790), n, l, w=350)
    return s


def p_language() -> Slide:
    s = page("VOICE & INTENT", "音声と指示の解釈: 専門用語を読みで直し、工程から推定する")
    rows = [("鑷子", "接し"), ("持針器", "自信機 / 受信機"), ("縫合", "方向 / 報告"), ("ペアン", "レアン")]
    s.text(0, (130, 250), "Whisper の書き起こし → 読みで補正", size=30, bold=True)
    for i, (ok, bad) in enumerate(rows):
        y = 310 + i * 64
        s.rect(0, (130, y, 760, y + 52), CARD, radius=10, outline=(225, 225, 220), dx=0)
        s.text(0, (154, y + 8), f"「{bad}」", size=28, color=INK2, dx=0)
        s.text(0, (470, y + 8), f"→ {ok}", size=28, bold=True, dx=0)
    s.text(0, (130, 590), "書き起こしをひらがなに直し、器械名・工程名の読みとの近さで置き換える", size=22, color=INK2)
    s.text(0, (900, 250), "Gemini に状況ごと渡し、決まった形の JSON で答えさせる", size=30, bold=True)
    s.image(0, (900, 310), _json(), dx=0)
    bullets(s, (130, 680), [
        "工程表 (皮膚切開 → 剥離 → 止血 → 縫合) と術者が持っている器械を渡し、「次」「あれ取って」を推定させる",
        "応答はキャッシュして動画を再現できるようにし、回数制限や通信エラーのときはキーワードのルールで解釈する",
        "器械名の聞き取りは補正前 9/18 → 補正後 18/18。解釈は Gemini 23/23。ただし別の声では補正後 11〜16/18",
    ], width=1560, size=26)
    return s


def _json() -> np.ndarray:
    lines = ['状況: 工程「止血」 / 術者が持っている器械 [鑷子, ペアン]', '発話: 「縫合するのであれ取って」', '',
             '{ "intent": "request",', '  "instrument": "needle_holder",', '  "step": "縫合",',
             '  "reason": "縫合の工程で持針器を求めているため" }']
    im = Image.new("RGB", (790, 330), (36, 38, 42))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        dr.text((22, 20 + i * 42), ln, font=_font(24), fill=(230, 220, 160) if ln.startswith(("{", " ")) else (210, 210, 210))
    return np.array(im)


def p_count() -> Slide:
    s = page("COUNT & LEDGER", "台帳と器械カウント: 記録と画像の二重で見張る")
    picture(s, (130, 250), img("count_examples.jpg"), 760, 427, "評価の例 (器械台・トレイの範囲で数え、台帳と照合)")
    bullets(s, (940, 250), [
        "台帳は状態機械。許される移り方は 渡す・返却・器械台に戻す だけ",
        "検出の 3D 位置で器械台・トレイに振り分ける。カメラに写る術者の手の器械は数えない",
        "合わなければトレイの器械を 1 本ずつ並べ直して数え直す",
        "重なりが残った・台帳と違う器械が見えた・同じ器械が 2 か所、のときは「一致」と言わない",
        "回収は撮り直して確かめてから台帳に付ける",
    ], width=800, size=25)
    for i, (n, l) in enumerate([("98.3%", "照合の正解 (ランダム 60 場面)"), ("0/51", "足りない器械の見逃し"),
                                ("100%", "初期カウント")]):
        stat(s, (130 + i * 380, 790), n, l, w=350)
    return s


def p_demo() -> Slide:
    s = page("DEMO", "デモの流れ (動画 2 分 41 秒)")
    shots = [("demo_handoff.jpg", "「メス」→ 持ち手を術者に向けて渡す"),
             ("demo_rearrange.jpg", "「閉創前カウント」→ 重なりを並べ直す"),
             ("demo_alert.jpg", "持針器が足りない → 警告して止まる"),
             ("demo_ok.jpg", "返してもらい数え直して「カウント一致」")]
    for i, (f, c) in enumerate(shots):
        x, y = 130 + (i % 2) * 800, 240 + (i // 2) * 400
        picture(s, (x, y), img(f), 760, 320, f"{i + 1}. {c}")
    return s


def p_points() -> Slide:
    s = page("HIGHLIGHTS", "工夫した点")
    items = [("創造性", "「渡す」だけでなく「数えて照合する」器械出し。体内遺残を防ぐ確認を、記録と画像の二重にした"),
             ("技術力", "合成データの自動ラベル・深度での把持点推定・手首の可動域を考えた把持計画・読みでの音声補正・状態機械"),
             ("実用性", "迷ったら止まる設計 (見逃し 0)。言われなくても返却に気づいて片付け、術者の手がある間は手を出さない"),
             ("表現力", "術者とロボットが同時に動く台本、話者付きの字幕、シミュレーター内の時刻でそろえた再現できる動画")]
    for i, (k, v) in enumerate(items):
        y = 260 + i * 170
        s.rect(0, (130, y, 1700, y + 140), CARD, radius=16, outline=(225, 225, 220), dx=0)
        s.rect(0, (130, y, 140, y + 140), OLIVE, dx=0)
        s.text(0, (170, y + 40), k, size=44, bold=True, color=OLIVE_DARK, dx=0)
        s.para(0, (400, y + 30), v, width=1260, size=30, dx=0)
    return s


def p_discussion() -> Slide:
    s = page("DISCUSSION", "考察・制限・今後")
    s.text(0, (130, 250), "分かったこと", size=34, bold=True)
    y = bullets(s, (130, 310), [
        "練習していない状況 (重なり) はデータに入れると改善したが、易しい問題の成績が少し落ちた (mAP50-95 0.855 → 0.833)",
        "重なりの崩れ方は少しの差で大きく変わる。30 通り試して、きれいに並べ直せたのは 6 通り。安全側に倒す仕組みが要る",
        "読み上げ音声どうしでも声で聞き取りが大きく違う (補正後 11〜18/18)。人の声での評価は今後",
    ], width=1560, size=26)
    s.text(0, (130, y + 20), "制限と今後", size=34, bold=True)
    bullets(s, (130, y + 80), [
        "実機なし (摩擦・形・照明の差は未測定) → 実機の Franka と実際の器械台で、シミュレーションとの差を測る",
        "真上のカメラ 1 台では横倒しや重なりを取り違える → 斜めや横から見る 2 台目のカメラ",
        "器械は 5 種 1 本ずつ・ガーゼと針は数えない → カウントの対象を広げる",
        "術者の動きはスクリプト → 手の動きから次に要る器械を先読みする",
    ], width=1560, size=26)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-url", default=VIDEO_URL, help="動画 (YouTube) の URL")
    args = ap.parse_args()
    pages = []
    if COVER.exists():
        pages.append(cv2.resize(cv2.imread(str(COVER))[..., ::-1], (1920, 1080)))
    for sl in (p_title(args.video_url), background_slide(), p_system(), p_scene(), p_perception(), p_grasp(),
               p_language(), p_count(), p_demo(), p_points(), results_slide(), p_discussion()):
        pages.append(sl.render(sl.seconds + 30))  # 動きの終わった状態 (静止画)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, pg in enumerate(pages):
        cv2.imwrite(str(OUT / f"page_{i + 1:02d}.jpg"), pg[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    ims = [Image.fromarray(p).convert("RGB") for p in pages]
    pdf = OUT / f"{ACCOUNT}.pdf"  # 提出するファイル名 (ユーザーの指定)
    ims[0].save(pdf, save_all=True, append_images=ims[1:], resolution=150)
    print(f"saved {pdf} ({len(pages)} ページ, {pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
