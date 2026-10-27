"""Phase 6: 動画のイントロ (背景・コンセプト・仕組み) と、結果・展望のスライド。動きを付けて 1 コマずつ描く。

見た目は講座から配られた OP/ED に合わせる: 白地、黒の文字、オリーブ色の斜めの帯。
各要素は「何秒目に、どう入ってくるか」(appear) を持ち、frames() が 30fps のコマを順に返す。
ロボットが動かない画面なので等倍で流す (ユーザーと相談して決定)。

背景の数字の出典 (動画の画面の下にも出す):
  Cima RR et al. J Am Coll Surg 2008;207:80-7  … 体内遺残は 5,500 件に 1 件。遺残が見つかった患者の 62% でカウントは「合っている」とされていた
  Gawande AA et al. N Engl J Med 2003;348:229-35 … 入院手術 8,800〜18,760 件に 1 件
  (いずれも Grant-Orser A et al. CMAJ 2012;184:1275-8 での引用で数字を確かめた)
"""

import json
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG = (250, 250, 248)
INK = (24, 24, 24)
INK2 = (90, 90, 90)  # 補足の文字
INK3 = (140, 140, 140)  # 出典など
OLIVE = (179, 176, 145)  # 講座の OP/ED の帯の色
OLIVE_DARK = (120, 117, 88)  # 差し色 (見出しの線・番号)
CARD = (255, 255, 255)
FONT_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_W6 if bold else FONT_W3, size)


def ease(x: float) -> float:
    """0→1 を、最初速く最後ゆっくり (ease-out cubic)。"""
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


@dataclass
class El:
    kind: str  # text / image / poly / rect / line
    t0: float  # 出てくる時刻 [s]
    data: dict = field(default_factory=dict)
    dur: float = 0.6  # 出てくるのにかける時間
    dx: float = 60  # 横から滑り込む量 [px]
    dy: float = 0


class Slide:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.els: list[El] = []

    def text(self, t0, xy, s, size=40, bold=False, color=INK, **kw) -> "Slide":
        self.els.append(El("text", t0, dict(xy=xy, s=s, size=size, bold=bold, color=color), **kw))
        return self

    def para(self, t0, xy, s, width: int, size=40, bold=False, color=INK, gap=1.45, stagger=0.0, **kw) -> int:
        """幅 width [px] で折り返して行ごとに置く。置いた行数を返す。"""
        f = _font(size, bold)
        # 英数字のかたまり (「8,800〜18,760」「YOLO-seg」など) は途中で切らない。閉じかっこ・句読点は行頭に来ないよう前に付ける
        tokens = re.findall(r"[0-9A-Za-z.,~〜%+\-/]+|.", s)
        lines, cur = [], ""
        for tok in tokens:
            if f.getlength(cur + tok) > width and cur and tok not in "、。」）)":
                carry = "「" if cur.endswith("「") else ""  # 開きかっこは行末に残さず次の行へ
                lines.append(cur[:len(cur) - len(carry)])
                cur = carry
            cur += tok
        lines.append(cur)
        for i, ln in enumerate(lines):
            self.text(t0 + i * stagger, (xy[0], xy[1] + int(i * size * gap)), ln, size=size, bold=bold, color=color, **kw)
        return len(lines)

    def image(self, t0, xy, img: np.ndarray, **kw) -> "Slide":
        self.els.append(El("image", t0, dict(xy=xy, img=Image.fromarray(img).convert("RGBA")), **kw))
        return self

    def poly(self, t0, pts, color, **kw) -> "Slide":
        self.els.append(El("poly", t0, dict(pts=pts, color=color), **kw))
        return self

    def rect(self, t0, box, color, radius=0, outline=None, **kw) -> "Slide":
        self.els.append(El("rect", t0, dict(box=box, color=color, radius=radius, outline=outline), **kw))
        return self

    def counter(self, t0, xy, target: float, fmt: str, size=96, dur=1.2, **kw) -> "Slide":
        """数字を 0 から target まで数え上げる (fmt は "{:.1f}" など)。"""
        self.els.append(El("count", t0, dict(xy=xy, target=target, fmt=fmt, size=size), dur=dur, dx=0, **kw))
        return self

    def render(self, t: float) -> np.ndarray:
        base = Image.new("RGBA", (W, H), (*BG, 255))
        for e in self.els:
            if t < e.t0:
                continue
            p = ease((t - e.t0) / e.dur)
            a = int(255 * p)
            ox, oy = (1 - p) * e.dx, (1 - p) * e.dy
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dr = ImageDraw.Draw(layer)
            d = e.data
            if e.kind == "text":
                x, y = d["xy"]
                dr.text((x + ox, y + oy), d["s"], font=_font(d["size"], d["bold"]), fill=(*d["color"], a))
            elif e.kind == "count":
                x, y = d["xy"]
                v = d["target"] * p
                dr.text((x, y), d["fmt"].format(v), font=_font(d["size"], True), fill=(*INK, 255))
            elif e.kind == "image":
                im = d["img"].copy()
                im.putalpha(Image.eval(im.getchannel("A"), lambda v: v * a // 255))
                layer.alpha_composite(im, (int(d["xy"][0] + ox), int(d["xy"][1] + oy)))
            elif e.kind == "poly":
                dr.polygon([(x + ox, y + oy) for x, y in d["pts"]], fill=(*d["color"], a))
            elif e.kind == "rect":
                x0, y0, x1, y1 = d["box"]
                dr.rounded_rectangle((x0 + ox, y0 + oy, x1 + ox, y1 + oy), radius=d["radius"], fill=(*d["color"], a),
                                     outline=(*d["outline"], a) if d["outline"] else None, width=2)
            elif e.kind == "line":
                pass
            base.alpha_composite(layer)
        return np.array(base.convert("RGB"))

    def frames(self):
        for i in range(int(round(self.seconds * FPS))):
            yield self.render(i / FPS)


# ---- 共通の飾り ----
def _bands(s: Slide, t0: float = 0.0, big: bool = False) -> None:
    """OP と同じオリーブの斜めの帯 (右から滑り込む)。big = 表紙のように大きく (文字は左半分に置く)。
    文字の多い画面では、帯は右端に細く出す (文字にかからないように x ≥ 1760)。"""
    if big:
        sh = 420  # 表紙より右に寄せて、左の文字にかからないようにする
        s.poly(t0, [(1180 + sh, 0), (1920, 0), (1920, 1080), (720 + sh, 1080)], OLIVE, dx=500, dur=0.9)
        s.poly(t0 + 0.15, [(1740 + sh / 3, 0), (1920, 0), (1920, 1080), (1290 + sh / 3, 1080)], (*BG,), dx=500, dur=0.9)
        s.poly(t0 + 0.25, [(1800 + sh / 3, 0), (1920, 0), (1920, 1080), (1350 + sh / 3, 1080)], OLIVE, dx=500, dur=0.9)
    else:
        s.poly(t0, [(1860, 0), (1920, 0), (1920, 1080), (1770, 1080)], OLIVE, dx=200, dur=0.8)


def _heading(s: Slide, t0: float, en: str, ja: str) -> None:
    s.rect(t0, (120, 96, 128, 196), OLIVE_DARK, dx=0, dy=20)
    s.text(t0, (150, 92), en, size=26, color=OLIVE_DARK, bold=True, dx=40)
    s.text(t0 + 0.1, (150, 126), ja, size=58, bold=True, dx=40)


def _source(s: Slide, t0: float, text: str) -> None:
    s.text(t0, (120, 1010), text, size=20, color=INK3, dx=0, dur=0.8)


# ---- 各スライド ----
def background_slide() -> Slide:
    s = Slide(10.0)
    _bands(s)
    _heading(s, 0.2, "BACKGROUND", "器械出しは「渡す」と「数える」を同時にこなす仕事")
    s.text(0.9, (150, 258), "器械出し看護師は、術者が求める器械を素早く渡しながら、", size=36, color=INK2)
    s.text(1.1, (150, 312), "すべての器械の所在を管理し、置き忘れを防ぐ。", size=36, color=INK2)
    s.text(1.6, (150, 380), "しかし手術室は忙しく、人の注意には限りがある。数え間違いや記録の誤りは起こりうる。", size=36,
           bold=True)
    # 裏付けの数字 (数字は本文と同じ濃い文字色。差し色は上の細い帯だけ)
    for i, (num, unit, label, sub) in enumerate([
        ("1 / 5,500", "", "手術で器械やガーゼが体内に残る頻度", "米国の医療機関の調査"),
        ("62", "%", "置き忘れが見つかった患者のうち、カウントは「合っている」とされていた割合", "数えていても見落としは起きる"),
    ]):
        x0, t = 150 + i * 760, 2.6 + i * 0.8
        s.rect(t, (x0, 470, x0 + 720, 790), CARD, radius=18, outline=(225, 225, 220), dx=0, dy=30)
        s.rect(t, (x0, 470, x0 + 720, 480), OLIVE, dx=0, dy=30)
        s.text(t + 0.2, (x0 + 40, 500), num, size=104, bold=True, dx=0, dy=20)
        if unit:
            s.text(t + 0.2, (x0 + 40 + int(_font(104, True).getlength(num)) + 8, 546), unit, size=56, bold=True, dx=0,
                   dy=20)
        n = s.para(t + 0.4, (x0 + 40, 650), label, width=640, size=28, dx=0, dy=10)
        s.text(t + 0.5, (x0 + 40, 650 + n * 42 + 6), sub, size=22, color=INK2, dx=0, dy=10)
    s.text(5.0, (150, 850), "渡す作業を任せ、器械の所在を「記録」と「目」の二重で見張る相棒がいれば、", size=40, bold=True)
    s.text(5.4, (150, 912), "看護師は患者と術野に集中でき、置き忘れも防げる。", size=40, bold=True, color=OLIVE_DARK)
    _source(s, 1.0, "出典: Cima RR et al. J Am Coll Surg 2008;207:80-7 / Gawande AA et al. N Engl J Med 2003;348:229-35")
    return s


def concept_slide() -> Slide:
    s = Slide(6.0)
    _bands(s, big=True)
    s.text(0.3, (130, 250), "SCRUB NURSE ROBOT", size=34, bold=True, color=OLIVE_DARK)
    s.text(0.5, (122, 290), "ScrubMate", size=170, bold=True)
    s.text(0.9, (132, 505), "声で渡し、目で数える器械出しロボット", size=48, bold=True, color=INK2)
    for i, ln in enumerate(["頼まれた器械を、持ち手を向けて渡す",
                            "返された器械に自分で気づいて片付ける",
                            "閉創前に画像で数えて台帳と照合し、合わなければ止めて知らせる"]):
        s.rect(1.5 + i * 0.35, (134, 640 + i * 70, 146, 652 + i * 70), OLIVE_DARK, dx=0)
        s.text(1.5 + i * 0.35, (166, 626 + i * 70), ln, size=34, color=INK)
    s.text(2.8, (134, 880), "MuJoCo シミュレーション / Franka Panda", size=26, color=INK3)
    return s


def _waveform(wav: Path, w: int, h: int) -> np.ndarray:
    with wave.open(str(wav), "rb") as f:
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    img = np.full((h, w, 3), 245, np.uint8)
    bins = np.array_split(np.abs(a), w // 4)
    for i, b in enumerate(bins):
        v = float(b.max()) if len(b) else 0
        hh = max(2, int(v * h * 0.9))
        cv2.rectangle(img, (i * 4, h // 2 - hh // 2), (i * 4 + 2, h // 2 + hh // 2), OLIVE_DARK, -1)
    return img


def _json_card(w: int, h: int, lines: list[str]) -> np.ndarray:
    im = Image.new("RGB", (w, h), (36, 38, 42))
    dr = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        dr.text((14, 16 + i * 44), ln, font=_font(22), fill=(230, 220, 160) if ":" in ln else (210, 210, 210))
    return np.array(im)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """縦横比を保って w×h に収め、はみ出す分を切る。"""
    ih, iw = img.shape[:2]
    sc = max(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * sc + 0.5), int(ih * sc + 0.5)))
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def pipeline_slide(thumbs: dict[str, np.ndarray]) -> Slide:
    s = Slide(9.0)
    _bands(s)
    _heading(s, 0.1, "HOW IT WORKS", "聞く → 解釈する → 見つける → 渡す → 数える")
    steps = [
        ("1", "聞く", "Whisper で書き起こす", "器械名は読みの近さで補正", "voice"),
        ("2", "解釈する", "Gemini が意図を JSON で返す", "「次」「あれ」は工程表から推定", "json"),
        ("3", "見つける", "YOLO-seg と深度で器械を見つける", "把持点と向きを推定", "detect"),
        ("4", "渡す", "微分 IK で腕を動かす", "物理接触で掴み、持ち手を術者へ", "grasp"),
        ("5", "数える", "台帳と画像を照合する", "合わなければ警告して停止", "count"),
    ]
    cw, gap, x_start, y0 = 300, 30, 130, 260
    for i, (no, title, l1, l2, key) in enumerate(steps):
        x, t = x_start + i * (cw + gap), 0.8 + i * 0.9
        s.rect(t, (x, y0, x + cw, y0 + 620), CARD, radius=16, outline=(225, 225, 220), dx=0, dy=40)
        s.text(t + 0.1, (x + 22, y0 + 18), no, size=64, bold=True, color=OLIVE_DARK, dx=0, dy=30)
        s.text(t + 0.1, (x + 90, y0 + 34), title, size=44, bold=True, dx=0, dy=30)
        s.image(t + 0.2, (x + 14, y0 + 120), _fit(thumbs[key], cw - 28, 250), dx=0, dy=30)
        n = s.para(t + 0.3, (x + 20, y0 + 396), l1, width=cw - 40, size=24, dx=0, dy=20)
        s.para(t + 0.35, (x + 20, y0 + 396 + n * 35 + 10), l2, width=cw - 40, size=24, color=INK2, dx=0, dy=20)
        if i < len(steps) - 1:
            ax = x + cw + 3
            s.poly(t + 0.5, [(ax, y0 + 290), (ax + gap - 6, y0 + 300), (ax, y0 + 310)], OLIVE_DARK, dx=-10)
    s.text(6.0, (130, 930), "掴む位置はすべてカメラの推定から得る。シミュレーターの正解は評価にだけ使う。", size=28, color=INK2)
    return s


def results_slide() -> Slide:
    s = Slide(11.0)
    _bands(s)
    _heading(s, 0.1, "RESULTS", "評価結果")
    s.text(0.4, (420, 158), "すべてシミュレーター内で計測", size=28, color=INK3, dx=20)
    # (数字, 書式, 後ろに付ける文字, 見出し, 補足)。数字は数え上げる
    tiles = [
        (300, "{:.0f}", " / 300", "pick & place 成功", "カメラの推定で掴む・置き方 30 通り"),
        (2.2, "{:.1f}", " mm", "把持点の推定誤差 (平均)", "先端の向きの正解は 100%"),
        (0.83, "{:.2f}", "", "検出 mask mAP50-95", "YOLO11n-seg・合成データで学習"),
        (18, "{:.0f}", " / 18", "器械名の聞き取り", "読みの補正前は 9 / 18"),
        (98.3, "{:.1f}", " %", "照合カウントの正解", "ランダムな 60 場面"),
        (0, "{:.0f}", " / 51", "足りない器械の見逃し", "器械が足りない 51 場面"),
    ]
    tw, th, gx, gy, x0, y0 = 500, 300, 36, 36, 130, 250
    for i, (val, fmt, suffix, label, sub) in enumerate(tiles):
        r, c = divmod(i, 3)
        x, y, t = x0 + c * (tw + gx), y0 + r * (th + gy), 0.7 + i * 0.35
        s.rect(t, (x, y, x + tw, y + th), CARD, radius=18, outline=(225, 225, 220), dx=0, dy=30)
        s.rect(t, (x, y, x + 10, y + th), OLIVE, dx=0, dy=30)
        s.counter(t + 0.2, (x + 40, y + 34), val, fmt, size=92)
        # 数え上げる数字の後ろの文字は、最終的な数字の幅の後ろに置く
        num_w = _font(92, True).getlength(fmt.format(val))
        if suffix:
            s.text(t + 0.2, (x + 40 + num_w + 6, y + 78), suffix, size=46, bold=True, color=INK2, dx=0)
        s.text(t + 0.4, (x + 40, y + 170), label, size=32, bold=True, dx=0, dy=10)
        s.text(t + 0.5, (x + 40, y + 222), sub, size=22, color=INK2, dx=0, dy=10)
    s.text(4.0, (130, 930), "課題: 絡まった器械の山はほどけず、真上のカメラ 1 台では横倒しの器械を取り違えることがある。", size=26,
           color=INK2, dx=0)
    return s


def outlook_slide() -> Slide:
    s = Slide(6.0)
    _bands(s, big=True)
    s.text(0.2, (130, 250), "NEXT", size=34, bold=True, color=OLIVE_DARK)
    s.text(0.4, (126, 300), "今後の展望", size=84, bold=True)
    for i, ln in enumerate(["斜めや横から見る 2 台目のカメラで、重なった器械を見分ける",
                            "器械に加えて、ガーゼや針のカウントへ広げる",
                            "実機の Franka と実際の器械台で、シミュレーションとの差を測る",
                            "術者の手の動きから、次に要る器械を先読みする"]):
        s.text(1.0 + i * 0.5, (140, 470 + i * 84), f"・{ln}", size=38, color=INK2)
    return s


def make_thumbs(pip_dir: Path, main_video: Path, log_path: Path, wav: Path, speed: float) -> dict[str, np.ndarray]:
    """仕組みのスライドに載せる縮小画像を、実際のデモの記録から作る。"""
    log = json.loads(log_path.read_text())
    pips = sorted(pip_dir.glob("*.jpg"))

    def pip_at(t: float) -> np.ndarray:
        best = min(pips, key=lambda p: abs(float(p.stem.split("_")[1][:-1]) - t))
        return cv2.imread(str(best))[..., ::-1]

    t_hand = next(x["t"] for x in log if "note" in x and "を渡す" in x["note"])
    t_say = next(x["t"] for x in log if x.get("robot", "").endswith("です"))
    rd = imageio.get_reader(main_video)
    grasp = rd.get_data(int(t_say / speed * FPS))[60:760, 200:1240]
    heard = next(x for x in log if x.get("uid") == "d3_sesshi")
    it = heard["intent"]
    return {
        "voice": _waveform(wav, 600, 300),
        "json": _json_card(272, 250, [f'「{heard["heard"]}」', "↓", f'intent: {it["intent"]}',
                                      f'instrument: {it["instrument"]}', f'step: {it["step"]}']),
        "detect": pip_at(t_hand),
        "grasp": grasp,
        "count": cv2.imread(str(pips[-1]))[..., ::-1],
    }
