"""日本語の文字を画像に描く。OpenCV の putText は日本語を描けないので、PIL と Mac の日本語フォント (ヒラギノ) を使う。

フォントは .env (または環境変数) の JP_FONT で差し替えられる (例: JP_FONT=/Library/Fonts/NotoSansJP-Regular.otf)。
スライドの太字は JP_FONT_BOLD (demo/slides.py)。
"""

import os
import re
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()  # .env の JP_FONT を読む

DEFAULT_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"


@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.environ.get("JP_FONT", DEFAULT_FONT), size)


def put_lines(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], org: tuple[int, int], size: int = 22,
              line_gap: float = 1.35, bg: tuple[int, int, int, int] | None = (0, 0, 0, 160), pad: int = 8) -> np.ndarray:
    """img (RGB, uint8) の org (左上) から、(文字列, 色 RGB) の行を並べて描く。bg を渡すと半透明の下地を敷く。"""
    pil = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    f = font(size)
    x, y = org
    step = int(size * line_gap)
    if bg is not None and lines:
        w = max(draw.textlength(t, font=f) for t, _ in lines)
        draw.rectangle([x - pad, y - pad, x + w + pad, y + step * len(lines) - (step - size) + pad], fill=bg)
    for i, (text, col) in enumerate(lines):
        draw.text((x, y + i * step), text, font=f, fill=(*col, 255))
    return np.array(Image.alpha_composite(pil, layer).convert("RGB"))


def wrap(text: str, size: int, width: int) -> list[str]:
    """幅 width [px] に収まるように折り返す。英数字のかたまりは途中で切らず、句読点・閉じかっこは行頭に置かない。"""
    f = font(size)
    lines, cur = [], ""
    for tok in re.findall(r"[0-9A-Za-z.,~〜%+\-/_]+|.", text):
        if f.getlength(cur + tok) > width and cur and tok not in "、。」）)":
            carry = "「" if cur.endswith("「") else ""
            lines.append(cur[:len(cur) - len(carry)])
            cur = carry
        cur += tok
    lines.append(cur)
    return lines
