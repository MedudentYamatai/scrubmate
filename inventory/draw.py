"""カウント結果の可視化: 検出カメラの画像に、数える範囲 (器械台・トレイの箱) と検出結果、台帳、警告を重ねる。"""

import cv2
import mujoco
import numpy as np

from demo.jptext import put_lines
from inventory.count import REGIONS, CountResult
from inventory.ledger import PLACE_JA, Ledger
from perception.locate import HEIGHT, WIDTH, Locator
from sim.instruments import JA_NAMES

REGION_COLORS = {"table": (80, 220, 120), "tray": (80, 200, 255), "outside": (150, 150, 150)}  # RGB
REGION_JA = {"table": "器械台", "tray": "トレイ", "outside": "範囲外 (数えない)"}


def _to_px(loc: Locator, data: mujoco.MjData, p: np.ndarray) -> tuple[int, int]:
    """world 座標 → 画像の画素 (Locator.rays の逆変換)。"""
    R = data.cam_xmat[loc.cam].reshape(3, 3)
    pc = R.T @ (p - data.cam_xpos[loc.cam])
    f = HEIGHT / (2 * np.tan(np.radians(loc.model.cam_fovy[loc.cam]) / 2))
    return int(round(pc[0] / -pc[2] * f + WIDTH / 2 - 0.5)), int(round(-pc[1] / -pc[2] * f + HEIGHT / 2 - 0.5))


def draw_count(count: CountResult, loc: Locator, data: mujoco.MjData, ledger: Ledger | None = None,
               message: str | None = None, alert: bool = False) -> np.ndarray:
    """BGR 画像 (cv2 で保存する用) を返す。"""
    img = count.rgb.copy()
    for name, ((x0, x1), (y0, y1)) in REGIONS.items():
        corners = [_to_px(loc, data, np.array([x, y, 0.76])) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        cv2.polylines(img, [np.array(corners, np.int32)], True, REGION_COLORS[name], 2)
    labels = []
    for d in count.detections:
        col = REGION_COLORS[d.region]
        cv2.polylines(img, [d.polygon.astype(np.int32)], True, col, 2)
        u, v = d.polygon.mean(0).astype(int)
        labels.append((f"{JA_NAMES[d.kind]} {d.conf:.2f}", col, (int(u) - 30, int(v) - 34)))
    for text, col, org in labels:
        img = put_lines(img, [(text, col)], org, size=20, bg=(0, 0, 0, 140), pad=3)

    lines = [(f"器械台 {len(count.at('table'))} 本 / トレイ {len(count.at('tray'))} 本", (255, 255, 255))]
    if ledger is not None and ledger.established:
        lines.append(("台帳", (200, 200, 200)))
        lines += [(f"  {JA_NAMES[k]}: {PLACE_JA[s]}", (230, 230, 230)) for k, s in ledger.state.items()]
    img = put_lines(img, lines, (14, 12), size=20)
    if message:
        img = put_lines(img, [(message, (255, 90, 90) if alert else (120, 255, 140))], (14, HEIGHT - 44), size=24)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
