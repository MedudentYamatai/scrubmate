"""Phase 5: 検出カメラの画像で器械を数え、器械台 / 返却トレイ のどちらにあるかで振り分ける。

流れ
  1. (呼ぶ側が腕を待避させてから) RGB と深度を撮り、YOLO-seg で器械を検出する (perception/locate.py の Locator を流用)
  2. マスクの画素を深度で 3D に戻し、その中央値 = 器械の 3D 位置 (world 座標) とする
  3. 3D 位置が「器械台の範囲」「トレイの範囲」のどちらの箱に入るかで振り分ける。どちらでもなければ数えない
     検出カメラには術者の手 (と、術者が持っている器械) も写る。画像の上の位置ではなく 3D の箱で判定するのは、
     手に持った器械 (高さ約 0.96m) を「器械台にある」と数えないため。手はトレイの真上付近にあるので、高さで区別する

箱の範囲は、シーンの寸法 (sim/assets/or_scene.xml) から決めた。評価データを見て決めたものではない。
"""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from perception.locate import Locator
from sim import scene
from sim.instruments import THICK

# 器械の 3D 位置 (マスクの中央値) が入るべき箱 [m]。器械は支持バーの上 (z≈0.755) かトレイの底 (z≈0.72) にある。
# 高さの上限 0.85m: トレイで器械が 3 本重なっても 0.79m 程度。術者の手 (0.98m) に持った器械は 0.9m 以上
Z_RANGE = (0.70, 0.85)
REGIONS = {
    # 器械台の天板のうち、トレイより手前 (y < 0.27)。天板は x 0.32〜0.64、y −0.26〜0.50
    "table": ((0.32, 0.64), (-0.26, 0.27)),
    # 返却トレイの壁の内側 (中心 (0.48, 0.38)、内寸 ±0.115 × ±0.105)
    "tray": ((scene.TRAY_CENTER[0] - 0.115, scene.TRAY_CENTER[0] + 0.115),
             (scene.TRAY_CENTER[1] - 0.105, scene.TRAY_CENTER[1] + 0.105)),
}
# 器械の上面の高さ (支持バーに載っているとき) と、「浮いている」とみなす高さ。
# 浮いている = 下に別の器械が挟まっている (重なり)。上の器械に隠れて、下の器械は検出されないことがある。
# 重なっていない器械のマスクの中央値は、上面から −3〜0mm (開発用の場面 30 個 × 2 の 269 検出で実測)。重なると約 +10mm
TOP_Z = scene.INSTRUMENT_Z + THICK / 2
ELEVATED_DZ = 0.006


@dataclass
class Detection:
    kind: str
    conf: float
    pos: np.ndarray  # マスクの 3D 点の中央値 (world 座標) [m]
    region: str  # table / tray / outside
    polygon: np.ndarray  # 画像上の輪郭

    @property
    def elevated(self) -> bool:
        return self.region != "outside" and self.pos[2] > TOP_Z + ELEVATED_DZ


@dataclass
class CountResult:
    detections: list[Detection]
    rgb: np.ndarray
    depth: np.ndarray

    def counted(self) -> list[tuple[str, str]]:
        """数えた器械: (器械, 場所 table / tray) の一覧。同じ器械が 2 回出てきたらそのまま 2 回入る (取り違えの印)。"""
        return [(d.kind, d.region) for d in self.detections if d.region != "outside"]

    def where(self) -> dict[str, str]:
        """器械 → 見えた場所 (器械台かトレイのものだけ)。同じ器械が 2 か所で検出されたら確信度の高い方。"""
        out: dict[str, tuple[str, float]] = {}
        for d in self.detections:
            if d.region != "outside" and (d.kind not in out or out[d.kind][1] < d.conf):
                out[d.kind] = (d.region, d.conf)
        return {k: v[0] for k, v in out.items()}

    def at(self, region: str) -> list[str]:
        return sorted(d.kind for d in self.detections if d.region == region)


MASK_IOU_SAME = 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    return inter / max(np.count_nonzero(a | b), 1)


def region_of(p: np.ndarray) -> str:
    if not Z_RANGE[0] <= p[2] <= Z_RANGE[1]:
        return "outside"
    for name, ((x0, x1), (y0, y1)) in REGIONS.items():
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return name
    return "outside"


class InstrumentCounter:
    def __init__(self, locator: Locator):
        self.loc = locator

    def count(self, data: mujoco.MjData) -> CountResult:
        """撮影して数える。腕が器械を隠すと数え間違えるので、腕は呼ぶ側で待避させておくこと (Phase 2: 待避時 100%)。"""
        rgb, depth = self.loc.capture(data)
        dets, kept_masks = [], []
        for kind, conf, poly in sorted(self.loc.detect(rgb), key=lambda r: -r[1]):
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            v, u = np.nonzero(mask)
            if len(u) == 0:
                continue
            # 同じ 1 本に 2 つの種類の名前が付くことがある (開発用の場面 種 18: ペアン 0.93 と 持針器 0.47 が同じ輪郭)。
            # YOLO は種類ごとにしか重複を消さないので、輪郭が半分以上重なる検出は確信度の高い方だけ残す。
            # 重なって置かれた 2 本は、下の器械の見えている部分が上の器械と重ならないので、ここでは消えない
            if any(_iou(mask, m) > MASK_IOU_SAME for m in kept_masks):
                continue
            kept_masks.append(mask)
            origin, d = self.loc.rays(data, u, v)
            pts = origin + d * depth[v, u][:, None]
            # 中央値: マスクの縁から下の天板が見えている画素 (外れ値) に引っ張られにくい
            p = np.median(pts, axis=0)
            dets.append(Detection(kind=kind, conf=float(conf), pos=p, region=region_of(p), polygon=poly))
        return CountResult(detections=dets, rgb=rgb, depth=depth)
