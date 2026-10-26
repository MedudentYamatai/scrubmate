"""器械 5 種をプリミティブ (box) の組み合わせで作る。

方針:
- 全部同じ金属色。色ではなく「上から見たシルエット」で区別させる (検出器の学習を意味のあるものにするため)。
- 器械のローカル座標: x = 長手方向 (先端が +x)、y = 幅、z = 厚み。原点 = 重心。
  重心を原点にしておくと「把持点 = body 原点」にでき、持ち上げたときに回転モーメントが小さい。
- 断面は実物より太め (厚さ 1cm 前後) にして、指で挟みやすく・接触計算を安定させる。

シルエットの特徴 (検出時の手がかり):
  scalpel       … リング無し。まっすぐな柄 + 先端に幅広の刃
  forceps       … リング無し。細い 2 本の腕が後端でつながった V 字
  scissors      … リング有り。幅広で長い刃 (先端に向かって細くなる)
  needle_holder … リング有り。長い柄 + 短くて太い顎
  pean          … リング有り。短めの柄 + 細長い顎
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

INSTRUMENT_NAMES = ["scalpel", "needle_holder", "forceps", "scissors", "pean"]
JA_NAMES = {
    "scalpel": "メス",
    "needle_holder": "持針器",
    "forceps": "鑷子",
    "scissors": "ハサミ",  # 正式名は剪刀だが、手術室ではハサミと呼ぶことが多い
    "pean": "ペアン",
}

THICK = 0.010  # 標準の厚さ [m]
DENSITY = 4000.0  # [kg/m^3]。ステンレス (7900) だと 100g 超になるので軽めにしている
METAL_RGBA = (0.72, 0.74, 0.77, 1.0)


@dataclass
class Box:
    """xy 平面上の線分 p0→p1 を、幅 width・厚さ thick の直方体にしたもの。"""

    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float
    thick: float = THICK

    @property
    def center(self) -> np.ndarray:
        return (np.array(self.p0) + np.array(self.p1)) / 2

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.array(self.p1) - np.array(self.p0)))

    @property
    def yaw(self) -> float:
        d = np.array(self.p1) - np.array(self.p0)
        return float(np.arctan2(d[1], d[0]))

    @property
    def volume(self) -> float:
        return self.length * self.width * self.thick


@dataclass
class Shape:
    boxes: list[Box] = field(default_factory=list)

    def seg(self, p0, p1, width, thick=THICK):
        self.boxes.append(Box(p0, p1, width, thick))

    def ring(self, center, radius, bar=0.004, n=8):
        """指を通すリング (中空)。n 角形の辺を box で並べて近似する。"""
        cx, cy = center
        angles = np.linspace(0, 2 * np.pi, n + 1)
        pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        # 辺の長さを bar だけ伸ばして、角の隙間を埋める
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            d = np.array([x1 - x0, y1 - y0])
            ext = d / np.linalg.norm(d) * bar / 2
            self.seg((x0 - ext[0], y0 - ext[1]), (x1 + ext[0], y1 + ext[1]), bar)

    def mirrored_seg(self, p0, p1, width, thick=THICK):
        """y = 0 に対して対称な 2 本 (リング器械の左右の柄など)。"""
        self.seg(p0, p1, width, thick)
        self.seg((p0[0], -p0[1]), (p1[0], -p1[1]), width, thick)

    def com(self) -> np.ndarray:
        """体積加重の重心 (密度一様なので質量重心と同じ)。"""
        vols = np.array([b.volume for b in self.boxes])
        centers = np.array([b.center for b in self.boxes])
        return (vols[:, None] * centers).sum(0) / vols.sum()


def _ring_handles(s: Shape, ring_x: float, ring_dy: float, radius: float = 0.010) -> None:
    """リング 2 個 + その間をつなぐラチェット (止め金具)。"""
    s.ring((ring_x, ring_dy), radius)
    s.ring((ring_x, -ring_dy), radius)
    s.seg((ring_x + radius + 0.004, 0.006), (ring_x + radius + 0.004, -0.006), 0.004)


def make_shape(kind: str) -> Shape:
    s = Shape()
    if kind == "scalpel":
        s.seg((-0.085, 0.0), (0.035, 0.0), 0.010)  # 柄
        s.seg((0.035, 0.002), (0.066, 0.002), 0.017, thick=0.004)  # 刃 (幅広・薄い)
        s.seg((0.064, -0.001), (0.080, -0.004), 0.008, thick=0.004)  # 刃先
    elif kind == "forceps":
        s.seg((-0.078, 0.0), (-0.060, 0.0), 0.011)  # 後端の接合部
        s.seg((-0.064, 0.003), (0.064, 0.007), 0.006)  # 腕 (左)
        s.seg((-0.064, -0.003), (0.064, -0.007), 0.006)  # 腕 (右)
        s.seg((0.062, 0.007), (0.076, 0.0025), 0.004)  # 先端 (左)
        s.seg((0.062, -0.007), (0.076, -0.0025), 0.004)  # 先端 (右)
    elif kind == "scissors":
        _ring_handles(s, ring_x=-0.076, ring_dy=0.014, radius=0.011)
        s.mirrored_seg((-0.065, 0.010), (-0.006, 0.003), 0.006)  # 柄
        s.seg((-0.008, 0.0), (0.006, 0.0), 0.012)  # 支点 (ネジ部)
        s.seg((0.004, 0.0), (0.058, 0.0), 0.016)  # 刃 (幅広)
        s.seg((0.056, 0.0), (0.078, 0.0), 0.010)  # 刃の先 (細くなる)
        s.seg((0.076, 0.0), (0.088, 0.0), 0.004)  # 刃先
    elif kind == "needle_holder":
        _ring_handles(s, ring_x=-0.086, ring_dy=0.014)
        s.mirrored_seg((-0.076, 0.009), (0.020, 0.003), 0.006)  # 長い柄
        s.seg((0.018, 0.0), (0.034, 0.0), 0.013)  # ボックスジョイント
        s.seg((0.032, 0.0), (0.054, 0.0), 0.011)  # 顎 (短く太い・先端が丸い)
    elif kind == "pean":
        _ring_handles(s, ring_x=-0.070, ring_dy=0.013)
        s.mirrored_seg((-0.060, 0.008), (-0.004, 0.003), 0.005)  # 柄 (短め)
        s.seg((-0.006, 0.0), (0.008, 0.0), 0.010)  # ボックスジョイント
        s.seg((0.006, 0.0), (0.050, 0.0), 0.006)  # 顎 (細長い)
        s.seg((0.048, 0.0), (0.068, 0.0), 0.004)  # 顎の先
    else:
        raise ValueError(f"unknown instrument: {kind}")
    return s


def add_instrument(spec: mujoco.MjSpec, kind: str, pos, yaw: float) -> mujoco.MjsBody:
    """worldbody に器械を free body として追加する。pos は重心 (= 把持点) の world 座標。"""
    shape = make_shape(kind)
    com = shape.com()

    body = spec.worldbody.add_body(name=kind, pos=list(pos), quat=_yaw_quat(yaw))
    # freejoint: 6 自由度 (並進 3 + 回転 3) で自由に動ける。qpos は位置 3 + クォータニオン 4 = 7 次元
    body.add_freejoint(name=f"{kind}_free")
    for i, b in enumerate(shape.boxes):
        c = b.center - com  # 重心が原点になるようにずらす
        body.add_geom(
            name=f"{kind}_g{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[b.length / 2, b.width / 2, b.thick / 2],
            pos=[c[0], c[1], 0.0],
            quat=_yaw_quat(b.yaw),
            density=DENSITY,
            rgba=list(METAL_RGBA),
            # condim=4: 通常の摩擦 (接線 2 方向) に加えて「ねじれ摩擦」を入れる。
            # 指で挟んだ器械が、挟んだ点を軸にくるっと回ってしまうのを防ぐ。
            condim=4,
            friction=[1.0, 0.01, 0.0001],
            # solref の第 1 要素 = 接触の時定数。小さいほど硬い接触 (めり込みが少ない)
            solref=[0.005, 1.0],
        )
    # 把持点 (= 重心)。ログや可視化用。制御には使わない (Phase 3 以降は知覚から求める)
    body.add_site(name=f"{kind}_grasp", pos=[0, 0, 0], size=[0.003, 0, 0], rgba=[1, 0, 0, 0.0])
    return body


def _yaw_quat(yaw: float) -> list[float]:
    """z 軸まわりの回転 yaw [rad] を (w, x, y, z) クォータニオンにする。"""
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
