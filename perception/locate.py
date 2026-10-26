"""Phase 3: 検出カメラの RGB + 深度画像から、器械ごとの「把持点 (3D)」と「向き (先端方向込みの yaw)」を推定する。

流れ
  1. RGB を YOLO-seg に通して、器械ごとのマスク (輪郭の内側の画素) を得る
  2. 深度画像で、マスクの画素が「カメラから何 m 先か」を読み、器械の上面の高さを決める
  3. マスクの各画素を、カメラの内部パラメータ (焦点距離 f・画像中心) で 3D の光線に直し、
     上面の高さの平面と交わる点 = 器械の上面の点 (world 座標, m 単位) にする
  4. その点群から
       把持点 = 重心 (+ 器械モデルから分かる補正)
       向き   = 点の散らばりが最大の方向 (主成分分析 PCA の第 1 主成分)
       先端側 = 長手方向に沿った「幅のプロファイル」を器械の形状モデルと照合して決める

ここで使う「器械の形状モデル」(sim/instruments.make_shape) は、実機でいえば器械のカタログ寸法に当たる事前知識。
シミュレーターの真値 (今どこにあるか) は一切使わない。真値は評価 (perception/eval_locate.py) でのみ使う。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from ultralytics import YOLO

from sim.instruments import INSTRUMENT_NAMES, THICK, make_shape
from sim.scene import BAR_TOP_Z

CAMERA = "detect_cam"
WIDTH, HEIGHT = 1024, 768  # 学習データと同じ解像度・縦横比
DEFAULT_WEIGHTS = Path("perception/weights/instruments_seg.pt")
CONF = 0.25
BIN = 0.002  # 幅のプロファイルを取る刻み [m]
GRID = 0.001  # 真上から見た地図の方眼の間隔 [m]。カメラの 1 画素 (器械台の高さで約 0.9mm) と同程度
GROW_PX = 5  # 深度でマスクを広げてよい幅 [px] (約 4.5mm)
PROFILE_HALF = 0.12  # プロファイルを取る範囲 (重心から ±12cm。一番長い器械でも収まる)


@dataclass
class Estimate:
    kind: str
    conf: float  # YOLO の確信度
    pos: np.ndarray  # 把持点の world 座標 [m]
    yaw: float  # 先端 (器械ローカルの +x) が向いている方向 [rad]
    tip_margin: float  # 先端判定の確からしさ。0 = 前後どちらとも言えない / 1 = はっきりしている
    elongation: float  # 細長さ (第 2 主成分 / 第 1 主成分 の標準偏差比)。1 に近いと長手方向が決まらない
    n_px: int
    polygon: np.ndarray  # 画像上の輪郭 (表示用)


def _profile(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """長手方向の座標 s ごとに、幅方向 t の広がり (max − min) を並べたもの。
    面積ではなく「外形の幅」を使うのは、鑷子のように 2 本の腕が開く V 字形だと、
    面積は前後でほぼ同じでも、外形の幅は先端側で広がっていて区別できるため。"""
    edges = np.arange(-PROFILE_HALF, PROFILE_HALF + 1e-9, BIN)
    idx = np.digitize(s, edges) - 1
    out = np.zeros(len(edges) - 1)
    ok = (idx >= 0) & (idx < len(out))
    for i in np.unique(idx[ok]):
        ti = t[idx == i]
        out[i] = ti.max() - ti.min()
    return out


def _fill_silhouette(img: np.ndarray) -> np.ndarray:
    """方眼に塗った器械の形を「外側の輪郭で囲まれた範囲」に揃える。
    YOLO のマスクは外形の多角形 (学習ラベルをそう作った) なので、リングの穴や鑷子の 2 本の腕の隙間も塗りつぶされている。
    比べる相手 (形状モデル) も同じ処理をしないと、重心が穴の分だけずれる (試作では剪刀で 6mm・鑷子で 7mm ずれた)。
    先に 3x3 の closing (膨張→収縮) で 2mm 以下の細い隙間を埋める: カメラの解像度 (約 0.9mm/px) では
    鑷子の先端の 1mm の隙間は写らず、閉じた輪郭として検出されるため。"""
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(img)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _grid_points(xy: np.ndarray) -> np.ndarray:
    """xy 点群 [m] を GRID 間隔の方眼に塗り、_fill_silhouette を通して、塗られたマスの中心を返す。"""
    o = xy.min(0) - 3 * GRID
    ij = np.floor((xy - o) / GRID).astype(np.int64)
    img = np.zeros((ij[:, 1].max() + 4, ij[:, 0].max() + 4), np.uint8)
    img[ij[:, 1], ij[:, 0]] = 1
    y, x = np.nonzero(_fill_silhouette(img))
    return o + (np.stack([x, y], -1) + 0.5) * GRID


def _shape_points(kind: str) -> np.ndarray:
    """器械を真上から見たシルエットを、観測と同じ方眼・同じ処理で点にしたもの (器械ローカル座標)。"""
    pts = []
    for b in make_shape(kind).boxes:
        # 0.25mm 間隔で box の上面を埋める点を作る (方眼に塗る前の、隙間の無い点群)
        u = np.arange(-b.length / 2, b.length / 2 + 1e-9, 0.00025)
        v = np.arange(-b.width / 2, b.width / 2 + 1e-9, 0.00025)
        uu, vv = np.meshgrid(u, v)
        c, sn = np.cos(b.yaw), np.sin(b.yaw)
        pts.append(np.stack([b.center[0] + c * uu - sn * vv, b.center[1] + sn * uu + c * vv], -1).reshape(-1, 2))
    return _grid_points(np.concatenate(pts))


@dataclass
class _Template:
    profile: np.ndarray  # 先端を +s 側に置いたときの幅のプロファイル
    com_offset: float  # 「上から見た面積の重心」から「質量の重心 (= 把持点)」までの長手方向の距離 [m]


def _build_templates() -> dict[str, _Template]:
    out = {}
    for kind in INSTRUMENT_NAMES:
        p = _shape_points(kind)
        area_c = p.mean(0)
        # 上から見た面積の重心と、厚みも含めた体積の重心は少し違う (メスの刃は薄いので、面積で見ると刃側に寄る)。
        # 把持点は体積の重心 (持ち上げたときに傾かない点) なので、その差を形状モデルから求めておく
        com = make_shape(kind).com()
        q = p - area_c
        out[kind] = _Template(profile=_profile(q[:, 0], q[:, 1]), com_offset=float(com[0] - area_c[0]))
    return out


class Locator:
    """カメラ画像 → 器械ごとの Estimate。"""

    def __init__(self, model: mujoco.MjModel, weights: Path = DEFAULT_WEIGHTS, camera: str = CAMERA):
        self.model = model
        self.cam = model.camera(camera).id
        self.camera = camera
        self.detector = YOLO(str(weights))
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # 把持点の印などは写さない (実カメラには写らないもの)
        self.templates = _build_templates()

    def close(self) -> None:
        self.renderer.close()

    # ---- 撮影 ----
    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """RGB (H, W, 3) と深度 (H, W) [m] を撮る。深度は「カメラの光軸方向の距離」(画素ごとの奥行き)。"""
        r = self.renderer
        r.update_scene(data, camera=self.camera, scene_option=self.opt)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.camera, scene_option=self.opt)
        depth = r.render()
        r.disable_depth_rendering()
        return rgb, depth

    # ---- 画素 → 3D ----
    def rays(self, data: mujoco.MjData, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """画素 (u = 右向き, v = 下向き) を通る光線を world 座標で返す (始点 = カメラ位置, 方向 = 奥行き 1m あたりの移動量)。

        ピンホールカメラ: 焦点距離 f [px] = H / (2 tan(fovy/2))。
        画像中心から (du, dv) px ずれた画素は、奥行き Z の所で (du·Z/f, dv·Z/f) m だけ光軸から離れている。
        MuJoCo のカメラ座標は x = 右, y = 上, 視線 = −z なので、画像の下向き v とは y の符号が逆。"""
        f = HEIGHT / (2 * np.tan(np.radians(self.model.cam_fovy[self.cam]) / 2))
        du = (u + 0.5) - WIDTH / 2  # +0.5: 画素の左上ではなく中心を使う
        dv = (v + 0.5) - HEIGHT / 2
        d_cam = np.stack([du / f, -dv / f, -np.ones_like(du, dtype=float)], -1)
        R = data.cam_xmat[self.cam].reshape(3, 3)  # 列 = カメラの x, y, z 軸 (world 座標)
        return data.cam_xpos[self.cam].copy(), d_cam @ R.T

    # ---- 推定 ----
    def detect(self, rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        # ultralytics は numpy 画像を OpenCV と同じ BGR の並びとして受け取る (学習画像も cv2 で BGR 保存した)
        r = self.detector.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=CONF, verbose=False)[0]
        if r.masks is None:
            return []
        return [(INSTRUMENT_NAMES[c], conf, poly) for poly, c, conf in
                zip(r.masks.xy, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()) if len(poly) >= 3]

    def estimate_one(self, data: mujoco.MjData, depth: np.ndarray, kind: str, conf: float,
                     poly: np.ndarray) -> Estimate | None:
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        v, u = np.nonzero(mask)
        if len(u) < 30:
            return None
        origin, d = self.rays(data, u, v)
        # 各画素の 3D 点 = カメラ位置 + 方向 × 奥行き。その高さ (z) を見る
        z = origin[2] + d[:, 2] * depth[v, u]
        # マスクの縁や、鑷子の 2 本の腕の隙間からは下の天板・バーが見えている。
        # 上位 10% の高さ = 器械の上面とみなす (上面より高い物は器械の上に無いので、外れ値に強い)
        z_top = float(np.percentile(z, 90))
        # 深度でマスクを補う: YOLO のマスクの外側 GROW_PX 以内で、支持バーより高い所にある画素 = 器械の取りこぼし。
        # 明るいトレイの上だと、メスの刃が背景に溶けてマスクから欠け、先端を逆に判定したことがあった (評価 layout 18)。
        # 範囲を狭くしているのは、トレイの壁 (上面 0.76m = 器械の上面と同じ高さ) を拾わないため
        k = 2 * GROW_PX + 1
        rv, ru = np.nonzero(cv2.dilate(mask, np.ones((k, k), np.uint8)) & (1 - mask))
        ro, rd = self.rays(data, ru, rv)
        rz = ro[2] + rd[:, 2] * depth[rv, ru]
        add = (rz > BAR_TOP_Z + 0.002) & (rz < z_top + 0.002)
        u, v = np.concatenate([u, ru[add]]), np.concatenate([v, rv[add]])
        origin, d = self.rays(data, u, v)
        z = origin[2] + d[:, 2] * depth[v, u]
        # 真上から見た地図に描き直す (透視の影響を消す)。
        # カメラは斜めから見ている部分があるので、厚い部分 (柄) は側面まで写り、画像上では実際より太く見える。
        # 画素をそのまま数えると重心が厚い側に寄る (試作ではメスで 3mm)。そこで、器械の表面 (上面・側面・薄い刃) に
        # 当たっている画素は深度から求めた本当の 3D 位置に置く。側面の点は器械の縁の真上に重なるので、面積が増えない。
        # 隙間から下の天板が見えている画素は、本当の位置だと遠近で 2cm 近く外にずれるので、上面の高さの平面に投影する
        on_inst = z > z_top - THICK - 0.001
        lam = np.where(on_inst, depth[v, u], (z_top - origin[2]) / d[:, 2])
        xy = _grid_points(origin[:2] + d[:, :2] * lam[:, None])

        c = xy.mean(0)
        q = xy - c
        # 主成分分析: 共分散行列の固有ベクトルのうち、固有値 (= その方向の分散) が最大のものが長手方向
        evals, evecs = np.linalg.eigh(np.cov(q.T))  # 固有値の小さい順に並ぶ
        e = evecs[:, 1]
        elong = float(np.sqrt(max(evals[0], 0) / max(evals[1], 1e-12)))

        # 先端の判定: 長手方向の符号 (e か −e か) を、形状モデルの幅のプロファイルとの一致度で決める
        s, t = q @ e, q @ np.array([-e[1], e[0]])
        prof = _profile(s, t)
        tmpl = self.templates[kind].profile
        err_fwd = np.abs(prof - tmpl).sum()  # e の向きが先端だと仮定したときのずれ
        err_rev = np.abs(prof[::-1] - tmpl).sum()  # 逆向き (s → −s) だと仮定したときのずれ
        if err_rev < err_fwd:
            e = -e
        margin = float(abs(err_fwd - err_rev) / max(err_fwd + err_rev, 1e-9))

        grasp_xy = c + e * self.templates[kind].com_offset
        # 把持点の高さ = 上面 − 厚みの半分 (器械の厚みの中央を指で挟む)
        pos = np.array([grasp_xy[0], grasp_xy[1], z_top - THICK / 2])
        return Estimate(kind=kind, conf=float(conf), pos=pos, yaw=float(np.arctan2(e[1], e[0])),
                        tip_margin=margin, elongation=elong, n_px=len(u), polygon=poly)

    def locate(self, data: mujoco.MjData) -> tuple[dict[str, Estimate], np.ndarray]:
        """撮影して全器械を推定する。同じ種類が複数検出されたら確信度が一番高いものを採る。"""
        rgb, depth = self.capture(data)
        out: dict[str, Estimate] = {}
        for kind, conf, poly in self.detect(rgb):
            if kind in out and out[kind].conf >= conf:
                continue
            est = self.estimate_one(data, depth, kind, conf, poly)
            if est is not None:
                out[kind] = est
        return out, rgb


def draw(rgb: np.ndarray, ests: dict[str, Estimate], truth: dict[str, tuple[np.ndarray, float]] | None = None,
         locator: Locator | None = None, data: mujoco.MjData | None = None) -> np.ndarray:
    """推定結果を画像に描く: 輪郭 + 把持点 + 先端方向の矢印 (色付き)。truth を渡すと真値の矢印 (白) も重ねる。"""
    from perception.show_labels import COLORS

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        # world → 画像 (rays の逆変換)
        R = data.cam_xmat[locator.cam].reshape(3, 3)
        pc = R.T @ (p - data.cam_xpos[locator.cam])
        f = HEIGHT / (2 * np.tan(np.radians(locator.model.cam_fovy[locator.cam]) / 2))
        return int(round(pc[0] / -pc[2] * f + WIDTH / 2 - 0.5)), int(round(-pc[1] / -pc[2] * f + HEIGHT / 2 - 0.5))

    for kind, est in ests.items():
        col = COLORS[INSTRUMENT_NAMES.index(kind)]
        cv2.polylines(img, [est.polygon.astype(np.int32)], True, col, 1)
        if locator is None:
            continue
        p0 = to_px(est.pos)
        tip = est.pos + 0.06 * np.array([np.cos(est.yaw), np.sin(est.yaw), 0])
        if truth and kind in truth:
            tp, ty = truth[kind]
            cv2.arrowedLine(img, to_px(tp), to_px(tp + 0.06 * np.array([np.cos(ty), np.sin(ty), 0])),
                            (255, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(img, p0, to_px(tip), col, 2, tipLength=0.3)
        cv2.circle(img, p0, 4, col, -1)
        cv2.putText(img, f"{kind} {est.conf:.2f}", (p0[0] + 8, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img
