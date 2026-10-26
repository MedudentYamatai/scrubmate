"""Phase 2: 検出用カメラの合成画像と、YOLO-seg 形式の正解ラベルを自動で作る。

    uv run python -m perception.gen_dataset                     # 1500 枚 (train 1200 / val 300)
    uv run python -m perception.gen_dataset -n 200 --out datasets/quick
    uv run python -m perception.gen_dataset --no-random --out datasets/norand   # 実験用: 見た目のランダム化なし
    uv run python -m perception.gen_dataset -n 600 --spread 2 --crowd 0.5 --out datasets/crowd   # 実験用: ずれ 2 倍 + トレイで重なり

正解ラベルは MuJoCo の segmentation レンダリングで作る:
  通常の描画は「各ピクセルの色」を返すが、segmentation モードは「各ピクセルに写っている物体の ID」を返す。
  ID → 器械名 の対応はシミュレーターが知っているので、人手のアノテーション無しでマスクが手に入る。

ドメインランダム化 (毎枚変える):
  配置   … どの器械がどのスロット (器械台 5 + 返却トレイ 3) にあるか、欠品、位置 ±1cm、向き (逆向き含む)
  見た目 … 照明 (明るさ・位置)、天板/バー/トレイの色、器械の金属の明るさ、カメラ位置・角度・画角の微小ずれ、
           画像ノイズ、腕の写り込み (一部の画像だけ腕を器械台の上に出して器械を隠す)
  見た目を毎回変えると、検出器は「毎回変わらないもの = 器械の形」だけを頼りに学習せざるを得なくなる。

出力 (ultralytics の形式):
  <out>/images/{train,val}/000123.jpg
  <out>/labels/{train,val}/000123.txt   … 1 行 1 器械: class x1 y1 x2 y2 ... (輪郭の頂点、画像サイズで 0〜1 に正規化)
  <out>/meta/{train,val}.jsonl          … 各画像の真値 (器械の位置・向き、カメラ) 。評価・デバッグ用
  <out>/data.yaml
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from ultralytics.data.converter import merge_multi_segment

from control.ik import DiffIK
from control.motion import grasp_rotation
from sim import robot, scene
from sim.instruments import INSTRUMENT_NAMES

CAMERA = "detect_cam"
WIDTH, HEIGHT = 1024, 768  # 4:3。器械台 (+y 側) から返却トレイ (−y 側) までが横に収まる
MIN_PIXELS = 40  # これより見えている面積が小さい器械はラベルにしない (ほぼ完全に隠れている)

# 器械を置ける場所 (把持点の xy)。器械台 5 スロット + 返却トレイ 3 箇所
PLACES = [("table", i, xy) for i, xy in scene.TABLE_SLOTS.items()] + \
         [("tray", i, xy) for i, xy in scene.TRAY_SLOTS.items()]
HIDDEN_POS = np.array([3.0, 3.0, 0.1])  # 欠品の器械は視野の外の床に置いておく


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class Randomizer:
    """model / data の見た目と配置を 1 枚ごとに書き換える。元の値を覚えておき、毎回そこからずらす。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, appearance: bool,
                 spread: float = 1.0, crowd: float = 0.0):
        self.m, self.d, self.rng, self.appearance = model, data, rng, appearance
        self.spread = spread  # 位置・向きのずれの倍率
        self.crowd = crowd  # 返却トレイに 2〜3 本をまとめて落とす (重なりが起きる) 画像の割合
        self.cam_id = model.camera(CAMERA).id
        self.cam_pos0 = model.cam_pos[self.cam_id].copy()
        self.cam_quat0 = model.cam_quat[self.cam_id].copy()
        self.fovy0 = float(model.cam_fovy[self.cam_id])
        self.light_pos0 = model.light_pos.copy()
        self.light_diffuse0 = model.light_diffuse.copy()
        self.mat_ids = {n: model.material(n).id for n in ("table_top", "bar", "tray")}
        self.mat_rgba0 = model.mat_rgba.copy()
        self.inst_geoms = [g for g in range(model.ngeom)
                           if model.body(model.geom_bodyid[g]).name in INSTRUMENT_NAMES]
        self.geom_rgba0 = model.geom_rgba.copy()
        self.headlight0 = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
        self.ik = DiffIK(model)

    def place_instruments(self) -> list[dict]:
        """各器械を、ランダムに選んだ場所へ置く (物理計算はせず、qpos を直接書き込む)。"""
        r = self.rng
        n_present = r.choice([5, 5, 5, 4, 4, 3, 2, 1])  # 全部ある画像を多めに、欠品ありも混ぜる
        present = r.permutation(INSTRUMENT_NAMES)[:n_present]
        places = r.permutation(len(PLACES))
        # 重なりあり: 2〜3 本をトレイの中のランダムな位置に、少しずつ高さを変えて落とす (物理で落ち着かせるのは呼ぶ側)
        crowded = set()
        if r.random() < self.crowd and n_present >= 2:
            crowded = set(present[:r.integers(2, min(3, n_present) + 1)])
            # 残りの器械は器械台のスロットへ (トレイはまとめて落とす器械で使う)
            table = [j for j in places if PLACES[j][0] == "table"]
            rest = [k for k in present if k not in crowded]
            places = {k: table[j] for j, k in enumerate(rest)}
        else:
            places = {k: places[j] for j, k in enumerate(present)}
        truth = []
        for i, kind in enumerate(INSTRUMENT_NAMES):
            jnt = self.m.joint(f"{kind}_free")
            adr = jnt.qposadr[0]
            if kind not in present:
                self.d.qpos[adr:adr + 3] = HIDDEN_POS + [0.3 * i, 0, 0]
                self.d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
                continue
            # 先端の向き: 基本は +x (yaw=0) だが、返却時などに逆向き (yaw=π) になることもある
            yaw = (np.pi if r.random() < 0.3 else 0.0) + r.normal(0, np.radians(6 * self.spread))
            if kind in crowded:
                where, slot = "tray_pile", -1
                yaw += r.normal(0, np.radians(15))  # 術者が無造作に返したように、向きも大きくばらつかせる
                k = sorted(crowded).index(kind)
                cx, cy = scene.TRAY_CENTER
                pos = np.array([r.uniform(cx - 0.02, cx + 0.02), r.uniform(cy - 0.06, cy + 0.06),
                                scene.INSTRUMENT_Z + 0.015 * (k + 1)])
            else:
                where, slot, (x, y) = PLACES[places[kind]]
                d = 0.01 * self.spread
                pos = np.array([x + r.uniform(-d, d), y + r.uniform(-d, d), scene.INSTRUMENT_Z])
            self.d.qpos[adr:adr + 3] = pos
            self.d.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
            truth.append({"kind": kind, "where": where, "slot": int(slot),
                          "pos": pos.round(5).tolist(), "yaw": round(float(yaw), 5)})
        return truth

    def pose_arm(self) -> str:
        """基本は待避姿勢。一部の画像では、器械台の上で何かを掴みに行く途中のような姿勢にして器械を隠す。"""
        r = self.rng
        if not self.appearance or r.random() > 0.15:
            robot.set_arm_qpos(self.m, self.d, robot.PARK_QPOS)
            return "park"
        _, _, (x, y) = PLACES[r.integers(len(PLACES))]
        tgt = np.array([x + r.uniform(-0.05, 0.05), y + r.uniform(-0.05, 0.05), r.uniform(0.80, 1.05)])
        q, _, _ = self.ik.solve(robot.HOME_QPOS, tgt, grasp_rotation(r.uniform(-np.pi, np.pi)), iters=150)
        robot.set_arm_qpos(self.m, self.d, q)
        return "over_table"

    def appearance_jitter(self) -> dict:
        m, r = self.m, self.rng
        # 毎回、元の値に戻してからずらす (ずれが積み重ならないように)
        m.cam_pos[self.cam_id] = self.cam_pos0
        m.cam_quat[self.cam_id] = self.cam_quat0
        m.cam_fovy[self.cam_id] = self.fovy0
        m.light_pos[:] = self.light_pos0
        m.light_diffuse[:] = self.light_diffuse0
        m.mat_rgba[:] = self.mat_rgba0
        m.geom_rgba[:] = self.geom_rgba0
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self.headlight0
        if not self.appearance:
            return {}

        # カメラ: 取り付け位置 ±1cm (高さ ±2cm)、向き ±1.5°、画角 ±1.5°。実機のカメラ取り付け誤差の想定
        m.cam_pos[self.cam_id] = self.cam_pos0 + r.uniform([-0.01, -0.01, -0.02], [0.01, 0.01, 0.02])
        axis = r.normal(size=3)
        dq, out = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.radians(r.uniform(0, 1.5)))
        mujoco.mju_mulQuat(out, self.cam_quat0, dq)
        m.cam_quat[self.cam_id] = out
        m.cam_fovy[self.cam_id] = self.fovy0 + r.uniform(-1.5, 1.5)

        # 照明: 位置を ±0.6m 動かし、明るさを 0.4〜1.6 倍に (無影灯の当たり方の違い)
        m.light_pos[:] = self.light_pos0 + r.uniform(-0.6, 0.6, size=self.light_pos0.shape) * [1, 1, 0.3]
        m.light_diffuse[:] = self.light_diffuse0 * r.uniform(0.4, 1.6, size=(m.nlight, 1))
        h = r.uniform(0.1, 0.4)
        m.vis.headlight.diffuse[:] = h
        m.vis.headlight.ambient[:] = h * r.uniform(0.6, 1.2)

        # 天板・バー・トレイの色: 色相はランダム、明るさは「器械 (明るい金属) と見分けがつく」範囲で広めに
        def rand_color(v_lo, v_hi, sat_hi=0.6):
            hsv = np.uint8([[[r.integers(0, 180), r.uniform(0, sat_hi) * 255, r.uniform(v_lo, v_hi) * 255]]])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
        m.mat_rgba[self.mat_ids["table_top"], :3] = rand_color(0.08, 0.6)
        m.mat_rgba[self.mat_ids["bar"], :3] = rand_color(0.1, 0.45, 0.3)
        m.mat_rgba[self.mat_ids["tray"], :3] = rand_color(0.35, 0.75, 0.15)
        # 器械: 5 本とも同じ灰色のまま、明るさだけ変える (色で区別できないようにするという前提は守る)
        g = r.uniform(0.55, 0.9)
        m.geom_rgba[self.inst_geoms, :3] = g + r.uniform(-0.02, 0.02, size=3)
        return {"table_rgb": m.mat_rgba[self.mat_ids["table_top"], :3].round(3).tolist(), "metal": round(float(g), 3)}


def masks_from_segmentation(model: mujoco.MjModel, seg: np.ndarray) -> dict[str, np.ndarray]:
    """segmentation 画像 (H, W, 2) = (物体 ID, 物体の種類) から、器械ごとの 0/1 マスクを作る。"""
    obj_id, obj_type = seg[..., 0], seg[..., 1]
    is_geom = obj_type == mujoco.mjtObj.mjOBJ_GEOM
    # 各ピクセルの geom → それが属する body → 器械名
    body_of_pixel = np.where(is_geom, model.geom_bodyid[np.clip(obj_id, 0, None)], -1)
    masks = {}
    for kind in INSTRUMENT_NAMES:
        mk = (body_of_pixel == model.body(kind).id).astype(np.uint8)
        if mk.sum() >= MIN_PIXELS:
            masks[kind] = mk
    return masks


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """マスク → 輪郭の頂点列 (N, 2) [px]。
    YOLO-seg のラベルは「1 物体 = 1 本の多角形」。腕に隠されてマスクが 2 つに分かれたときは、
    ultralytics と同じやり方 (細い線でつないで 1 本にする) でまとめる。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = [c.reshape(-1, 2).astype(np.float32) for c in contours if cv2.contourArea(c) >= 4 and len(c) >= 3]
    if not segs:
        return None
    if len(segs) == 1:
        return segs[0]
    return np.concatenate(merge_multi_segment(segs), axis=0)


def add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """カメラのセンサーノイズとピンぼけ。レンダリング画像は実画像よりきれいすぎるので少し汚す。"""
    out = img.astype(np.float32)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.2))
    out += rng.normal(0, rng.uniform(0, 6), size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1500, help="枚数")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=Path("datasets/instruments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-random", action="store_true", help="見た目のランダム化を切る (配置だけランダム)")
    ap.add_argument("--spread", type=float, default=1.0, help="位置・向きのずれの倍率 (1 = ±1cm・±6°)")
    ap.add_argument("--crowd", type=float, default=0.0, help="返却トレイで器械が重なる画像の割合 (0〜1)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, data = scene.build()
    rand = Randomizer(model, data, rng, appearance=not args.no_random, spread=args.spread, crowd=args.crowd)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    # 把持点の site などは画像に写らないようにする (実カメラには写らないもの)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    n_val = int(args.n * args.val_frac)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)
    meta_files = {s: open(args.out / "meta" / f"{s}.jsonl", "w") for s in ("train", "val")}
    n_inst = {k: 0 for k in INSTRUMENT_NAMES}

    for i in range(args.n):
        split = "val" if i >= args.n - n_val else "train"
        truth = rand.place_instruments()
        if any(t["where"] == "tray_pile" for t in truth):
            # 重なりは物理で作る (位置を直接書くと器械どうしがめり込んだ、ありえない形になる)。
            # 腕を待避させて 1 秒分落とし、落ち着いた後の本当の位置を真値として記録し直す
            robot.set_arm_qpos(model, data, robot.PARK_QPOS)
            data.ctrl[:7] = robot.PARK_QPOS
            data.qvel[:] = 0
            for _ in range(500):
                mujoco.mj_step(model, data)
            for t in truth:
                b = data.body(t["kind"])
                t["pos"] = b.xpos.round(5).tolist()
                t["yaw"] = round(float(np.arctan2(b.xmat[3], b.xmat[0])), 5)
        arm = rand.pose_arm()
        look = rand.appearance_jitter()
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=CAMERA, scene_option=opt)
        rgb = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA, scene_option=opt)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        if not args.no_random:
            rgb = add_noise(rgb, rng)
        masks = masks_from_segmentation(model, seg)
        lines = []
        for kind, mk in masks.items():
            poly = mask_to_polygon(mk)
            if poly is None:
                continue
            poly = poly / [WIDTH, HEIGHT]
            lines.append(f"{INSTRUMENT_NAMES.index(kind)} " + " ".join(f"{v:.5f}" for v in poly.flatten()))
            n_inst[kind] += 1
        for t in truth:
            t["visible_px"] = int(masks[t["kind"]].sum()) if t["kind"] in masks else 0

        stem = f"{i:06d}"
        cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        cam = rand.cam_id
        meta_files[split].write(json.dumps({
            "image": f"{stem}.jpg", "arm": arm, "instruments": truth, "look": look,
            "camera": {"pos": model.cam_pos[cam].round(5).tolist(), "quat": model.cam_quat[cam].round(6).tolist(),
                       "fovy": round(float(model.cam_fovy[cam]), 3)},
        }, ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.n}", flush=True)

    for f in meta_files.values():
        f.close()
    renderer.close()
    names = "\n".join(f"  {i}: {k}" for i, k in enumerate(INSTRUMENT_NAMES))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"saved {args.out}  (train {args.n - n_val} / val {n_val})")
    print("ラベル数:", n_inst)


if __name__ == "__main__":
    main()
