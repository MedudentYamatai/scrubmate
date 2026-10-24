"""オフスクリーン描画と mp4 録画。"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


class Recorder:
    """シミュレーション時間で fps ごとに 1 フレーム撮って mp4 に書く。

    物理は 2ms 刻みで進むので、毎ステップ撮ると重すぎる。
    「次に撮るべき時刻」を持っておき、それを過ぎたときだけ描画する。
    """

    def __init__(self, model: mujoco.MjModel, path: Path, camera: str = "overview_cam",
                 width: int = 1280, height: int = 720, fps: int = 30):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera, self.fps = camera, fps
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # macro_block_size=1: 1280x720 のような 16 の倍数でない解像度でも勝手にリサイズさせない
        self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        self.next_t = 0.0
        self.n_frames = 0

    def maybe_capture(self, data: mujoco.MjData) -> np.ndarray | None:
        if data.time + 1e-9 < self.next_t:
            return None
        self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()
        self.writer.append_data(frame)
        self.next_t += 1.0 / self.fps
        self.n_frames += 1
        return frame

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
        print(f"saved {self.path} ({self.n_frames} frames, {self.n_frames / self.fps:.1f}s)")


def snapshot(model: mujoco.MjModel, data: mujoco.MjData, path: Path, camera: str,
             width: int = 1280, height: int = 720) -> np.ndarray:
    from PIL import Image

    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera)
        img = r.render()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return img
