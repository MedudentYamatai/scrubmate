"""Phase 6: 動画の合成。シミュレーター内の時刻 (data.time) を基準に、映像のコマと音声を並べる。

画面 (1920x1080。講座から配られた OP/ED と同じ)
  +---------------------------------+----------+
  |                                 | 検出カメラ |
  |   全体カメラ (1440x810)          | (小窓)    |
  |                                 +----------+
  |                                 | 工程・台帳 |
  +---------------------------------+----------+
  | 字幕: 術者の音声 → 書き起こし → 補正 → 解釈 / ロボットの返事        |
  +--------------------------------------------+

本編 (ロボットが動く部分) は、映像も声も 1.5 倍速 (SPEED)。シミュレーター内の 1.5 秒を動画の 1 秒にする。
声はシミュレーター内の時刻に並べた音声トラックを、最後に高さを変えずに 1.5 倍に縮める (ffmpeg の atempo)。
ロボットが動かないタイトル (構成の説明) と結果のまとめは等倍 (ユーザーと相談して決定)。

時刻をシミュレーター内の時刻でそろえるのは、Whisper や描画にかかる計算時間が毎回変わっても、同じ動画になるようにするため。
音声 (術者の wav・ロボットの読み上げ) は「何秒の所に何を鳴らすか」を記録しておき、最後に 1 本の音声トラックに重ねて動画と合わせる。
"""

import hashlib
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np

from demo.jptext import put_lines, wrap

W, H = 1920, 1080
MAIN_W, MAIN_H = 1440, 810
PANEL_W = W - MAIN_W
SUB_H = H - MAIN_H
FPS = 30
# 本編の再生速度。シミュレーター内の時刻 t は、本編の動画の t / SPEED 秒の所に来る
SPEED = 1.5
SAMPLE_RATE = 16000
ROBOT_VOICE = "Eddy (日本語（日本）)"  # ロボットの声。術者 (Kyoko) と聞き分けられる声
TTS_DIR = Path("outputs/phase6/tts")

WHITE, GRAY, GREEN, RED, YELLOW, CYAN = (255, 255, 255), (180, 180, 180), (120, 230, 140), (255, 90, 90), \
    (255, 215, 110), (120, 210, 255)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


# 読み上げで読み間違えやすい語は、読み上げる文だけかなに置き換える (字幕は漢字のまま)
TTS_READINGS = {"鑷子": "せっし", "持針器": "じしんき", "器械": "きかい", "閉創": "へいそう", "剥離": "はくり",
                "止血": "しけつ", "縫合": "ほうごう", "照合": "しょうごう", "未返却": "みへんきゃく", "術者": "じゅつしゃ"}


def tts(text: str) -> Path:
    """ロボットの返事を読み上げ音声にする (同じ文は作り直さない)。"""
    spoken = text
    for k, v in TTS_READINGS.items():
        spoken = spoken.replace(k, v)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TTS_DIR / f"{hashlib.sha256((ROBOT_VOICE + spoken).encode()).hexdigest()[:12]}.wav"
    if not path.exists():
        subprocess.run(["say", "-v", ROBOT_VOICE, "--file-format=WAVE", "--data-format=LEI16@16000",
                        "-o", str(path), spoken], check=True)
    return path


@dataclass
class Subtitle:
    start: float
    end: float
    lines: list[tuple[str, tuple[int, int, int]]]
    slot: str  # surgeon / robot (字幕の段)


@dataclass
class Overlay:
    """画面に重ねる情報 (agent や director が書き換える)。"""
    pip: np.ndarray | None = None  # 検出カメラの画像 (RGB)
    pip_caption: str = ""
    subtitles: list[Subtitle] = field(default_factory=list)
    notes: list[tuple[float, str]] = field(default_factory=list)  # 状況の字幕 (時刻, 文)
    alert: str | None = None
    labels: list[tuple[str, np.ndarray]] = field(default_factory=list)  # シーン紹介の名札 (文, world 座標)
    status: Callable | None = None  # () -> 右パネルに出す行 (工程・台帳)

    def note(self, t: float, text: str) -> None:
        self.notes.append((t, text))



class Composer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, path: Path, camera: str = "overview_cam"):
        self.model, self.data = model, data
        # 全体カメラを 1440x810 で描くために、オフスクリーンの描画領域を広げる (XML の既定は 1280x960)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
        model.vis.global_.offheight = max(model.vis.global_.offheight, H)
        self.renderer = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
        self.cam = model.camera(camera).id
        self.camera = camera
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(self.path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        self.ov = Overlay()
        self.audio: list[tuple[float, Path]] = []
        self.t0 = data.time
        self.next_t = data.time
        self.n_frames = 0
        self._side_cache: tuple | None = None
        self._sub_cache: tuple | None = None

    # ---- 音 ----
    def play(self, path: Path) -> float:
        """今の時刻 (シミュレーター内) から wav を鳴らす。長さ [s] を返す。"""
        self.audio.append((self.data.time - self.t0, Path(path)))
        return wav_seconds(path)

    def say(self, text: str) -> float:
        p = tts(text)
        dur = self.play(p)
        self.ov.subtitles.append(Subtitle(self.data.time, self.data.time + dur + 2.0,
                                          [(f"【ロボット】「{text}」", CYAN)], "robot"))
        return dur

    # ---- 画 ----
    def _project(self, p: np.ndarray) -> tuple[int, int] | None:
        R = self.data.cam_xmat[self.cam].reshape(3, 3)
        pc = R.T @ (p - self.data.cam_xpos[self.cam])
        if pc[2] >= 0:
            return None
        f = MAIN_H / (2 * np.tan(np.radians(self.model.cam_fovy[self.cam]) / 2))
        return int(pc[0] / -pc[2] * f + MAIN_W / 2), int(-pc[1] / -pc[2] * f + MAIN_H / 2)

    def _side(self, note: str | None) -> np.ndarray:
        """右パネル (検出カメラの小窓 + 工程・台帳 + 状況)。中身が変わったときだけ描き直す。"""
        status = self.ov.status() if self.ov.status else []
        key = (id(self.ov.pip), self.ov.pip_caption, tuple(status), note)
        if self._side_cache and self._side_cache[0] == key:
            return self._side_cache[1]
        img = np.full((MAIN_H, PANEL_W, 3), 24, np.uint8)
        img = put_lines(img, [("検出カメラ (真上から・RGB+深度)", GRAY)], (12, 8), size=20, bg=None)
        if self.ov.pip is not None:
            pip = cv2.resize(self.ov.pip, (PANEL_W - 16, int((PANEL_W - 16) * 0.75)))
            img[40:40 + pip.shape[0], 8:8 + pip.shape[1]] = pip
            y = 46 + pip.shape[0]
        else:
            y = 46 + int((PANEL_W - 16) * 0.75)
        if self.ov.pip_caption:
            img = put_lines(img, [(ln, YELLOW) for ln in wrap(self.ov.pip_caption, 20, PANEL_W - 24)], (12, y),
                            size=20, bg=None)
        img = put_lines(img, status, (12, y + 40), size=22, bg=None, line_gap=1.4)
        if note:
            lines = wrap(f"状況: {note}", 20, PANEL_W - 24)[:3]
            img = put_lines(img, [(ln, GREEN) for ln in lines], (12, MAIN_H - 16 - 28 * len(lines)), size=20, bg=None)
        self._side_cache = (key, img)
        return img

    def _subs(self, t: float) -> np.ndarray:
        """下の字幕の段。行は (文, 色) か (文, 色, 大きさ)。画面の幅で折り返し、段からはみ出す行は描かない。"""
        active = [s for s in self.ov.subtitles if s.start <= t < s.end]
        # 術者の字幕は最新の 1 つだけ。ロボットの返事も最新の 1 つ
        sur = [s for s in active if s.slot == "surgeon"][-1:]
        rob = [s for s in active if s.slot == "robot"][-1:]
        key = tuple(id(s) for s in sur + rob)
        if self._sub_cache and self._sub_cache[0] == key:
            return self._sub_cache[1]
        img = np.full((SUB_H, W, 3), 12, np.uint8)
        y = 16
        for ln in [ln for s in sur + rob for ln in s.lines]:
            text, col, size = (*ln, 30) if len(ln) == 2 else ln
            for piece in wrap(text, size, W - 48):
                if y + size > SUB_H - 8:
                    break
                img = put_lines(img, [(piece, col)], (24, y), size=size, bg=None)
                y += int(size * 1.4)
        self._sub_cache = (key, img)
        return img

    def frame(self) -> np.ndarray:
        t = self.data.time
        self.renderer.update_scene(self.data, camera=self.camera)
        main = self.renderer.render()
        if self.ov.labels:
            main = main.copy()
            for text, p in self.ov.labels:
                px = self._project(np.asarray(p, float))
                if px:
                    cv2.circle(main, px, 6, YELLOW, -1)
                    main = put_lines(main, [(text, YELLOW)], (px[0] + 10, px[1] - 34), size=28, bg=(0, 0, 0, 170), pad=6)
        if self.ov.alert:
            main = main.copy()
            if int(t * 2) % 2 == 0:  # 点滅させる
                cv2.rectangle(main, (0, 0), (MAIN_W - 1, MAIN_H - 1), (255, 40, 40), 14)
            main = put_lines(main, [("【警告】" + self.ov.alert, WHITE)], (40, 40), size=40, bg=(200, 20, 20, 220), pad=14)
        if SPEED != 1:
            main = put_lines(main if self.ov.alert or self.ov.labels else main.copy(), [(f"{SPEED:g}倍速", WHITE)],
                             (MAIN_W - 130, MAIN_H - 50), size=28, bg=(0, 0, 0, 150), pad=6)
        out = np.empty((H, W, 3), np.uint8)
        out[:MAIN_H, :MAIN_W] = main
        notes = [n for tn, n in self.ov.notes if t - 4.0 * SPEED <= tn <= t][-1:]
        out[:MAIN_H, MAIN_W:] = self._side(notes[0] if notes else None)
        out[MAIN_H:] = self._subs(t)
        return out

    def capture(self) -> None:
        """制御周期ごとに呼ぶ。動画の 1 コマ (シミュレーター内では SPEED / FPS 秒) の刻みを過ぎていたら 1 コマ描く。"""
        while self.data.time + 1e-9 >= self.next_t:
            self.writer.append_data(self.frame())
            self.next_t += SPEED / FPS
            self.n_frames += 1

    def close(self) -> float:
        self.writer.close()
        self.renderer.close()
        return self.n_frames / FPS  # 動画の長さ [s]


def card(lines: list[tuple[str, tuple[int, int, int], int]], bg=(250, 250, 248)) -> np.ndarray:
    """文字だけの画面 (タイトル・結果のまとめ)。lines = (文, 色, 大きさ)。"""
    img = np.full((H, W, 3), bg, np.uint8)
    y = 120
    for text, col, size in lines:
        img = put_lines(img, [(text, col)], (140, y), size=size, bg=None)
        y += int(size * 1.6)
    return img


def _write_wav(path: Path, track: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())


def _mix(clips: list[tuple[float, Path]], seconds: float) -> np.ndarray:
    track = np.zeros(int(seconds * SAMPLE_RATE) + SAMPLE_RATE, np.float32)
    for ts, p in clips:
        a = load_wav16k(p)
        i = int(ts * SAMPLE_RATE)
        track[i:i + len(a)] += a[:max(0, len(track) - i)]
    return track


def assemble(parts: list[tuple[str, object]], audio: list[tuple[float, Path]], main_key: str, out: Path,
             main_sim_seconds: float) -> float:
    """parts = [(名前, 動画のパス | スライド | (画像, 秒))] を順につなぐ。audio = (本編のシミュレーター内の時刻, wav) は
    本編の音声トラックにまとめ、高さを変えずに SPEED 倍速に縮めて (atempo) 本編の位置に重ねる。
    OP/ED は 29.97fps だが、30fps として並べる (3 秒で 0.003 秒の差なので気にしない)。"""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = out.with_suffix(".noaudio.mp4")
    wr = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    t = 0.0
    offsets = {}
    for key, src in parts:
        offsets[key] = t
        if hasattr(src, "frames"):  # 動きのあるスライド (demo/slides.py)
            for fr in src.frames():
                wr.append_data(fr)
                t += 1 / FPS
        elif isinstance(src, Path):
            for fr in imageio.get_reader(src):
                if fr.shape[:2] != (H, W):
                    fr = cv2.resize(fr, (W, H))
                wr.append_data(fr[..., :3])
                t += 1 / FPS
        else:
            img, sec = src
            for _ in range(int(round(sec * FPS))):
                wr.append_data(img)
            t += round(sec * FPS) / FPS
    wr.close()
    main_raw, main_fast = out.with_suffix(".main_raw.wav"), out.with_suffix(".main_fast.wav")
    _write_wav(main_raw, _mix(audio, main_sim_seconds))
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(main_raw), "-filter:a", f"atempo={SPEED}",
                    "-ar", str(SAMPLE_RATE), "-ac", "1", str(main_fast)], check=True)
    track = np.zeros(int(t * SAMPLE_RATE) + SAMPLE_RATE, np.float32)
    fast = load_wav16k(main_fast)
    i = int(offsets[main_key] * SAMPLE_RATE)
    track[i:i + len(fast)] += fast[:max(0, len(track) - i)]
    wav_path = out.with_suffix(".wav")
    _write_wav(wav_path, track)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav_path), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)
    for f in (tmp, main_raw, main_fast):
        f.unlink()
    return t
