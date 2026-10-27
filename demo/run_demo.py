"""Phase 6: デモシナリオを通しで実行し、動画を書き出す。

    uv run python -m demo.run_demo                 # OP + イントロ (背景・コンセプト・仕組み) + 本編 + 結果・展望 + ED → outputs/phase6/demo.mp4
    uv run python -m demo.run_demo --main-only     # 本編だけ (確認用。OP/ED をつながない)
    uv run python -m demo.run_demo --no-video      # 映像を描かずに流れだけ確かめる (速い)
    uv run python -m demo.run_demo --main-only --fallback   # 予備のルールだけで解釈して本編を書き出す (実験用)

台本 (BEATS): 「どんな状況になったら」「何をするか」の組を順に並べたもの。
秒数ではなく状況 (ロボットが指示待ち・術者の手が空いた など) で次へ進むので、ロボットの動きの長さが変わっても順番は崩れない。
術者の台詞は Phase 4 の音声 (macOS の読み上げ Kyoko)。台詞を聞き終えた時点で Whisper で書き起こし、解釈してロボットの指示の列に入れる。
解釈は Phase 4 のキャッシュ (language/cache/) で再現できる (同じ文・同じ状況なら Gemini を呼ばない)。
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from control.motion import ArmController
from demo.agent import ScrubNurse
from demo.surgeon import Surgeon
from demo import slides
from demo.video import CYAN, GRAY, GREEN, RED, SPEED, WHITE, YELLOW, Composer, Subtitle, assemble, tts, wav_seconds
from inventory.eval_count import settle
from inventory.ledger import PLACE_JA, SURGEON
from language.asr import transcribe
from language.interpret import Interpreter
from language.utterances import UTTERANCES
from sim import robot, scene
from sim.instruments import JA_NAMES

OUT_DIR = Path("outputs/phase6")
SPOKEN = {u.id: u.text for u in UTTERANCES}  # 台詞 id → 術者が実際に言った文 (字幕に出す)
PIP_DIR = OUT_DIR / "pip"
AUDIO_DIR = Path("assets/audio/demo")
BRANDING = Path("assets/branding")
# 閉創前カウントで並べ直しを見せる場面: 術者が返却トレイでペアンの上に鑷子を少しずらして重ねる。
# 重なりの崩れ方は条件で大きく変わる。30 通り (ずらす量 2〜4cm × 向き −8/0/+8° × 上下の順) を試し、
# 「本当に載っている (上が 6mm 高い)」「並べ直しが 1 本 1 回で済む」「最後の警告が正しい」を満たし、
# 隣の条件 (2cm/+8°, 2.5cm/0°, 3cm/+8°) でも同じように成功する所を選んだ (learning_log の Phase 6)
PILE_DY = 0.025  # 上の器械をずらす量 [m]
PILE_YAW_DEG = 8.0  # 上に重ねる器械の向きのずれ [°]
PILE = ("pean", "forceps")  # (下, 上)
SPEECH_GAP = 0.6  # 声と声の間 [s] (シミュレーター内。動画では 1.5 倍速なので 0.4 秒)


class Director:
    def __init__(self, model, data, comp: Composer | None, use_llm: bool = True):
        self.model, self.data, self.comp = model, data, comp
        self.surgeon = Surgeon(model, data)
        self.ctrl = ArmController(model, data, on_step=self.on_step)
        self.timers: list[tuple[float, callable]] = []
        self.beat_t = 0.0  # 直前の場面を始めた時刻
        self.log: list[dict] = []
        ov = comp.ov if comp else None
        self.agent = ScrubNurse(model, data, self.ctrl, self.surgeon, speak=self.robot_say, show=self.show,
                                note=self.note, interpreter=Interpreter(use_llm=use_llm))
        if ov is not None:
            ov.status = self.status_lines
        self.beats = self.make_beats()
        self.finished = False
        self.n_pip = 0
        self.speech_free_t = 0.0  # この時刻まで誰かが話している

    # ---- 表示・音 ----
    # ---- 声 (術者とロボットが同時に話さないように、声は 1 本の列で順番に鳴らす) ----
    def speech_free(self) -> bool:
        return self.data.time >= self.speech_free_t

    def robot_say(self, text: str) -> None:
        """ロボットの返事。誰かが話している間は、話し終わるまで待ってから話す。"""
        if not self.speech_free():
            self.timers.append((self.speech_free_t, lambda: self.robot_say(text)))
            return
        # 映像を描かないときも声の長さを数える (描くときと同じ時間の流れにして、動きの経過を変えない)
        dur = wav_seconds(tts(text))
        self.speech_free_t = self.data.time + dur + SPEECH_GAP
        self.log.append({"t": round(self.data.time, 2), "robot": text})
        print(f"[{self.data.time:6.1f}s] ロボット: {text}", flush=True)
        if self.comp:
            self.comp.say(text)

    def show(self, rgb: np.ndarray, caption: str) -> None:
        # 検出カメラの画像を毎回保存する (動画を描かないときの確認用、レポートの図の材料)
        PIP_DIR.mkdir(parents=True, exist_ok=True)
        self.n_pip += 1
        cv2.imwrite(str(PIP_DIR / f"{self.n_pip:03d}_{self.data.time:06.1f}s.jpg"), rgb[..., ::-1])
        if self.comp:
            self.comp.ov.pip, self.comp.ov.pip_caption = np.ascontiguousarray(rgb), caption

    def note(self, text: str) -> None:
        self.log.append({"t": round(self.data.time, 2), "note": text})
        print(f"[{self.data.time:6.1f}s]   {text}", flush=True)
        if self.comp:
            self.comp.ov.note(self.data.time, text)

    def status_lines(self) -> list:
        a = self.agent
        lines = [(f"工程: {a.step or '開始前'}", WHITE), (f"ロボット: {PHASE_JA[a.phase]}", YELLOW), ("", WHITE),
                 ("台帳", GRAY)]
        if a.ledger.established:
            for k, s in a.ledger.state.items():
                lines.append((f"  {JA_NAMES[k]:　<4} {PLACE_JA[s]}", RED if s == SURGEON else GREEN))
        else:
            lines.append(("  (初期カウント前)", GRAY))
        return lines

    # ---- 術者の台詞 ----
    def say(self, uid: str) -> None:
        wav = AUDIO_DIR / f"{uid}.wav"
        t = self.data.time
        dur = wav_seconds(wav)
        self.speech_free_t = t + dur + SPEECH_GAP
        print(f"[{t:6.1f}s] 術者: ({uid})", flush=True)
        if self.comp:
            self.comp.play(wav)
            # 字幕は術者が実際に言った文 (台詞の正解)。書き起こしの誤りや補正の過程は字幕に出さない (ユーザーと相談して決定)
            self.comp.ov.subtitles.append(Subtitle(t, t + dur + 0.6, [(f"【術者】「{SPOKEN[uid]}」", WHITE, 32)],
                                                   "surgeon"))
        # 聞き終えてから書き起こす (人も、言い終わるまで待ってから動く)
        self.timers.append((t + dur + 0.2, lambda: self.heard(uid, wav)))

    def heard(self, uid: str, wav: Path) -> None:
        text = transcribe(wav)
        r = self.agent.hear(text)
        it = r.intent
        # キャッシュは Gemini が前に出した答えなので「Gemini」と出す (キャッシュや予備は動画では見せない裏の仕組み)
        src = {"cache": "Gemini", "gemini": "Gemini", "fallback": "ルール"}[r.source]
        print(f"[{self.data.time:6.1f}s]   書き起こし「{text}」 補正「{r.text}」 → {it.model_dump()} ({r.source})", flush=True)
        self.log.append({"t": round(self.data.time, 2), "uid": uid, "heard": text, "text": r.text,
                         "intent": it.model_dump(), "source": r.source})
        if self.comp:
            t = self.data.time
            inst = JA_NAMES.get(it.instrument, "-") if it.instrument else "-"
            lines = [(f"【術者】「{SPOKEN[uid]}」", WHITE, 32),
                     (f"解釈 [{src}]: {it.intent} / 器械 {inst} / 工程 {it.step or '-'}", YELLOW, 28),
                     (f"理由: {it.reason}", GRAY, 24)]
            self.comp.ov.subtitles.append(Subtitle(t, t + 7.0, lines, "surgeon"))

    # ---- 台本 ----
    def make_beats(self) -> list:
        a, s, d = self.agent, self.surgeon, self.data

        def calm(sec: float = 1.0):
            # ロボットが指示待ち・指示の列が空・術者の手が空いている・誰も話していない状態が、直前の場面から sec 秒以上
            return lambda: (a.phase == "idle" and not a.queue and not s.busy and self.speech_free()
                            and d.time - self.beat_t >= sec)

        def on_drape(kind):
            return lambda: kind in s.on_drape

        def both(*fs):
            return lambda: all(f() for f in fs)

        return [
            ("シーン紹介", lambda: d.time >= 4.0 * SPEED, self.end_intro),
            ("1. 器械カウントお願いします", lambda: d.time >= 4.5 * SPEED, lambda: self.say("d1_count")),
            ("2. メス", both(calm(1.0), lambda: a.ledger.established), lambda: self.say("d2_mesu")),
            ("3. 鑷子ちょうだい", both(calm(0.8), on_drape("scalpel")), lambda: self.say("d3_sesshi")),
            ("4. 術者がメスを返す (黙って)", both(calm(0.8), on_drape("forceps")),
             lambda: s.start("return scalpel", s.return_to_tray("scalpel", slot=1))),
            # 回収でメスを運んでいる途中に「次」(優先順位: 今の 1 本を置いてから渡す)
            ("5. 次 (回収の途中)", lambda: a.phase == "retrieve" and d.time - a.phase_t0 >= 3.5 and self.speech_free(),
             lambda: self.say("d5_tsugi")),
            ("6. 縫合するのであれ取って", both(calm(1.0), on_drape("pean"), lambda: a.ledger.state["scalpel"] == "table"),
             lambda: self.say("d6_are")),
            ("7a. 術者が下になる器械を返す", both(calm(1.0), on_drape("needle_holder")),
             lambda: s.start(f"return {PILE[0]}", s.return_to_tray(PILE[0], slot=1, go_rest=False))),
            ("7b. 術者が上になる器械を少しずらして重ねて返す", lambda: not s.busy and d.time - self.beat_t >= 0.3,
             lambda: s.start(f"return {PILE[1]}", s.return_to_tray(PILE[1], slot=1, dy=PILE_DY,
                                                                   yaw_dev=np.radians(PILE_YAW_DEG)))),
            # ペアンの返却を見張りで台帳に記録できたら (= 照合の前に台帳が実物と合っている) すぐ言う
            ("8. 閉創前カウント", lambda: (a.ledger.state["pean"] == "tray" and a.ledger.state["forceps"] == "tray"
                                       and self.speech_free()),
             lambda: (self.log_pile(), self.say("d8_heisou"))),
            ("終わり", lambda: a.phase == "halted" and d.time - a.phase_t0 >= 6.0 * SPEED, self.finish),
        ]

    def end_intro(self) -> None:
        if self.comp:
            self.comp.ov.labels = []

    def log_pile(self) -> None:
        """重なりの確認用: 照合カウントを頼む時点の 2 本の真値 (評価専用。ロボットの判断には使わない)。"""
        lo, hi = (self.data.body(k).xpos for k in PILE)
        self.log.append({"t": round(self.data.time, 2), "pile_truth": {
            "dxy_mm": round(float(np.linalg.norm(hi[:2] - lo[:2]) * 1000), 1),
            "top_dz_mm": round(float((hi[2] - lo[2]) * 1000), 1)}})

    def finish(self) -> None:
        self.finished = True

    def on_step(self) -> None:
        """制御周期ごと: 術者を 1 歩進める → 時刻の来た予定を実行 → 台本の次の場面の条件を見る → 1 コマ描く。"""
        self.surgeon.update()
        # 警告の帯は、照合の最終判定で止まったときだけ出す (並べ直す前の途中経過では出さない)
        if self.comp and self.agent.phase == "halted" and self.comp.ov.alert is None:
            self.comp.ov.alert = self.agent.ledger_message
        due = [f for t, f in self.timers if self.data.time >= t]
        self.timers = [(t, f) for t, f in self.timers if self.data.time < t]
        for f in due:
            f()
        if self.beats and self.beats[0][1]():
            name, _, act = self.beats.pop(0)
            print(f"[{self.data.time:6.1f}s] ==== {name}", flush=True)
            self.log.append({"t": round(self.data.time, 2), "beat": name})
            self.beat_t = self.data.time
            act()
        if self.comp:
            self.comp.capture()

    def run(self, limit: float = 400.0) -> None:
        while not self.finished and self.data.time < limit:
            self.agent.tick()


PHASE_JA = {"idle": "指示待ち (トレイを見張る)", "handoff": "器械を渡している", "retrieve": "返却された器械を回収",
            "count": "カウント中", "halted": "停止中 (確認待ち)"}

INTRO_LABELS = [
    ("ロボット (Franka Panda)", [0.0, 0.0, 1.0]),
    ("器械台 (5 本)", [0.50, -0.05, 0.78]),
    ("返却トレイ", [0.50, 0.40, 0.80]),
    ("術者の手", None),
    ("術野", [0.20, 0.80, 0.98]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="Gemini もキャッシュも使わず、予備のルールだけで解釈する (実験用)")
    ap.add_argument("--pile-dy", type=float, default=PILE_DY, help="トレイで上の器械を下の器械からずらす量 [m]")
    ap.add_argument("--pile-yaw", type=float, default=PILE_YAW_DEG, help="上に重ねる器械の向きのずれ [°]")
    ap.add_argument("--pile-swap", action="store_true", help="重ねる順番を逆にする (鑷子が下、ペアンが上)")
    args = ap.parse_args()
    globals()["PILE_DY"] = args.pile_dy
    globals()["PILE_YAW_DEG"] = args.pile_yaw
    if args.pile_swap:
        globals()["PILE"] = PILE[::-1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in PIP_DIR.glob("*.jpg"):
        f.unlink()
    model, data = scene.build()
    # デモ動画用の構図: 元の全体カメラより寄せ、少し高い所から見下ろす (腕を上げてもロボット全体が入り、器械が大きく見える)。
    # Phase 1〜3 の動画と XML の設定は変えない
    cam = model.camera("overview_cam").id
    model.cam_fovy[cam], model.cam_pos[cam] = 46.0, [1.30, 0.80, 1.85]
    model.body_pos[model.body("scene_center").id] = [0.30, 0.18, 0.88]
    robot.set_arm_qpos(model, data, robot.PARK_QPOS)
    settle(model, data)
    comp = None if args.no_video else Composer(model, data, OUT_DIR / "main.mp4")
    director = Director(model, data, comp, use_llm=not args.fallback)
    if comp:
        mujoco.mj_forward(model, data)
        comp.ov.labels = [(t, p if p is not None else data.mocap_pos[0].copy()) for t, p in INTRO_LABELS]
    t0 = time.time()
    director.run()
    print(f"\nシミュレーター内 {data.time:.1f} 秒 (計算 {time.time() - t0:.0f} 秒)")
    (OUT_DIR / "demo_log.json").write_text(json.dumps(director.log, ensure_ascii=False, indent=1))
    if comp:
        sec = comp.close()
        print(f"本編 {sec:.1f} 秒 → {comp.path}")
        if args.main_only:
            out = OUT_DIR / "main_with_audio.mp4"
            total = assemble([("main", comp.path)], comp.audio, "main", out, data.time)
        else:
            out = OUT_DIR / "demo.mp4"
            # イントロ (背景 → コンセプト → 仕組み) と結果・展望は等倍。仕組みの縮小画像は、この実行の記録から作る
            thumbs = slides.make_thumbs(PIP_DIR, comp.path, OUT_DIR / "demo_log.json", AUDIO_DIR / "d3_sesshi.wav", SPEED)
            parts = [("op", BRANDING / "op.mp4"), ("background", slides.background_slide()),
                     ("concept", slides.concept_slide()), ("pipeline", slides.pipeline_slide(thumbs)),
                     ("main", comp.path), ("results", slides.results_slide()), ("outlook", slides.outlook_slide()),
                     ("ed", BRANDING / "ed.mp4")]
            total = assemble(parts, comp.audio, "main", out, data.time)
        print(f"saved {out} ({total:.1f} 秒)")


if __name__ == "__main__":
    main()
