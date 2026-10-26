"""台詞の音声 (wav) を macOS の読み上げ (say コマンド) で作る。

    uv run python -m language.make_audio                    # デモ用の声で全台詞 → assets/audio/demo/
    uv run python -m language.make_audio --voices all       # 評価用に 4 種類の声でも作る → assets/audio/eval/<声>/

自分の声を録音しない理由: 声は個人を特定できる情報になりうるので、公開リポジトリと動画には入れない (ユーザーと相談して決定)。
読み上げ音声は何度でも同じものを作り直せる、という利点もある。
"""

import argparse
import subprocess
from pathlib import Path

from language.utterances import UTTERANCES

OUT = Path("assets/audio")
# 術者役。最初は男性の声 (Reed) にしたが、「持針器」がほぼ毎回「リシンピ」「GT」と書き起こされた。
# 5 種類の声で比べると、macOS 標準の日本語音声 Kyoko だけが器械名を全部正しく聞き取れたので、これにした
DEMO_VOICE = "Kyoko"
# 評価用: 声の高さ・話し方が違うもの。読み上げ音声どうしでも、声が変わると認識が崩れるかを見る
EVAL_VOICES = ["Reed (日本語（日本）)", "Eddy (日本語（日本）)", "Grandpa (日本語（日本）)", "Sandy (日本語（日本）)"]


def slug(voice: str) -> str:
    return voice.split(" ")[0].lower()


def synth(text: str, voice: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Whisper が前提とする 16kHz・モノラル・16bit (LEI16 = リトルエンディアンの 16bit 整数) で直接書き出す
    subprocess.run(["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(path), text],
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", choices=["demo", "all"], default="demo")
    args = ap.parse_args()
    jobs = [(DEMO_VOICE, OUT / "demo")]
    if args.voices == "all":
        jobs += [(v, OUT / "eval" / slug(v)) for v in EVAL_VOICES]
    for voice, d in jobs:
        for u in UTTERANCES:
            synth(u.tts or u.text, voice, d / f"{u.id}.wav")
        print(f"saved {d}/ ({len(UTTERANCES)} 個, 声: {voice})")


if __name__ == "__main__":
    main()
