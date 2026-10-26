"""音声 (wav) → 文字。mlx-whisper で Mac の中だけで処理する (音声を外部に送らない)。

    uv run python -m language.asr assets/audio/02_mesu.wav

初回はモデル (whisper-large-v3-turbo, 約 1.6GB) を Hugging Face から ~/.cache/huggingface/ にダウンロードする。
"""

import sys
import wave
from pathlib import Path

import numpy as np

from language.vocab import WHISPER_PROMPT

# large-v3 の層を減らして速くした版。small より日本語の専門用語に強く、M5 なら 1 文 1 秒程度
MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000  # Whisper は 16kHz・モノラルの音声を前提に作られている


def load_wav(path: Path) -> np.ndarray:
    """16kHz・モノラル・16bit の wav を、−1〜1 の float32 の配列にする。
    mlx-whisper はファイル名を渡すと ffmpeg で読み込むが、この Mac には ffmpeg が無いので自前で読む。"""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: 16kHz・モノラル・16bit の wav が必要 "
                             f"(今は {w.getframerate()}Hz・{w.getnchannels()}ch・{8 * w.getsampwidth()}bit)")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe(path: Path, use_prompt: bool = False) -> str:
    import mlx_whisper  # 読み込みに数秒かかるので、使うときだけ import する

    r = mlx_whisper.transcribe(
        load_wav(Path(path)),
        path_or_hf_repo=MODEL,
        language="ja",  # 言語の自動判定を省く (短い音声だと判定を誤ることがある)
        initial_prompt=WHISPER_PROMPT if use_prompt else None,
        temperature=0.0,  # 毎回同じ結果にする
        condition_on_previous_text=False,
    )
    return r["text"].strip()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {transcribe(Path(p))}")
