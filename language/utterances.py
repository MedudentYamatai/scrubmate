"""術者の台詞の一覧 (デモ用 + 評価用)。音声の生成 (make_audio) と評価 (eval_language) で使う。

器械の呼び名は、手術室で実際に使う 5 つ (メス・ハサミ・ペアン・鑷子・持針器) に絞る (ユーザーと相談して決定)。
曖昧な指示は、デモで使う 2 つ (「次」「縫合するのであれ取って」) だけを扱う。

各台詞が持つもの:
  text     … 正しい書き起こし (Whisper の正解)
  tts      … 読み上げ用の文。漢字の専門用語は読み間違えるので、かなで書く (無ければ text をそのまま読む)
  step, with_surgeon … その時点の状況 (今の工程、術者が今持っている器械)
  expect   … 正しい解釈 (intent, instrument)
"""

from dataclasses import dataclass

from sim.instruments import INSTRUMENT_NAMES, JA_NAMES


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    expect: tuple[str, str | None]
    step: str | None = None
    with_surgeon: tuple[str, ...] = ()
    tts: str | None = None
    demo: bool = False  # デモ動画のシナリオで使う台詞


# 読み上げで読み間違える器械名は、かなで読ませる
_TTS_NAME = {"forceps": "せっし", "needle_holder": "じしんき"}


def _say(kind: str) -> str:
    return _TTS_NAME.get(kind, JA_NAMES[kind])


UTTERANCES = [
    # --- デモのシナリオ (claude_code_prompt.md の 1〜8) ---
    Utterance("d1_count", "器械カウントお願いします", ("count", None), tts="きかいカウントおねがいします", demo=True),
    Utterance("d2_mesu", "メス", ("request", "scalpel"), demo=True),
    Utterance("d3_sesshi", "鑷子ちょうだい", ("request", "forceps"), step="皮膚切開", with_surgeon=("scalpel",),
              tts="せっし、ちょうだい", demo=True),
    Utterance("d5_tsugi", "次", ("next", "pean"), step="剥離", with_surgeon=("forceps",), tts="つぎ", demo=True),
    Utterance("d6_are", "縫合するのであれ取って", ("request", "needle_holder"), step="止血",
              with_surgeon=("forceps", "pean"), tts="ほうごうするので、あれとって", demo=True),
    Utterance("d8_heisou", "閉創前カウント", ("count", None), step="縫合", with_surgeon=("needle_holder",),
              tts="へいそうまえカウント", demo=True),
    # --- 5 つの呼び名 × (名前だけ / ちょうだい / 返します) ---
    *[Utterance(f"n_{k}_1", JA_NAMES[k], ("request", k), tts=_say(k)) for k in INSTRUMENT_NAMES],
    *[Utterance(f"n_{k}_2", f"{JA_NAMES[k]}ちょうだい", ("request", k), tts=f"{_say(k)}、ちょうだい")
      for k in INSTRUMENT_NAMES],
    *[Utterance(f"n_{k}_3", f"{JA_NAMES[k]}返します", ("return", k), with_surgeon=(k,), tts=f"{_say(k)}、かえします")
      for k in INSTRUMENT_NAMES],
    # --- 器械と関係ない言葉 (器械名に化けないか) ---
    Utterance("o_thanks", "ありがとう", ("other", None), step="剥離", with_surgeon=("forceps",)),
    Utterance("o_wait", "ちょっと待って", ("other", None), step="止血", with_surgeon=("pean",), tts="ちょっとまって"),
]

# デモの締め (Phase 6 で追加): 未返却の警告の後、術者が持針器を見つけて返し、数え直して一致するまで。
# 評価 (eval_language) の台詞の一覧 (UTTERANCES) には入れない (Phase 4 の評価の数字を変えないため)
DEMO_ENDING = [
    Utterance("d9_found", "持針器ありました、返します", ("return", "needle_holder"), step="縫合",
              with_surgeon=("needle_holder",), tts="じしんき、ありました。かえします", demo=True),
    Utterance("d10_recount", "もう一度カウントお願いします", ("count", None), step="縫合",
              tts="もういちど、カウントおねがいします", demo=True),
]
