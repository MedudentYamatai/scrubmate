"""音声・言語処理で共有する知識: 器械の呼び名と、術式の工程。

器械の呼び名は、手術室で実際に使う 5 つ (メス・ハサミ・ペアン・鑷子・持針器) に絞る (ユーザーと相談して決定)。
言い換え (クーパー・コッヘルなど) や、一般的な工程予測には手を広げない。
"""

from sim.instruments import INSTRUMENT_NAMES, JA_NAMES

# 呼び名の読み (ひらがな)。書き起こしの誤り (鑷子 →「接し」) を、読みの近さで直すのに使う (normalize.py)
READINGS: dict[str, str] = {
    "scalpel": "めす",
    "needle_holder": "じしんき",
    "forceps": "せっし",
    "scissors": "はさみ",
    "pean": "ぺあん",
}
assert set(READINGS) == set(INSTRUMENT_NAMES)

# 術式の工程と、それぞれの工程で主に使う器械 (先頭が最も典型的なもの)。
# デモの「次」(剥離 → 止血 → ペアン) と「縫合するのであれ取って」(→ 持針器) を解くために使う
PROCEDURE: list[tuple[str, list[str]]] = [
    ("皮膚切開", ["scalpel"]),
    ("剥離", ["forceps", "scissors"]),
    ("止血", ["pean"]),
    ("縫合", ["needle_holder", "forceps"]),
]
STEP_NAMES = [s for s, _ in PROCEDURE]
# 工程名の読み。デモの「縫合するので」が「報告するので」と書き起こされたので、器械名と同じく読みで直す
STEP_READINGS: dict[str, str] = {"皮膚切開": "ひふせっかい", "剥離": "はくり", "止血": "しけつ", "縫合": "ほうごう"}
assert set(STEP_READINGS) == set(STEP_NAMES)

# Whisper に渡すヒント (initial_prompt)。「この会話にはこういう単語が出てくる」と先に見せる。
# ただし評価では、ヒントの有無で文字誤り率はほぼ変わらなかった (27.8% / 28.8%)。主な対策は読みでの補正 (normalize.py)
WHISPER_PROMPT = "手術室での器械出しの会話。" + "、".join(JA_NAMES[k] for k in INSTRUMENT_NAMES) + "、器械カウント。"
