"""書き起こしの器械名の誤りを、読み (ひらがな) の近さで直す。

    uv run python -m language.normalize "接し、ちょうだい。" "スペアアンください。"

Whisper は器械名を、読みは合っているが別の漢字にしてしまうことが多い (鑷子 →「接し」、持針器 →「自信機」)。
そこで、書き起こしを読み (ひらがな) に直し、5 つの器械名と 4 つの工程名の読みと一番似ている部分を探す。
似ている度合いが THRESHOLD 以上なら、その部分を正しい器械名に置き換える。
"""

import difflib
import sys
import unicodedata
from dataclasses import dataclass

import pykakasi

from language.vocab import READINGS, STEP_READINGS
from sim.instruments import INSTRUMENT_NAMES, JA_NAMES

# 似ている度合い (0〜1) がこれ以上なら同じ器械名とみなす。
# Reed の声の書き起こしだけを見て決め、他の 4 種類の声で確かめた (eval_language)。
# 器械名を含まない台詞は最大 0.5 だった (「カウント」の「かうん」が「せんとう」に 2 文字似る、など)
THRESHOLD = 0.65

_kks = pykakasi.kakasi()


# 直す対象: 器械名 5 つ + 工程名 4 つ。(表示名, 読み)
TARGETS: dict[str, str] = {JA_NAMES[k]: r for k, r in READINGS.items()} | {s: r for s, r in STEP_READINGS.items()}


@dataclass
class Fix:
    name: str  # 直した後の名前 (器械名か工程名)
    heard: str  # 書き起こしの元の部分
    score: float


def _seion(s: str) -> str:
    """濁点・半濁点を外す (ご → こ、ぺ → へ)。音声認識は濁る・濁らないをよく取り違える (縫合 →「報告」)。
    NFD (分解) すると「ご」は「こ」+「゛」に分かれるので、「゛」「゜」を捨てる。"""
    return "".join(c for c in unicodedata.normalize("NFD", s) if c not in "\u3099\u309a")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _seion(a), _seion(b)).ratio()


def correct_names(text: str) -> tuple[str, list[Fix]]:
    """器械名・工程名らしき部分を正しい名前に置き換えた文と、置き換えの一覧を返す。"""
    tokens = _kks.convert(text)  # 単語ごとに {orig: 元の表記, hira: 読み}
    hira, owner = "", []  # 読みを 1 本の文字列につなげ、各文字がどの単語から来たかを覚えておく
    for i, t in enumerate(tokens):
        hira += t["hira"]
        owner += [i] * len(t["hira"])

    # 名前ごとに、読みと同じ長さの窓をずらしながら一番似ている所を探す
    found: list[tuple[float, str, int, int]] = []
    for name, r in TARGETS.items():
        best = (0.0, -1)
        for s in range(max(1, len(hira) - len(r) + 1)):
            sc = _similarity(hira[s:s + len(r)], r)
            if sc > best[0]:
                best = (sc, s)
        sc, s = best
        if sc >= THRESHOLD:
            found.append((sc, name, owner[s], owner[min(s + len(r), len(hira)) - 1]))

    # 似ている順に採用する。同じ単語を 2 つの名前が取り合ったら、似ている方を採る
    replaced: dict[int, str] = {}
    used: set[int] = set()
    fixes: list[Fix] = []
    for sc, name, t0, t1 in sorted(found, reverse=True):
        span = set(range(t0, t1 + 1))
        if span & used:
            continue
        used |= span
        heard = "".join(tokens[i]["orig"] for i in sorted(span))
        replaced[t0] = name
        for i in sorted(span)[1:]:
            replaced[i] = ""
        if heard != name:
            fixes.append(Fix(name, heard, round(sc, 2)))
    out = "".join(replaced.get(i, t["orig"]) for i, t in enumerate(tokens))
    return out, fixes


def mentioned(text: str) -> list[str]:
    """文 (補正後) に出てくる器械を、出てきた順に返す。"""
    pos = sorted((text.find(JA_NAMES[k]), k) for k in INSTRUMENT_NAMES if JA_NAMES[k] in text)
    return [k for _, k in pos]


if __name__ == "__main__":
    for s in sys.argv[1:]:
        out, fixes = correct_names(s)
        print(f"{s} → {out}  " + ", ".join(f"「{f.heard}」→{f.name} ({f.score})" for f in fixes))
