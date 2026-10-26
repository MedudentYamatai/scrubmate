"""文字 → 意図。Gemini に決まった形 (JSON スキーマ) で答えさせる。

    uv run python -m language.interpret --text "次" --step 剥離 --with forceps
    uv run python -m language.interpret --wav assets/audio/demo/d6_are.wav --step 止血 --with forceps pean
    uv run python -m language.interpret --text "次" --step 剥離 --fallback     # ルールベースの予備だけで解釈

無料枠で止まらないための 2 つの仕組み:
  キャッシュ   … (モデル, プロンプト, 文, 状況) が同じなら、前回の答えを language/cache/ から返す。API を呼ばない。
                 キャッシュはリポジトリに含めるので、API キーが無くてもデモ動画を同じ結果で再現できる。
  予備         … 回数制限 (429) や通信エラーのときは、キーワード辞書のルールで解釈する (fallback_interpret)。
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from language.normalize import Fix, correct_names, mentioned
from language.vocab import PROCEDURE, STEP_NAMES
from sim.instruments import INSTRUMENT_NAMES, JA_NAMES

CACHE_DIR = Path("language/cache")
# 応答を待つ上限 [ms]。これを過ぎたら予備に切り替える。
# 時間制限なしだと、サーバーが混んでいるときに 5 分以上止まった (gemini-3.8-flash は 504 で応答なし)
TIMEOUT_MS = 20000

IntentName = Literal["request", "return", "next", "count", "other"]
InstrumentName = Literal["scalpel", "needle_holder", "forceps", "scissors", "pean"]
StepName = Literal["皮膚切開", "剥離", "止血", "縫合"]


class Intent(BaseModel):
    """LLM に返させる JSON の形。Gemini にはこのクラスから作ったスキーマを渡し、この形以外で答えられないようにする。"""

    intent: IntentName = Field(description="request=器械を求める / return=器械を返す・不要になった / "
                                           "next=工程を次に進める / count=器械カウントの依頼 / other=それ以外")
    instrument: InstrumentName | None = Field(description="対象の器械。該当しなければ null")
    step: StepName | None = Field(description="この指示の後の工程。開始前なら null")
    reason: str = Field(description="そう判断した理由 (日本語で短く)")


@dataclass
class Context:
    """解釈に必要な、その時点の状況。"""

    step: str | None = None  # 今の工程 (None = 開始前)
    with_surgeon: tuple[str, ...] = ()  # 術者が今持っている器械

    def to_prompt(self) -> dict:
        return {"現在の工程": self.step or "開始前",
                "術者が持っている器械": [f"{k} ({JA_NAMES[k]})" for k in self.with_surgeon] or "なし"}


@dataclass
class Result:
    intent: Intent
    source: Literal["cache", "gemini", "fallback"]  # 誰が解釈したか (動画・ログで表示する)
    seconds: float = 0.0
    error: str | None = None  # 予備に切り替えた理由
    heard: str = ""  # 解釈する前の文 (書き起こしそのまま)
    text: str = ""  # 読みで器械名・工程名を補正した後の文 (実際に解釈した文)
    fixes: list[Fix] = field(default_factory=list)  # 補正の内容


def _system_prompt() -> str:
    inst = "\n".join(f"- {k}: {JA_NAMES[k]}" for k in INSTRUMENT_NAMES)
    proc = "\n".join(f"{i + 1}. {s}: " + "、".join(f"{k} ({JA_NAMES[k]})" for k in ks)
                     for i, (s, ks) in enumerate(PROCEDURE))
    return f"""あなたは手術室の器械出し看護師ロボットの指示解釈係です。
術者の発話 (音声認識の結果なので誤字を含むことがある) と現在の状況から、術者の意図を JSON で答えてください。

扱う器械は次の 5 種類だけで、この呼び名で呼ばれます。
{inst}

術式の工程と、各工程で使う器械 (先頭が最も典型的):
{proc}

判断のルール:
- request: 器械を求めている。器械名が無く「縫合するのであれ」のように工程の言葉と指示語だけなら、
  その工程の器械のうち術者がまだ持っていないものを選ぶ。
- next: 「次」のように工程を進める指示。現在の工程の次の工程に進み、その工程の器械のうち術者がまだ持っていない最も典型的なものを instrument にする。
- return: 器械を返す・もう要らない。器械名が無ければ、術者が持っている器械から推定する。
- count: 器械カウントの依頼 (開始時のカウント、閉創前カウントなど)。instrument は null。
- other: それ以外 (お礼、待機の指示、5 種類以外の物の要求など)。5 種類以外の物を、無理にどれかの器械に当てはめない。
- step: この指示の後の工程。request なら求められた器械を使う工程、next なら進んだ後の工程、それ以外は現在の工程のまま。
"""


SYSTEM_PROMPT = _system_prompt()


def fallback_interpret(text: str, ctx: Context) -> Intent:
    """API が使えないときの予備: キーワードによるルール。text は読みで補正した後の文 (normalize.correct_names)。
    5 つの器械名と「次」・工程名が入った指示なら解釈できる。"""
    named = mentioned(text)
    cur = STEP_NAMES.index(ctx.step) if ctx.step in STEP_NAMES else -1

    def pick(step_idx: int) -> str:
        cands = PROCEDURE[step_idx][1]
        return next((k for k in cands if k not in ctx.with_surgeon), cands[0])

    if "カウント" in text:
        return Intent(intent="count", instrument=None, step=ctx.step, reason="「カウント」を含む")
    if any(w in text for w in ("返", "戻", "いらない", "要らない")):
        inst = named[0] if named else (ctx.with_surgeon[0] if len(ctx.with_surgeon) == 1 else None)
        return Intent(intent="return", instrument=inst, step=ctx.step, reason="返却の言葉を含む")
    if named:
        step = next((s for s, ks in PROCEDURE if named[0] in ks), ctx.step)
        return Intent(intent="request", instrument=named[0], step=step, reason=f"器械名「{JA_NAMES[named[0]]}」を含む")
    if text.startswith("次") and cur + 1 < len(PROCEDURE):
        return Intent(intent="next", instrument=pick(cur + 1), step=STEP_NAMES[cur + 1], reason="「次」→ 工程表の次")
    for i, step in enumerate(STEP_NAMES):
        if step in text:
            return Intent(intent="request", instrument=pick(i), step=step, reason=f"工程名「{step}」を含む")
    return Intent(intent="other", instrument=None, step=ctx.step, reason="該当するキーワードなし")


class Interpreter:
    def __init__(self, use_llm: bool = True):
        from dotenv import load_dotenv

        load_dotenv()
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.api_key = os.environ.get("GEMINI_API_KEY") or None
        self.use_llm = use_llm
        self._client = None

    def _cache_path(self, text: str, ctx: Context) -> Path:
        key = json.dumps({"model": self.model, "system": SYSTEM_PROMPT, "text": text, "context": ctx.to_prompt()},
                         ensure_ascii=False, sort_keys=True)
        return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"

    def interpret(self, heard: str, ctx: Context) -> Result:
        """heard = 書き起こし (またはデバッグ用に打ち込んだ文)。先に器械名・工程名を読みで補正してから解釈する。"""
        text, fixes = correct_names(heard)
        r = self._interpret(text, ctx)
        r.heard, r.text, r.fixes = heard, text, fixes
        return r

    def _interpret(self, text: str, ctx: Context) -> Result:
        t0 = time.time()
        if not self.use_llm:
            return Result(fallback_interpret(text, ctx), "fallback", time.time() - t0, error="予備のみのモード")
        path = self._cache_path(text, ctx)
        if path.exists():
            return Result(Intent(**json.loads(path.read_text())["intent"]), "cache", time.time() - t0)
        if not self.api_key:
            return Result(fallback_interpret(text, ctx), "fallback", time.time() - t0,
                          error="GEMINI_API_KEY が .env に無い")
        try:
            intent = self._ask_gemini(text, ctx)
        except Exception as e:  # 429 (回数制限)・通信エラー・形式エラーなど、何であれ止まらずに予備へ
            msg = f"{type(e).__name__}: {e}"
            return Result(fallback_interpret(text, ctx), "fallback", time.time() - t0, error=msg[:800])  # 429 のときは末尾に「何秒後に再試行」が入っているので長めに残す
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "text": text, "context": ctx.to_prompt(),
                                    "intent": intent.model_dump()}, ensure_ascii=False, indent=2))
        return Result(intent, "gemini", time.time() - t0)

    def _ask_gemini(self, text: str, ctx: Context) -> Intent:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        user = json.dumps({"状況": ctx.to_prompt(), "術者の発話": text}, ensure_ascii=False)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=Intent,  # structured output: この形の JSON しか返せない
                temperature=0.0,
            ),
        )
        if resp.parsed is None:
            raise ValueError(f"JSON として読めない応答: {resp.text!r}")
        return resp.parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="文字で指示を入れる (デバッグ用)")
    g.add_argument("--wav", type=Path, help="音声ファイル (Whisper で書き起こしてから解釈)")
    ap.add_argument("--step", choices=STEP_NAMES, help="今の工程 (省略 = 開始前)")
    ap.add_argument("--with", dest="with_surgeon", nargs="*", default=[], choices=INSTRUMENT_NAMES,
                    help="術者が今持っている器械")
    ap.add_argument("--fallback", action="store_true", help="LLM を使わず、ルールベースの予備だけで解釈")
    args = ap.parse_args()

    text = args.text
    if args.wav:
        from language.asr import transcribe

        text = transcribe(args.wav)
        print(f"書き起こし: {text}")
    r = Interpreter(use_llm=not args.fallback).interpret(text, Context(args.step, tuple(args.with_surgeon)))
    if r.fixes:
        print(f"読みで補正: {r.text}  (" + ", ".join(f"「{f.heard}」→{f.name}" for f in r.fixes) + ")")
    print(f"解釈 ({r.source}, {r.seconds:.2f}s): {r.intent.model_dump_json()}")
    if r.error:
        print(f"  予備に切り替えた理由: {r.error}")


if __name__ == "__main__":
    main()
