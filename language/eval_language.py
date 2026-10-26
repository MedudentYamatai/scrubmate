"""Phase 4: 音声認識 (Whisper)・読みでの補正・指示解釈 (Gemini / ルールベースの予備) を、台詞の一覧で評価する。

    uv run python -m language.make_audio --voices all     # 先に音声を作る (5 種類の声)
    uv run python -m language.eval_language               # 全部 (Gemini はデモ用の声の書き起こしだけ。23 回呼ぶ)
    uv run python -m language.eval_language --no-llm      # Gemini を呼ばない

見るもの:
  名前の聞き取り … 台詞に入っている器械名・工程名が、書き起こしに正しく入っていた割合。
                   「補正前」= Whisper のまま / 「補正後」= 読みで直した後 (normalize.py)
  誤補正         … 器械名の入っていない台詞が、補正で器械名に化けた回数 (0 であるべき)
  解釈の正解率   … intent と instrument の両方が正解と一致した割合
  補正のしきい値は Reed の声だけを見て決めた。他の 4 種類の声は、決めた後の確認用 (未知のデータ)

出力: outputs/phase4/language_eval.json (台詞ごとの詳細)
書き起こしは outputs/phase4/asr_<声>.json に保存し、2 回目以降は再利用する (Whisper の実行に数分かかるため)
"""

import argparse
import json
import re
import time
from pathlib import Path

from language.interpret import Context, Interpreter, fallback_interpret
from language.make_audio import DEMO_VOICE, EVAL_VOICES, OUT as AUDIO_DIR, slug
from language.normalize import TARGETS, correct_names
from language.utterances import UTTERANCES

OUT_DIR = Path("outputs/phase4")


def normalize(s: str) -> str:
    return re.sub(r"[、。，．,.!?！？\s]", "", s)


def cer(ref: str, hyp: str) -> float:
    """文字誤り率 = 編集距離 (挿入・削除・置換の最少回数) ÷ 正解の文字数。"""
    a, b = normalize(ref), normalize(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def names_in(s: str) -> set[str]:
    return {n for n in TARGETS if n in s}


def transcripts(voice_dir: Path, name: str) -> dict[str, str]:
    cache = OUT_DIR / f"asr_{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from language.asr import transcribe

    out = {u.id: transcribe(voice_dir / f"{u.id}.wav") for u in UTTERANCES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


class DailyLimit(Exception):
    pass


def ask(interp: Interpreter, text: str, ctx: Context, pace: float):
    """Gemini で解釈する。予備に切り替わった回は None (Gemini の成績に数えない)。"""
    r = interp.interpret(text, ctx)
    if r.error and "PerDay" in r.error:
        # 1 日あたりの上限 (無料枠はモデルごとに 1 日 20 回など)。待っても戻らないので、Gemini の評価はここで打ち切る。
        # (最初はこれを 1 分あたりの上限と区別せず、60 秒待っては再試行を繰り返していた)
        raise DailyLimit(r.error)
    if r.error and "429" in r.error:
        # 無料枠は 1 分あたり 5 回まで。評価では、サーバーが指定する時間だけ待って 1 回だけやり直す
        # (デモ本番では待たずに予備へ切り替える。止まらないことを優先するため)
        wait = float(m.group(1)) + 1 if (m := re.search(r"retry in ([0-9.]+)s", r.error)) else 60
        print(f"  429 (回数制限) → {wait:.0f} 秒待ってやり直す", flush=True)
        time.sleep(wait)
        r = interp.interpret(text, ctx)
    if r.source == "gemini":
        time.sleep(pace)
    return [r.intent.intent, r.intent.instrument] if r.source != "fallback" else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Gemini を呼ばない")
    ap.add_argument("--pace", type=float, default=13.0, help="Gemini を呼ぶ間隔 [秒] (無料枠は 1 分 5 回)")
    args = ap.parse_args()

    demo = slug(DEMO_VOICE)
    voices = {demo: AUDIO_DIR / "demo"} | {slug(v): AUDIO_DIR / "eval" / slug(v) for v in EVAL_VOICES}
    asr = {name: transcripts(d, name) for name, d in voices.items()}
    interp = None if args.no_llm else Interpreter()

    rows = []
    for u in UTTERANCES:
        ctx = Context(u.step, u.with_surgeon)
        want = names_in(u.text)
        row = {"id": u.id, "text": u.text, "expect": list(u.expect), "voices": {}}
        for v in voices:
            heard = asr[v][u.id]
            fixed, _ = correct_names(heard)
            fb = fallback_interpret(fixed, ctx)
            row["voices"][v] = {
                "heard": heard, "fixed": fixed, "cer": round(cer(u.text, heard), 3),
                "names_before": want <= names_in(heard), "names_after": want <= names_in(fixed),
                "false_fix": bool(names_in(fixed) - want - names_in(heard)),
                "fallback": [fb.intent, fb.instrument],
            }
        row["fallback_clean"] = list(fallback_interpret(u.text, ctx).model_dump(include={"intent", "instrument"}).values())
        # Gemini は、実際の流れと同じ「デモ用の声の書き起こし」だけで評価する (無料枠の回数を節約するため)。
        # 台詞の一覧はデモの台詞が先頭なので、上限に当たってもデモの分は先にキャッシュされる
        row["gemini_" + demo] = None
        if interp:
            try:
                row["gemini_" + demo] = ask(interp, asr[demo][u.id], ctx, args.pace)
            except DailyLimit as e:
                print(f"  1 日の上限に達したので Gemini の評価を打ち切る ({u.id} から先は未評価): {str(e)[:80]}")
                interp = None
        rows.append(row)

    def pct(xs) -> str:
        xs = list(xs)
        return f"{sum(xs):2d}/{len(xs):2d}"

    named = [r for r in rows if names_in(r["text"])]
    print(f"\n■ 音声認識と補正 (器械名・工程名が入った台詞 {len(named)} 個)")
    print(f"  {'声':10s} 文字誤り率  名前の聞き取り 補正前 → 補正後   誤補正   予備での解釈")
    for v in voices:
        tag = "(補正のしきい値を決めた声)" if v == "reed" else ("(デモ用の声)" if v == demo else "")
        c = sum(r["voices"][v]["cer"] for r in rows) / len(rows)
        print(f"  {v:10s} {100 * c:6.1f}%      {pct(r['voices'][v]['names_before'] for r in named)} → "
              f"{pct(r['voices'][v]['names_after'] for r in named)}     "
              f"{sum(r['voices'][v]['false_fix'] for r in rows)}      "
              f"{pct(r['voices'][v]['fallback'] == r['expect'] for r in rows)}  {tag}")

    print(f"\n■ 解釈の正解率 (全 {len(rows)} 台詞)")
    print(f"  予備 / 正しい文             {pct(r['fallback_clean'] == r['expect'] for r in rows)}")
    key = "gemini_" + demo
    rs = [r for r in rows if r[key] is not None]
    if rs:
        print(f"  {'Gemini / 書き起こし (' + demo + ')':24s} {pct(r[key] == r['expect'] for r in rs)}"
              + (f"  (未評価・エラー {len(rows) - len(rs)} 回)" if len(rs) < len(rows) else ""))

    print(f"\n■ 間違えた台詞 (声: {demo})")
    for r in rows:
        d = r["voices"][demo]
        wrong = {k: r[k] for k in ("fallback_clean", "gemini_" + demo) if r.get(k) not in (None, r["expect"])}
        if d["fallback"] != r["expect"]:
            wrong[f"予備/{demo}"] = d["fallback"]
        if wrong or not d["names_after"]:
            print(f"  {r['text']}  →  書き起こし「{d['heard']}」→ 補正「{d['fixed']}」")
            for k, val in wrong.items():
                print(f"      {k}: {val}  (正解 {r['expect']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "language_eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nsaved {OUT_DIR}/language_eval.json")


if __name__ == "__main__":
    main()
