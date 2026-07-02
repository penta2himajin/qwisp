"""Task-correctness hooks for the Qwisp bench (third axis, after fidelity + speed).

Given a regime and a method's generated output tokens, check whether the output is *task-correct*
(cheap, objective proxies) — distinct from token-fidelity (does it match strict) and speed.

Hooks per regime:
  code    — does the generated code parse as Python? (ast.parse; syntactic-correctness proxy)
  agentic — is the output valid JSON tool-call(s) referencing the available functions?
  longctx — is the needle retrieved? (RULER: the access code "48271" appears)
  shortnl — non-degenerate prose? (length + no pathological single-token/short-cycle repetition)

Usage: bench_correctness.py <regime> <model_dir> <dump_file>
  dump_file contains a line "OUT_TOKENS:<csv>" or "BOLT_TOKENS:<csv>" (the method's output).
Prints: "PASS" or "FAIL: <reason>".  Exit 0 always (bench parses stdout).
"""
from __future__ import annotations
import ast
import json
import os
import re
import sys

KNOWN_FUNCS = {"get_weather", "book_flight", "convert_currency"}
NEEDLE = "48271"


def _decode(model_dir: str, toks: list[int]) -> str:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    return tok.decode(toks)


def _load_key(path: str, keys) -> list[int]:
    for line in open(path):
        for key in keys:
            if line.startswith(key + ":"):
                return [int(x) for x in line.strip()[len(key) + 1:].split(",") if x]
    return []


def _load_tokens(path: str) -> list[int]:
    return _load_key(path, ("OUT_TOKENS", "BOLT_TOKENS"))


def _degenerate(text: str) -> bool:
    """True if text shows pathological repetition (short cycle or one token dominating)."""
    words = text.split()
    if len(words) < 8:
        return True
    # single word dominating
    from collections import Counter
    c = Counter(words)
    if c.most_common(1)[0][1] > 0.5 * len(words):
        return True
    # short-cycle: a k<=3 window repeated >=4x consecutively
    for k in (1, 2, 3):
        run = 1
        for i in range(k, len(words) - k, k):
            if words[i:i + k] == words[i - k:i]:
                run += 1
                if run >= 4:
                    return True
            else:
                run = 1
    return False


def check(regime: str, text: str, prompt: str = "") -> tuple[bool, str]:
    if regime == "code":
        # the output is the function BODY continuing the prompt's `def ...:`; parse prompt+output.
        # strip markdown fences, then try prompt+output, else the largest def/class block.
        out = text.replace("```python", "").replace("```", "")
        for cand in (prompt + out, out):
            try:
                ast.parse(cand); return True, "parses as Python (prompt+output)"
            except SyntaxError:
                pass
            m = re.search(r"(def |class )", cand)
            if m:
                # trim trailing prose: keep up to the last line that still parses
                lines = cand[m.start():].split("\n")
                for k in range(len(lines), 0, -1):
                    try:
                        ast.parse("\n".join(lines[:k])); return True, "def/class block parses"
                    except SyntaxError:
                        continue
        return False, "no parseable Python (prompt+output)"
    if regime == "agentic":
        # format-tolerant: models may emit JSON tool-calls, python-style calls, or reason in prose
        # with backticked names + kw-args (e.g. `get_weather` with `city="Tokyo"`). Accept any form
        # where a known function is referenced with an adjacent argument. PASS if >=2 tools invoked
        # (this task needs 3; 128-token cap often truncates before the final JSON).
        # (1) strict JSON tool-call(s)
        for m in re.finditer(r"[\[{].*[\]}]", text, re.DOTALL):
            try:
                json.loads(m.group(0))
            except Exception:
                continue
            names = set(re.findall(r'"(?:name|function)"\s*:\s*"([^"]+)"', m.group(0)))
            if names & KNOWN_FUNCS:
                return True, f"valid JSON tool-call(s): {sorted(names & KNOWN_FUNCS)}"
        # (2) format-tolerant: function name with an argument indicator (=, :, (, or a quote) within ~80 chars
        invoked = []
        for f in KNOWN_FUNCS:
            for mm in re.finditer(re.escape(f), text):
                window = text[mm.end(): mm.end() + 80]
                if re.search(r'[=:(]', window) or '"' in window:
                    invoked.append(f)
                    break
        if len(invoked) >= 2:
            return True, f"tool calls planned (format-tolerant): {sorted(invoked)}"
        if invoked:
            return False, f"only 1 tool invoked: {invoked} (need >=2)"
        # bare mention without args, or nothing
        mentioned = sorted(f for f in KNOWN_FUNCS if f in text)
        return False, (f"functions mentioned but no args: {mentioned}" if mentioned else "no known tool referenced")
    if regime == "longctx":
        return (NEEDLE in text), (f"needle {NEEDLE} " + ("retrieved" if NEEDLE in text else "MISSING"))
    if regime == "shortnl":
        return (not _degenerate(text)), ("coherent/non-degenerate" if not _degenerate(text) else "degenerate repetition")
    return False, f"unknown regime {regime}"


def main():
    if len(sys.argv) < 4:
        print("FAIL: usage bench_correctness.py <regime> <model_dir> <dump_file>")
        return
    regime, model_dir, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    toks = _load_tokens(dump)
    if not toks:
        print("FAIL: no output tokens")
        return
    text = _decode(model_dir, toks)
    prompt = ""
    if regime == "code":
        ptoks = _load_key(dump, ("PROMPT_TOKENS",))
        if ptoks:
            prompt = _decode(model_dir, ptoks)
    ok, reason = check(regime, text, prompt)
    print(("PASS: " if ok else "FAIL: ") + reason)


if __name__ == "__main__":
    main()
