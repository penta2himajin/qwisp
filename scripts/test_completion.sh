#!/usr/bin/env bash
# Tokenizer self-test gate (step 5b): text↔ids round-trip + Qwen chat_template.
# GPU-free but needs the model's tokenizer files. Exits 0 only on COMPTEST X/X.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp}"
MODEL="${QWISP_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"

[ -x "$BIN" ] || { echo "ERROR: qwisp binary not found: $BIN (build the qwisp scheme first)"; exit 1; }
[ -d "$MODEL" ] || { echo "SKIP: model not found at $MODEL"; exit 0; }

echo "=== Qwisp completion-core self-test ==="
out="$(QWISP_MODEL="$MODEL" "$BIN" comptest 2>&1)"
echo "$out"
echo "==="

line="$(printf '%s\n' "$out" | grep '^COMPTEST')"
if [ -z "$line" ]; then echo "RESULT: FAIL (no COMPTEST line — crash?)"; exit 1; fi
passed="$(printf '%s\n' "$line" | sed 's|COMPTEST ||' | cut -d/ -f1)"
total="$(printf '%s\n' "$line" | sed 's|COMPTEST ||' | cut -d/ -f2)"
if printf '%s\n' "$out" | grep -q 'FAIL' || [ "$passed" != "$total" ]; then
    echo "RESULT: FAIL ($line)"; exit 1
fi
# ── #166: piped stdin must be read to EOF, not truncated to the first line ──────────
# QWISP_FAKE=1 uses FakeBackend: no model weights, no GPU — only the tokenizer this gate
# already requires. Asserts a strict increase rather than an exact count so it does not
# encode the tokenizer's segmentation. Before the fix both arms reported 11 tok.
tok_of() {
    printf '%b' "$1" | QWISP_FAKE=1 QWISP_MODEL="$MODEL" "$BIN" chat --max-tokens 2 2>&1 \
        | sed -n 's/.*prompt \([0-9]*\) tok.*/\1/p'
}
one="$(tok_of 'alpha\n')"
two="$(tok_of 'alpha\nbeta gamma delta epsilon zeta eta theta iota kappa lambda\n')"
if [ -z "$one" ] || [ -z "$two" ] || [ "$two" -le "$one" ]; then
    echo "RESULT: FAIL (piped stdin truncated to line 1: 1-line=${one:-NA} tok, 2-line=${two:-NA} tok — #166)"
    exit 1
fi
echo "[stdin] piped multi-line read to EOF: 1-line=$one tok < 2-line=$two tok  ok"

echo "RESULT: PASS ($line)"
