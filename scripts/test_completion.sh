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
echo "RESULT: PASS ($line)"
