#!/usr/bin/env bash
# Qwisp measurement bench: run the ref set (refs/*.safetensors) across regimes + methods in ONE
# command, reporting per-regime tok/s + quality and an equal-weight aggregate.
#
# Refs are generated (reproducibly) by qwisp/bench_refs.py into <repo>/refs (gitignored):
#   PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
#   PYTHONPATH=<repo> "$PY" -m qwisp.bench_refs <model_dir>
#
# Usage: qwisp/bench.sh [C] [GEN] [throttle_GBs] [methods]
#   C           per-layer cache slots (8GB=64, 16GB=128, 24GB=192).  default 64
#   GEN         tokens to generate per regime.                        default 48
#               (kept short: token-match quality is only meaningful before free-running greedy
#                paths diverge — even lossless strict scores ~62% at GEN=128 due to greedy chaos;
#                at GEN<=48 lossless=100%. Use a larger GEN only for tok/s, not for quality.)
#   throttle    SSD BW GB/s to emulate (0=fast-SSD, 1.5=slow-NAND).   default 0
#   methods     space-list of runners.                                default "suffix-spec bolt"
# Env overrides: QWISP_BENCH_MODEL, QWISP_BENCH_BIN, QWISP_BENCH_REFS.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BENCH_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc}"
MODEL="${QWISP_BENCH_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"
REFS="${QWISP_BENCH_REFS:-$REPO/refs}"
C="${1:-64}"; GEN="${2:-48}"; THR="${3:-0}"; METHODS="${4:-suffix-spec bolt}"
REGIMES="code agentic longctx shortnl"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN (build swift first)"; exit 1; }
[ -d "$REFS" ] || { echo "ERROR: refs dir not found: $REFS (run qwisp.bench_refs first)"; exit 1; }

echo "== Qwisp bench: C=$C GEN=$GEN throttle=${THR}GB/s  ($([ "$THR" = 0 ] && echo fast-SSD || echo slow-NAND)) =="
TMP="$(mktemp)"
for m in $METHODS; do
  for r in $REGIMES; do
    ref="$REFS/$r.safetensors"
    [ -f "$ref" ] || { echo "  (skip $m/$r: missing $ref)"; continue; }
    out="$(QWISP_RUN=$m QWISP_MODEL="$MODEL" QWISP_MTP_REF="$ref" QWISP_CACHE_C="$C" QWISP_GEN="$GEN" \
           QWISP_SWIFT_REF=1 QWISP_SSD_THROTTLE_GBS="$THR" "$BIN" stream 2>/dev/null)"
    tokps="$(printf '%s\n' "$out" | grep -oE '[0-9.]+ tok/s' | head -1 | grep -oE '[0-9.]+')"
    qual="$(printf '%s\n' "$out" | grep -oE '[0-9]+/[0-9]+=[0-9.]+%' | tail -1 | grep -oE '[0-9.]+%$')"
    printf '%s %s %s %s\n' "$m" "$r" "${tokps:-NA}" "${qual:-NA}" >> "$TMP"
    printf '  %-12s %-8s %8s tok/s   quality %s\n' "$m" "$r" "${tokps:-NA}" "${qual:-NA}"
  done
done
echo "-- equal-weight aggregate (mean tok/s over regimes) --"
awk '{ s[$1]+=$3; n[$1]++ } END { for (m in s) printf "  %-12s %.1f tok/s (mean of %d regimes)\n", m, s[m]/n[m], n[m] }' "$TMP"
rm -f "$TMP"
