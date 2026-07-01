#!/usr/bin/env bash
# Qwisp measurement bench: run the ref set (refs/*.safetensors) across regimes + methods and
# report 3 axes per cell in one command:
#   speed       free-running tok/s (production runner: suffix-spec / bolt)
#   fidelity    teacher-forced per-token argmax agreement vs reference greedy (chaos-free)
#   correctness task-level proxy on the free-run output (regime hooks: parse/json/needle/degen)
# Faithful: invokes the REAL production runners + the teacher-forced mlx-fidelity path (no
# re-implemented decode). Per cell = 2 model loads (speed + fidelity); correctness reuses the
# speed run's dumped output. (An efficiency variant folding all axes into one load would need a
# shared-decode refactor; deferred for faithfulness.)
#
# Refs are generated (reproducibly) by qwisp/bench_refs.py into <repo>/refs (gitignored).
# Usage: qwisp/bench.sh [C] [GEN] [throttle_GBs] [methods]
#   C=64 GEN=128 throttle=0 methods="suffix-spec bolt"  (defaults)
# Env: QWISP_BENCH_MODEL, QWISP_BENCH_BIN, QWISP_BENCH_REFS, QWISP_BENCH_PY.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BENCH_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc}"
MODEL="${QWISP_BENCH_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"
REFS="${QWISP_BENCH_REFS:-$REPO/refs}"
PY="${QWISP_BENCH_PY:-$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3}"
C="${1:-64}"; GEN="${2:-128}"; THR="${3:-0}"; METHODS="${4:-suffix-spec bolt}"
REGIMES="code agentic longctx shortnl"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN"; exit 1; }
[ -d "$REFS" ] || { echo "ERROR: refs dir not found: $REFS (run qwisp.bench_refs)"; exit 1; }

echo "== Qwisp bench: C=$C GEN=$GEN throttle=${THR}GB/s ($([ "$THR" = 0 ] && echo fast-SSD || echo slow-NAND)) =="
printf '  %-12s %-8s %10s %12s  %s\n' method regime "tok/s" "fidelity" "correctness"
TMP="$(mktemp)"
for m in $METHODS; do
  skip=0; [ "$m" = bolt ] && skip=3          # mlx-fidelity buddy toggle
  for r in $REGIMES; do
    ref="$REFS/$r.safetensors"
    [ -f "$ref" ] || { echo "  (skip $m/$r: missing $ref)"; continue; }
    # --- speed (free-run) + output dump for correctness ---
    dump="$(mktemp)"
    out="$(QWISP_RUN=$m QWISP_MODEL="$MODEL" QWISP_MTP_REF="$ref" QWISP_CACHE_C="$C" QWISP_GEN="$GEN" \
           QWISP_SSD_THROTTLE_GBS="$THR" QWISP_DUMP_TOKENS=1 "$BIN" stream 2>/dev/null)"
    printf '%s\n' "$out" | grep -E 'PROMPT_TOKENS|OUT_TOKENS|BOLT_TOKENS' > "$dump"
    tokps="$(printf '%s\n' "$out" | grep -oE '[0-9.]+ tok/s' | head -1 | grep -oE '[0-9.]+')"
    # --- correctness (regime hook on the free-run output) ---
    corr="$("$PY" "$REPO/qwisp/bench_correctness.py" "$r" "$MODEL" "$dump" 2>/dev/null)"
    [ -z "$corr" ] && corr="(checker error)"
    rm -f "$dump"
    # --- fidelity (teacher-forced, chaos-free; throttle-independent so run unthrottled) ---
    fout="$(QWISP_RUN=mlx-fidelity QWISP_MODEL="$MODEL" QWISP_MTP_REF="$ref" QWISP_CACHE_C="$C" \
            QWISP_GEN="$GEN" QWISP_SKIPMODE="$skip" "$BIN" stream 2>/dev/null)"
    fid="$(printf '%s\n' "$fout" | grep -oE 'fidelity vs gR: [0-9]+/[0-9]+=[0-9.]+%' | grep -oE '[0-9.]+%$')"
    printf '  %-12s %-8s %9s  %11s  %s\n' "$m" "$r" "${tokps:-NA}" "${fid:-NA}" "$corr"
    printf '%s %s %s %s\n' "$m" "$r" "${tokps:-NA}" "${fid:-NA%}" >> "$TMP"
  done
done
echo "-- equal-weight aggregate (mean over regimes) --"
awk '{ t[$1]+=$3; f[$1]+=($4+0); n[$1]++ }
     END { for (m in t) printf "  %-12s speed %.1f tok/s   fidelity %.1f%%  (mean of %d regimes)\n", m, t[m]/n[m], f[m]/n[m], n[m] }' "$TMP"
rm -f "$TMP"
echo "-- axes: fidelity=teacher-forced per-token vs strict (primary, chaos-free). correctness=L3"
echo "   acceptability, read as DELTA: strict PASS -> bolt PASS means the fidelity divergence is"
echo "   benign for that task. absolute correctness is bounded by base-model capability (if strict"
echo "   FAILs a regime, that hook is base-model-limited, not an engine signal). --"
