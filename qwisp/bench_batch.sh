#!/usr/bin/env bash
# Qwisp measurement bench — SINGLE-PROCESS batch variant (T1). Same CLI + same human-readable
# output table/footer as qwisp/bench.sh, but ONE binary invocation total for non-bolt methods:
# the in-process batch runner (QWISP_RUN=bench-batch, swift TellBench.swift) loops methods ×
# regimes with a single model load (vs 8-12 loads for bench.sh), resetting all engine state per
# cell (cold-start equivalent; OS page-cache warmth is identical to the multi-process bench).
# bolt cells are measured with per-regime single-shot raw-spec CLI (QWISP_RUN=raw-spec,
# RawSpecRunner.swift) — the shipping engine, not the MLX boltCore path.
# Post-processing of the runner's stdout per cell:
#   speed        non-bolt: BENCH|method=..|regime=..|tokps=..
#                bolt: [RawSpec] bolt(L3 near-lossless) C=<C>: <X> tok/s ...
#   fidelity     suffix-spec: T0 token compare (bench_tokcmp.py on OUT_TOKENS vs canonical ref);
#                bolt: [RawSpec] bolt TF fidelity vs strict-canonical: <a>/<b>=<Z>% (chaos-free)
#   correctness  bench_correctness.py on the cell's dumped PROMPT/OUT/BOLT token lines
#
# Refs are generated (reproducibly) by qwisp/bench_refs.py into <repo>/refs (gitignored).
# Usage: qwisp/bench_batch.sh [C] [GEN] [throttle_GBs] [methods]
#   C=64 GEN=128 throttle=0 methods="suffix-spec bolt"  (defaults)
# Env: QWISP_BENCH_MODEL, QWISP_BENCH_BIN, QWISP_BENCH_REFS, QWISP_BENCH_PY.
#      QWISP_THROTTLE_DEFER=1 (T2) defers the SSD throttle to decode start (~2x faster slow-NAND
#      runs). NOTE: QWISP_THROTTLE_DEFER is intentionally NOT forwarded to raw-spec bolt calls
#      (raw ignores the defer gate -> DEFER would silently measure unthrottled; raw cells always
#      run fully throttled from process start).
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

# --- split methods into non-bolt (in-process batch) and bolt (per-regime raw-spec CLI) ---
NON_BOLT=""; HAS_BOLT=0
for _m in $METHODS; do
  if [ "$_m" = bolt ]; then HAS_BOLT=1; else NON_BOLT="${NON_BOLT:+$NON_BOLT }$_m"; fi
done

# --- single batch invocation for non-bolt methods (1 model load; skip if none) ---
RAW="$(mktemp)"
if [ -n "$NON_BOLT" ]; then
  QWISP_RUN=bench-batch QWISP_MODEL="$MODEL" QWISP_BENCH_REFS_DIR="$REFS" \
    QWISP_BENCH_METHODS="$NON_BOLT" QWISP_BENCH_REGIMES="$REGIMES" \
    QWISP_CACHE_C="$C" QWISP_GEN="$GEN" QWISP_SSD_THROTTLE_GBS="$THR" \
    QWISP_DUMP_TOKENS=1 "$BIN" stream > "$RAW" 2>/dev/null
fi

TMP="$(mktemp)"
for m in $METHODS; do
  for r in $REGIMES; do
    ref="$REFS/$r.safetensors"
    [ -f "$ref" ] || { echo "  (skip $m/$r: missing $ref)"; continue; }

    if [ "$m" = bolt ]; then
      # --- per-regime single-shot raw-spec CLI ---
      # ponytail: env -u strips QWISP_THROTTLE_DEFER; raw measurement is invalid with DEFER set.
      raw_out="$(env -u QWISP_THROTTLE_DEFER \
        QWISP_RUN=raw-spec QWISP_MODEL="$MODEL" QWISP_RAW_C="$C" QWISP_RAW_BOLT=1 \
        QWISP_MTP_REF="$ref" QWISP_GEN="$GEN" QWISP_SSD_THROTTLE_GBS="$THR" \
        QWISP_DUMP_TOKENS=1 QWISP_RAW_TF=1 QWISP_BOLT_WORKLOAD="$r" \
        "$BIN" stream 2>/dev/null)"
      dump="$(mktemp)"
      printf '%s\n' "$raw_out" | grep -E 'PROMPT_TOKENS|OUT_TOKENS|BOLT_TOKENS' > "$dump"
      # speed: "[RawSpec] bolt(L3 near-lossless) C=<C>: <X> tok/s ..."  (RawSpecRunner.swift:1098)
      tokps="$(printf '%s\n' "$raw_out" \
        | grep '\[RawSpec\] bolt(L3 near-lossless)' | head -1 \
        | sed 's/.*C=[^:]*: *\([0-9.]*\) tok\/s.*/\1/')"
      # fidelity: "[RawSpec] bolt TF fidelity vs strict-canonical: a/b=Z% (chaos-free)"  (:1316)
      fid="$(printf '%s\n' "$raw_out" \
        | grep '\[RawSpec\] bolt TF fidelity' | head -1 \
        | grep -oE '[0-9.]+%' | tail -1)"
      corr="$("$PY" "$REPO/qwisp/bench_correctness.py" "$r" "$MODEL" "$dump" 2>/dev/null)"
      [ -z "$corr" ] && corr="(checker error)"
      rm -f "$dump"
    else
      # --- in-process batch cell (from $RAW) ---
      cell="$(awk -v tag="BENCHCELL|method=$m|regime=$r" \
              '$0 == tag { on=1; next } on && index($0, "BENCHCELL|") == 1 { exit } on { print }' "$RAW")"
      [ -n "$cell" ] || { echo "  (skip $m/$r: no cell output)"; continue; }
      dump="$(mktemp)"
      printf '%s\n' "$cell" | grep -E 'PROMPT_TOKENS|OUT_TOKENS|BOLT_TOKENS' > "$dump"
      tokps="$(printf '%s\n' "$cell" | grep '^BENCH|' | head -1 | sed 's/.*tokps=//')"
      corr="$("$PY" "$REPO/qwisp/bench_correctness.py" "$r" "$MODEL" "$dump" 2>/dev/null)"
      [ -z "$corr" ] && corr="(checker error)"
      # T0: strict fidelity == free-run tokens vs canonical ref (CPU compare, no 2nd model load;
      # stronger than TF: bit-verifies the actual trajectory). bolt uses TF (free-run token-match
      # is greedy-chaos, not the fidelity axis).
      fid="$("$PY" "$REPO/qwisp/bench_tokcmp.py" "$ref" "$dump" 2>/dev/null | grep -oE '[0-9.]+%$')"
      rm -f "$dump"
    fi

    printf '  %-12s %-8s %9s  %11s  %s\n' "$m" "$r" "${tokps:-NA}" "${fid:-NA}" "$corr"
    printf '%s %s %s %s\n' "$m" "$r" "${tokps:-NA}" "${fid:-NA%}" >> "$TMP"
  done
done
rm -f "$RAW"
echo "-- equal-weight aggregate (mean over regimes) --"
awk '{ t[$1]+=$3; f[$1]+=($4+0); n[$1]++ }
     END { for (m in t) printf "  %-12s speed %.1f tok/s   fidelity %.1f%%  (mean of %d regimes)\n", m, t[m]/n[m], f[m]/n[m], n[m] }' "$TMP"
rm -f "$TMP"
echo "-- axes: fidelity=teacher-forced per-token vs strict (primary, chaos-free). correctness=L3"
echo "   acceptability, read as DELTA: strict PASS -> bolt PASS means the fidelity divergence is"
echo "   benign for that task. absolute correctness is bounded by base-model capability (if strict"
echo "   FAILs a regime, that hook is base-model-limited, not an engine signal). --"
