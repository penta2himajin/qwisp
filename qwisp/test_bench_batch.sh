#!/usr/bin/env bash
# RED test for task ②A: bench_batch.sh bolt cell → raw bolt runner (single-shot CLI).
#
# Fixture test — no real GPU/model. A stub BIN (this script writes it to mktemp) impersonates
# qwisp-poc: it branches on QWISP_RUN. The stub proves that bench_batch.sh routes bolt through
# a per-regime `QWISP_RUN=raw-spec` CLI invocation (not the in-process bench-batch runner) and
# parses the raw human output lines (RawSpecRunner.swift:1098 speed / :1316 TF fidelity).
#
# Goal encoded here (fails on current tree, where bolt still goes through in-process batch):
#   1. suffix-spec rows still come from the single bench-batch invocation  (tokps 50.0)
#   2. bolt rows (4 regimes) come from raw-spec CLI  (92.5 tok/s, 92.9% TF fidelity)
#   3. QWISP_THROTTLE_DEFER is NOT propagated into the raw-spec call  (DEFER-leak marker absent)
#   4. QWISP_BOLT_WORKLOAD matches the regime of QWISP_MTP_REF          (workload marker absent)
#
# Exit 0 = PASS (prints BENCHBATCHTEST PASS), else FAIL (BENCHBATCHTEST FAIL).
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/qwisp/bench_batch.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- stub BIN: impersonates qwisp-poc, branching on QWISP_RUN ------------------------------
BIN="$WORK/qwisp-poc-stub"
cat > "$BIN" <<'STUB'
#!/usr/bin/env bash
case "${QWISP_RUN:-}" in
  bench-batch)
    # in-process batch: emit ONLY suffix-spec cells (bolt must NOT come from here anymore).
    for r in code agentic longctx shortnl; do
      echo "BENCHCELL|method=suffix-spec|regime=$r"
      echo "BENCH|method=suffix-spec|regime=$r|tokps=50.0"
      echo "PROMPT_TOKENS:1,2,3"
      echo "OUT_TOKENS:4,5,6"
    done
    ;;
  raw-spec)
    # trap the two known raw-CLI hazards:
    #  (a) QWISP_THROTTLE_DEFER must be stripped before the raw call (else invalid measurement)
    if [ -n "${QWISP_THROTTLE_DEFER:-}" ]; then
      echo "DEFER_LEAKED workload=${QWISP_BOLT_WORKLOAD:-?}" >> "${QWISP_TEST_DEFER_MARKER:?}"
    fi
    #  (b) QWISP_BOLT_WORKLOAD must equal the regime named by QWISP_MTP_REF
    ref_regime="$(basename "${QWISP_MTP_REF:-}" .safetensors)"
    if [ "${QWISP_BOLT_WORKLOAD:-}" != "$ref_regime" ]; then
      echo "MISMATCH bolt=${QWISP_BOLT_WORKLOAD:-?} ref=$ref_regime" >> "${QWISP_TEST_WORKLOAD_MARKER:?}"
    fi
    # canonical raw bolt human output (RawSpecRunner.swift:1098 / :1316)
    echo "[RawSpec] bolt(L3 near-lossless) C=${QWISP_RAW_C:-?}: 92.5 tok/s  accept/step=7.00  品質(vs ref) 120/128=94%"
    echo "[RawSpec] NOTE: bolt is L3 near-lossless (buddy remap, not strict). Quality vs ref is informational."
    echo "[RawSpec] bolt TF fidelity vs strict-canonical: 118/127=92.9% (chaos-free)"
    echo "PROMPT_TOKENS:1,2,3"
    echo "OUT_TOKENS:4,5,6"
    ;;
  *)
    echo "stub: unexpected QWISP_RUN=${QWISP_RUN:-<unset>}" >&2
    exit 3
    ;;
esac
STUB
chmod +x "$BIN"

# --- refs dir with empty per-regime safetensors -------------------------------------------
REFS="$WORK/refs"
mkdir -p "$REFS"
for r in code agentic longctx shortnl; do : > "$REFS/$r.safetensors"; done

# --- markers the stub writes on a violation (test asserts they stay absent) ---------------
DEFER_MARKER="$WORK/defer_leaked"
WORKLOAD_MARKER="$WORK/workload_mismatch"

# --- run bench_batch.sh against the stub.  QWISP_THROTTLE_DEFER=1 is set in the environment
#     on purpose: it MAY reach the in-process batch, but MUST NOT reach the raw-spec call. ---
OUT="$(
  QWISP_BENCH_BIN="$BIN" \
  QWISP_BENCH_MODEL="$WORK/model" \
  QWISP_BENCH_REFS="$REFS" \
  QWISP_BENCH_PY="/usr/bin/true" \
  QWISP_THROTTLE_DEFER=1 \
  QWISP_TEST_DEFER_MARKER="$DEFER_MARKER" \
  QWISP_TEST_WORKLOAD_MARKER="$WORKLOAD_MARKER" \
  bash "$SCRIPT" 64 128 0 "suffix-spec bolt" 2>/dev/null
)"

echo "$OUT"
echo "=== asserts ==="

fails=0
check() { # <desc> <0-if-ok>
  if [ "$2" -eq 0 ]; then echo "  ok   : $1"; else echo "  FAIL : $1"; fails=$((fails + 1)); fi
}

# 1) suffix-spec rows still present from the batch invocation (unchanged behavior)
printf '%s\n' "$OUT" | grep -qE 'suffix-spec[[:space:]]+code[[:space:]]+50\.0'
check "suffix-spec code row (tokps 50.0) present" $?

# 2) bolt rows for all 4 regimes, from raw-spec CLI: 92.5 tok/s AND 92.9% TF fidelity on the row
bolt_rows=0
for r in code agentic longctx shortnl; do
  if printf '%s\n' "$OUT" | grep -qE "bolt[[:space:]]+$r[[:space:]]+92\.5.*92\.9%"; then
    bolt_rows=$((bolt_rows + 1))
  fi
done
[ "$bolt_rows" -eq 4 ]
check "bolt raw rows for 4 regimes (92.5 tok/s / 92.9% fidelity) — got $bolt_rows/4" $?

# 3) QWISP_THROTTLE_DEFER must NOT have reached the raw-spec call
[ ! -f "$DEFER_MARKER" ]
check "QWISP_THROTTLE_DEFER not propagated to raw-spec" $?

# 4) QWISP_BOLT_WORKLOAD matched QWISP_MTP_REF regime on every raw call
[ ! -f "$WORKLOAD_MARKER" ]
check "QWISP_BOLT_WORKLOAD matches ref regime on every raw call" $?

echo "==="
if [ "$fails" -eq 0 ]; then
  echo "BENCHBATCHTEST PASS"
  exit 0
else
  echo "BENCHBATCHTEST FAIL ($fails assertion(s))"
  exit 1
fi
