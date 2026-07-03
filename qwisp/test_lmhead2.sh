#!/usr/bin/env bash
# test_lmhead2.sh — G2 identity oracle for lm_head margin-certified 2-bit
#                   (notes/05-lmhead-margin-cert-spec.md §6 G2).
#
# For each regime, runs the raw-spec loop twice:
#   QWISP_LMHEAD2=0  (baseline: standard 4-bit lm_head)
#   QWISP_LMHEAD2=1  (margin-cert 2-bit lm_head, opt-in)
# Then asserts:
#   (a) OUT_TOKENS byte-identical between the two runs        — strict lossless
#   (b) LMHEAD2=1 run reports "${GEN}/${GEN} LOSSLESS"       — self-check
#   (c) LMHEAD2=1 run prints a cert-rate line matching        — RED gate before impl
#       regex "lmhead2 cert-rate"
#
# Check (c) is the RED→GREEN transition signal:
#   RED  (pre-implementation): binary ignores QWISP_LMHEAD2 → no cert-rate line → (c) FAILS.
#   GREEN (post-implementation): binary prints "lmhead2 cert-rate: X/Y=Z%" → (c) PASSES.
#
# Usage:
#   ./qwisp/test_lmhead2.sh              # all 4 regimes (heavy: loads 35B model twice each)
#   ./qwisp/test_lmhead2.sh code         # single regime (fast smoke check / RED probe)
#
# Env overrides:
#   QWISP_BENCH_BIN — path to compiled qwisp-poc binary
#   QWISP_GEN       — token count (default 128)
#
# Exits 0 only when ALL tested regimes PASS all three checks.
# RED state exits 1 (check (c) fails for every regime before implementation).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BENCH_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc}"
GEN="${QWISP_GEN:-128}"

# Default regime order (longctx first for max cert-rate stress, then agentic, code, shortnl)
ALL_REGIMES=(longctx agentic code shortnl)

# Optional positional arg: single regime name for fast smoke / RED check.
REGIMES=("${ALL_REGIMES[@]}")
if [ "${1:-}" != "" ]; then
    MATCH=0
    for r in "${ALL_REGIMES[@]}"; do
        [ "$r" = "$1" ] && { MATCH=1; break; }
    done
    if [ "$MATCH" = "1" ]; then
        REGIMES=("$1")
    else
        echo "ERROR: unknown regime '$1'. Valid: ${ALL_REGIMES[*]}"
        exit 1
    fi
fi

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN  (build first)"; exit 1; }

FAILED=0

for regime in "${REGIMES[@]}"; do
    REF="$REPO/refs/${regime}.safetensors"
    echo "=== regime: ${regime} ==="

    if [ ! -f "$REF" ]; then
        echo "  ERROR: ref not found: $REF"
        FAILED=$((FAILED + 1))
        continue
    fi

    TMP_BASE=$(mktemp /tmp/qwisp_lmh2_base_XXXXXX)
    TMP_LMH2=$(mktemp /tmp/qwisp_lmh2_new_XXXXXX)

    # ── Run 1: baseline (QWISP_LMHEAD2=0) ──────────────────────────────────
    QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_LMHEAD2=0 \
        QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN="${GEN}" \
        QWISP_MTP_REF="${REF}" "${BIN}" stream >"$TMP_BASE" 2>&1 || true

    # ── Run 2: margin-cert 2-bit lm_head (QWISP_LMHEAD2=1) ─────────────────
    QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_LMHEAD2=1 \
        QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN="${GEN}" \
        QWISP_MTP_REF="${REF}" "${BIN}" stream >"$TMP_LMH2" 2>&1 || true

    # Extract OUT_TOKENS lines
    OUT_BASE=$(grep '^OUT_TOKENS:' "$TMP_BASE" 2>/dev/null || echo "MISSING")
    OUT_LMH2=$(grep '^OUT_TOKENS:' "$TMP_LMH2"  2>/dev/null || echo "MISSING")

    REGIME_PASS=1

    # ── Check (a): OUT_TOKENS byte-identical ────────────────────────────────
    # strict lossless requirement (§6 G2): both runs must produce the same token sequence.
    if [ "$OUT_BASE" = "$OUT_LMH2" ] && [ "$OUT_BASE" != "MISSING" ]; then
        echo "  [tokens]    PASS (${GEN} tokens byte-identical)"
    else
        echo "  [tokens]    FAIL"
        if [ "$OUT_BASE" = "MISSING" ]; then
            echo "    baseline OUT_TOKENS: MISSING (did the binary crash?)"
            tail -5 "$TMP_BASE" | sed 's/^/    /'
        fi
        if [ "$OUT_LMH2" = "MISSING" ]; then
            echo "    lmhead2 OUT_TOKENS: MISSING (did the binary crash?)"
            tail -5 "$TMP_LMH2" | sed 's/^/    /'
        fi
        if [ "$OUT_BASE" != "MISSING" ] && [ "$OUT_LMH2" != "MISSING" ]; then
            echo "    baseline: ${OUT_BASE:0:120}"
            echo "    lmhead2:  ${OUT_LMH2:0:120}"
        fi
        REGIME_PASS=0
    fi

    # ── Check (b): LMHEAD2=1 self-check reports GEN/GEN LOSSLESS ───────────
    # QWISP_RAWSPEC_CHECK=1 runs a greedy M=1 self-check and prints the result.
    LOSSLESS_PAT="${GEN}/${GEN} LOSSLESS"
    if grep -q "${LOSSLESS_PAT}" "$TMP_LMH2" 2>/dev/null; then
        echo "  [lossless]  PASS (${LOSSLESS_PAT})"
    else
        echo "  [lossless]  FAIL (expected '${LOSSLESS_PAT}' in LMHEAD2=1 output)"
        echo "  --- LMHEAD2=1 output tail ---"
        tail -10 "$TMP_LMH2" | sed 's/^/    /'
        REGIME_PASS=0
    fi

    # ── Check (c): cert-rate telemetry line present (RED gate) ──────────────
    # Spec §3.2 step 6: RawSpecRunner must print a line matching:
    #   "[RawSpec] lmhead2 cert-rate: X/Y=Z%"
    # when QWISP_LMHEAD2=1.
    # Before implementation: binary ignores QWISP_LMHEAD2 → no such line → FAIL (RED).
    # After implementation: line present → PASS (GREEN).
    if grep -qi 'lmhead2 cert-rate' "$TMP_LMH2" 2>/dev/null; then
        CERT_LINE=$(grep -i 'lmhead2 cert-rate' "$TMP_LMH2" | head -1)
        echo "  [cert-rate] PASS (${CERT_LINE})"
    else
        echo "  [cert-rate] FAIL (QWISP_LMHEAD2=1 produced no 'lmhead2 cert-rate' line)"
        echo "  Implementation must emit a line matching regex 'lmhead2 cert-rate' when"
        echo "  QWISP_LMHEAD2=1 is set (spec §3.2 step 6, via RawSpecRunner telemetry)."
        echo "  This is the RED gate: implementation not yet present."
        REGIME_PASS=0
    fi

    rm -f "$TMP_BASE" "$TMP_LMH2"

    if [ "$REGIME_PASS" = "1" ]; then
        echo "  → PASS"
    else
        echo "  → FAIL"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" = "0" ]; then
    echo "RESULT: PASS (regimes: ${REGIMES[*]})"
    exit 0
else
    echo "RESULT: FAIL (${FAILED} regime(s) failed)"
    exit 1
fi
