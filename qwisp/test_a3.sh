#!/usr/bin/env bash
# test_a3.sh — G2 identity oracle for A3 pending-prefix (notes/04-a3-pending-prefix-spec.md §6).
#
# For each regime, runs the raw-spec loop twice:
#   QWISP_RAW_A3=0  (non-A3, baseline)
#   QWISP_RAW_A3=1  (A3 pending-prefix enabled)
# Then asserts:
#   (1) OUT_TOKENS byte-identical between the two runs         — losslessness
#   (2) A3 run self-check reports "${GEN}/${GEN} LOSSLESS"    — spec-vs-greedy
#   (3) A3 run log acknowledges QWISP_RAW_A3=1 activation     — RED gate before impl
#
# Usage:
#   ./qwisp/test_a3.sh                # all 4 regimes (slow: loads 35B model twice each)
#   ./qwisp/test_a3.sh code           # single regime (fast smoke check / RED probe)
#   ./qwisp/test_a3.sh longctx        # A3 primary stress regime (§6 G2 mandatory)
#
# Env overrides:
#   QWISP_BENCH_BIN — path to compiled qwisp-poc binary
#   QWISP_GEN       — token count (default 128)
#
# Exits 0 only when ALL regimes PASS all three checks.
# RED state (before A3 is implemented): check (3) fails because the binary currently
# ignores QWISP_RAW_A3 and emits no acknowledgment log line.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BENCH_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc}"
GEN="${QWISP_GEN:-128}"

# Default regime order: longctx first (most A3 stress per spec §6), then agentic, code, shortnl.
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

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN (build first)"; exit 1; }

FAILED=0

for regime in "${REGIMES[@]}"; do
    REF="$REPO/refs/${regime}.safetensors"
    echo "=== regime: ${regime} ==="

    if [ ! -f "$REF" ]; then
        echo "  ERROR: ref not found: $REF"
        FAILED=$((FAILED + 1))
        continue
    fi

    TMP_REF=$(mktemp /tmp/qwisp_a3_ref_XXXXXX)
    TMP_A3=$(mktemp /tmp/qwisp_a3_new_XXXXXX)

    # Run non-A3 baseline (QWISP_RAW_A3=0)
    # '|| true' so script continues even if the binary exits nonzero
    QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_RAW_A3=0 \
        QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN="${GEN}" \
        QWISP_MTP_REF="${REF}" "${BIN}" stream >"$TMP_REF" 2>&1 || true

    # Run A3 (QWISP_RAW_A3=1)
    QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_RAW_A3=1 \
        QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN="${GEN}" \
        QWISP_MTP_REF="${REF}" "${BIN}" stream >"$TMP_A3" 2>&1 || true

    # Extract OUT_TOKENS lines
    OUT_REF=$(grep '^OUT_TOKENS:' "$TMP_REF" 2>/dev/null || echo "MISSING")
    OUT_A3=$(grep  '^OUT_TOKENS:' "$TMP_A3"  2>/dev/null || echo "MISSING")

    REGIME_PASS=1

    # ── Check 1: OUT_TOKENS byte-identical ──────────────────────────────────
    # A3 is strictly lossless (§4): both runs must produce the same token sequence.
    if [ "$OUT_REF" = "$OUT_A3" ] && [ "$OUT_REF" != "MISSING" ]; then
        echo "  [tokens]   PASS (${GEN} tokens byte-identical)"
    else
        echo "  [tokens]   FAIL"
        if [ "$OUT_REF" = "MISSING" ]; then
            echo "    non-A3 OUT_TOKENS: MISSING (did the binary crash?)"
            tail -5 "$TMP_REF" | sed 's/^/    /'
        fi
        if [ "$OUT_A3" = "MISSING" ]; then
            echo "    A3 OUT_TOKENS: MISSING (did the binary crash?)"
            tail -5 "$TMP_A3" | sed 's/^/    /'
        fi
        if [ "$OUT_REF" != "MISSING" ] && [ "$OUT_A3" != "MISSING" ]; then
            echo "    non-A3: ${OUT_REF:0:100}"
            echo "    A3:     ${OUT_A3:0:100}"
        fi
        REGIME_PASS=0
    fi

    # ── Check 2: A3 self-check reports GEN/GEN LOSSLESS ────────────────────
    # QWISP_RAWSPEC_CHECK=1 runs a greedy M=1 self-check and prints the result.
    # The A3 path must produce spec output that matches greedy exactly (§4 guarantee).
    LOSSLESS_PAT="${GEN}/${GEN} LOSSLESS"
    if grep -q "${LOSSLESS_PAT}" "$TMP_A3" 2>/dev/null; then
        echo "  [lossless] PASS (${LOSSLESS_PAT})"
    else
        echo "  [lossless] FAIL (expected '${LOSSLESS_PAT}' in A3 output)"
        echo "  --- A3 output tail ---"
        tail -10 "$TMP_A3" | sed 's/^/    /'
        REGIME_PASS=0
    fi

    # ── Check 3 (RED gate): A3 mode acknowledged by the binary ──────────────
    # When QWISP_RAW_A3=1 is not yet implemented, the env var is silently ignored
    # and the binary emits no "A3"-related log lines → this check FAILS (RED phase).
    #
    # The A3 implementation MUST emit a log line that matches one of:
    #   "A3=true"  /  "A3 pending"  /  "rawA3"  /  "[raw-spec] ... a3"  / "pending-prefix"
    # when QWISP_RAW_A3=1 is set. This is the authoritative RED→GREEN transition signal.
    if grep -qi \
        'A3=true\|A3.*pending\|pending.prefix.*a3\|rawA3\|raw-a3\|\[raw-spec\].*a3\b\|A3.*enabled\|A3.*active' \
        "$TMP_A3" 2>/dev/null; then
        echo "  [a3-mode]  PASS (A3 acknowledgment found in log)"
    else
        echo "  [a3-mode]  FAIL (QWISP_RAW_A3=1 not acknowledged — A3 not implemented yet)"
        echo "  Implementation must log a line matching:"
        echo "    A3=true | A3.*pending | pending-prefix.*A3 | rawA3 | [raw-spec].*a3"
        echo "  when QWISP_RAW_A3=1 is set."
        REGIME_PASS=0
    fi

    rm -f "$TMP_REF" "$TMP_A3"

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
