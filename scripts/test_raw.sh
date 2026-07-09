#!/usr/bin/env bash
# D1 TDD test runner: execute the raw-verify suite and exit nonzero on failure.
#
# Usage: scripts/test_raw.sh
# Env overrides (same as bench.sh):
#   QWISP_BENCH_BIN  — path to compiled qwisp-poc binary
#
# Exits 0 only when all tests PASS (RAWTESTS X/X).
# Currently expected: 2 PASS, 7 FAIL (RED phase — D1 stubs not yet implemented).
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QWISP_BENCH_BIN:-$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc}"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN (build first with xcodebuild)"; exit 1; }

echo "=== Qwisp D1 raw-verify tests ==="
out="$(QWISP_RUN=raw-tests "$BIN" stream 2>&1)"
echo "$out"
echo "==="

# Check for RAWTESTS summary line
rawtests_line="$(printf '%s\n' "$out" | grep '^RAWTESTS')"
if [ -z "$rawtests_line" ]; then
    echo "RESULT: FAIL (RAWTESTS summary line missing — did the suite crash?)"
    exit 1
fi

# Parse "RAWTESTS X/Y"
passed="$(printf '%s\n' "$rawtests_line" | sed 's|RAWTESTS ||' | cut -d/ -f1)"
total="$(printf '%s\n' "$rawtests_line" | sed 's|RAWTESTS ||' | cut -d/ -f2)"

# Exit nonzero if any FAIL line present OR if counts don't match
any_fail="$(printf '%s\n' "$out" | grep '\[raw-test\].*: FAIL' | wc -l | tr -d ' ')"
if [ "$any_fail" -gt 0 ] || [ "$passed" != "$total" ]; then
    echo "RESULT: FAIL ($rawtests_line, $any_fail test(s) with FAIL lines)"
    exit 1
fi

echo "RESULT: PASS ($rawtests_line)"
