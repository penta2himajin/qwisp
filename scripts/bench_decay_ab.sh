#!/usr/bin/env bash
# Paired same-session decay A/B for the MMA attention-prefill kernel (WS-A #137).
#
# Motivation: cross-day decay comparisons are VOID — 2026-07-24 measured ~45%
# drift on the untouched GDN/MoE stages vs the 07-23 baseline run. The only
# valid comparison is OFF→ON back-to-back in one session (same thermal/DVFS
# environment), which this script does.
#
# Usage:  scripts/bench_decay_ab.sh [maxCtx]     # default 24576 (diagnostic);
#                                                # use 49152 for the formal GO run
# Env:    QWISP_MODEL as usual. Requires AC power + no other GPU process
#         (the qwisp server counts as one).
# Output: two logs under /tmp + a side-by-side attn_ms table on stdout.
#         GO bar (formal run only): attn@47104 ON ≤ half of OFF.
set -euo pipefail

MAX="${1:-24576}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
POC="$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc"
[ -x "$POC" ] || { echo "ERROR: build qwisp-poc first"; exit 1; }
if pgrep -x qwisp >/dev/null; then echo "ERROR: qwisp server running — stop it first (GPU exclusive)"; exit 1; fi
if ! pmset -g batt | head -1 | grep -q "AC Power"; then
    echo "WARNING: on battery — DVFS makes reps spike; results are diagnostic only"
fi

TS=$(date +%H%M%S)
OFFLOG="/tmp/decay-ab-$TS-off.log"; ONLOG="/tmp/decay-ab-$TS-on.log"

echo "== pass 1/2: flag OFF (baseline), maxCtx=$MAX =="
QWISP_DECAY_MAX="$MAX" QWISP_RUN=long-context-decay \
    ~/bin/jacquard measure "$POC" stream 2>&1 | tee "$OFFLOG" | grep -E "^\s+[0-9]+ " || true
echo "== pass 2/2: flag ON (sdpa_prefill_mma) =="
QWISP_ATTN_MMA_PREFILL=1 QWISP_DECAY_MAX="$MAX" QWISP_RUN=long-context-decay \
    ~/bin/jacquard measure "$POC" stream 2>&1 | tee "$ONLOG" | grep -E "^\s+[0-9]+ " || true

echo ""
echo "== paired attn_ms (prefill chunks; decode rows excluded — flag is M>1 only) =="
python3 - "$OFFLOG" "$ONLOG" << 'EOF'
import re, sys
def prefill_rows(path):
    rows, in_prefill = {}, False
    for line in open(path):
        if "prefill: per-chunk" in line: in_prefill = True; continue
        if "decode (M=1)" in line: in_prefill = False
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if in_prefill and m:
            rows[int(m.group(1))] = float(m.group(2))
    return rows
off, on = prefill_rows(sys.argv[1]), prefill_rows(sys.argv[2])
print(f"{'pos':>8} {'attn OFF':>10} {'attn ON':>10} {'speedup':>8}")
for pos in sorted(off):
    if pos in on and on[pos] > 0:
        print(f"{pos:>8} {off[pos]:>10.0f} {on[pos]:>10.0f} {off[pos]/on[pos]:>7.2f}x")
EOF
echo ""
echo "logs: $OFFLOG / $ONLOG"
