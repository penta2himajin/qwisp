#!/usr/bin/env bash
# Memory-growth diagnostic + #148 regression check for the lane serving path.
# (Name kept for continuity with issue #148; "arena ratchet" is the DISPROVED original
# hypothesis, not what this measures — see below.)
#
# ORIGINAL HYPOTHESIS (DISPROVED 2026-07-27, kept for the record): that per-request
# arena sizing ratchets MLX's buffer pool, fixable with Memory.clearCache() on lane
# release. Both halves are wrong. Bounding MLX's pool (Memory.cacheLimit=2GB) held
# cacheMemory at ~2,050MB and did NOT stop the growth; and the growth is not driven by
# arena size at all.
#
# WHAT THIS SCRIPT ACTUALLY MEASURES (see issue #148): the retained quantity follows
#     nonMLXmetal ~= 19,750MB + concurrency * promptLen(current round) * ~2.03MB/token
# i.e. it tracks the PROMPT LENGTH of the round being sampled, not cumulative admissions
# and not arena size. `vary` grows prompt and arena together, so this A/B never actually
# isolated arena size — the two were confounded. `fixeduniq` is not "no leak"; it is the
# same law pinned at a constant prompt length (its flat 44.1GB is exactly where `vary`
# passes as its prompt crosses ~6,000 tokens).
#
# Root cause: the steel-hybrid prefill wraps per-layer MLX temporaries in noCopy
# MTLBuffers that are autoreleased, and the decode thread's autorelease pool drains only
# at thread exit (= the round boundary). Hence footprint SAWTOOTHS to a constant floor
# rather than ratcheting — read the full fp.txt series, not just first/last/peak, which
# is how this was missed initially. Fixed by draining per prefill chunk
# (QWISP_LANE_CHUNK_POOL, default ON): peak 46,080 -> 24,576MB on the `vary` shape.
#
# A/B (fresh server per shape, footprint sampled throughout):
#   vary      = prompt grows +1000 tok each round (prompt AND arena size both vary)
#   fixeduniq = constant prompt size, unique content (equal prefill work, no prefix reuse)
#   fixed     = constant prompt, SHARED prefix -> prefix cache serves it, almost no
#               prefill happens. CONFOUNDED; do not quote it as a control.
# Regression check for #148: `vary` peak footprint must stay near the floor.
#
# Usage: scripts/diag_lane_arena_ratchet.sh [rounds] [baseTokens] [concurrency]
set -euo pipefail

ROUNDS="${1:-10}"
BASE="${2:-1500}"
CONC="${3:-2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp"
MODEL="${QWISP_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"
PORT="${QWISP_PORT:-8099}"
LANES="${QWISP_LANES:-4}"

[ -x "$BIN" ] || { echo "ERROR: build the qwisp scheme first"; exit 1; }
if pgrep -x qwisp >/dev/null; then echo "ERROR: qwisp already running (GPU exclusive)"; exit 1; fi

TS=$(date +%H%M%S)
OUT="/tmp/lane-ratchet-$TS"
mkdir -p "$OUT"
echo "== arena-ratchet diagnostic: rounds=$ROUNDS base=$BASE conc=$CONC lanes=$LANES out=$OUT =="

run_shape() {
    local shape
    shape="$1"
    echo ""
    echo "== shape: $shape =="
    env QWISP_MODEL="$MODEL" QWISP_LANES="$LANES" QWISP_PORT="$PORT" "$BIN" serve \
        > "$OUT/$shape.server.log" 2>&1 &
    local pid
    pid=$!
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        sleep 2
    done
    # Baseline footprint BEFORE any request: isolates model residency from per-request growth.
    echo "baseline $(footprint "$pid" 2>/dev/null | grep -m1 'Footprint:')" | tee -a "$OUT/$shape.fp.txt"
    ( while kill -0 "$pid" 2>/dev/null; do
        footprint "$pid" 2>/dev/null | grep -m1 "Footprint:" >> "$OUT/$shape.fp.txt" || true
        sleep 3
      done ) &
    local sampler
    sampler=$!
    node "$REPO/tools/lane_stage_b_probe.mjs" ratchet "127.0.0.1:$PORT" "$ROUNDS" "$shape" "$BASE" "$CONC" \
        > "$OUT/$shape.json" 2> "$OUT/$shape.progress.txt" || echo '{"error":"probe failed"}' > "$OUT/$shape.json"
    kill "$sampler" 2>/dev/null || true
    echo "final    $(footprint "$pid" 2>/dev/null | grep -m1 'Footprint:')" | tee -a "$OUT/$shape.fp.txt"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 2
}

# SHAPES overrides the pass list, e.g. SHAPES="fixeduniq" to re-run one control only.
for sh in ${SHAPES:-vary fixeduniq}; do run_shape "$sh"; done

echo ""
echo "== summary (footprint MB over time) =="
python3 - "$OUT" << 'EOF'
import os, re, sys
out = sys.argv[1]
def series(shape):
    p = os.path.join(out, f"{shape}.fp.txt")
    vals = []
    for line in open(p):
        m = re.search(r"Footprint:\s+([\d.]+)\s+([KMG]B)", line)
        if not m: continue
        v = float(m.group(1))
        v = v / 1024 if m.group(2) == "KB" else v * 1024 if m.group(2) == "GB" else v
        vals.append(v)
    return vals
for shape in os.environ.get("SHAPES", "vary fixeduniq").split():
    v = series(shape)
    if not v:
        print(f"{shape}: no samples"); continue
    print(f"{shape:>5}: first={v[0]:.0f}MB  last={v[-1]:.0f}MB  peak={max(v):.0f}MB  "
          f"growth={v[-1]-v[0]:+.0f}MB  n={len(v)}")
print("\n(#148) vary's peak should now sit near its floor. A returning ramp means the")
print("per-chunk autoreleasepool drain regressed. Always read the full fp.txt series —")
print("this summary hides the per-round sawtooth that made the bug unreadable at first.")
EOF
echo ""
echo "artifacts: $OUT"
