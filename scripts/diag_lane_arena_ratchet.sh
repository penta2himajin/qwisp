#!/usr/bin/env bash
# Diagnostic for the 2026-07-25 trial-server OOM (HANDOFF RED block): does the lane
# path's per-request arena sizing ratchet MLX's buffer pool upward?
#
# Hypothesis: LaneServe.release() drops the lane forward but never calls
# Memory.clearCache(). MLX keeps freed buffers pooled (LLMBackend.swift:473-481
# documents exactly this on the serialize path, where it IS cleared: "observed
# 2026-07-22: 39GB IOAccelerator after growth rebuilds on a 64GB box -> swap thrash,
# prefill 1 tok/s"). Before Stage B every lane arena was the same 16384 tokens, so
# the pool recycled perfectly. Stage B sizes each arena per request, so a freed
# size-X buffer cannot serve a later size-Y request.
#
# A/B (two fresh servers, footprint sampled throughout):
#   vary  = prompt grows +1000 tok each round -> a DIFFERENT arena size every admit
#   fixed = same prompt size every round      -> IDENTICAL arena sizes
# Prompt CONTENT is distinct in both, so prefix-cache growth is common-mode and
# cannot explain a divergence. If `vary` ratchets and `fixed` stays flat, the
# hypothesis holds and the fix is a clearCache on lane release.
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
for sh in ${SHAPES:-vary fixed}; do run_shape "$sh"; done

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
for shape in os.environ.get("SHAPES", "vary fixed").split():
    v = series(shape)
    if not v:
        print(f"{shape}: no samples"); continue
    print(f"{shape:>5}: first={v[0]:.0f}MB  last={v[-1]:.0f}MB  peak={max(v):.0f}MB  "
          f"growth={v[-1]-v[0]:+.0f}MB  n={len(v)}")
print("\nIf vary grows substantially and fixed stays flat, the per-request arena size")
print("churn is ratcheting MLX's buffer pool -> fix is Memory.clearCache() on lane release.")
EOF
echo ""
echo "artifacts: $OUT"
