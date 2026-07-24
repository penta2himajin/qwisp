#!/usr/bin/env bash
# WS-B Stage A GO-bar measurement (notes/21): paired same-session A/B of the
# token-budget admission scheduler.
#
# Scenario: 2 lanes; lane 0 holds a steady decode stream, lane 1 admits a large
# prompt mid-stream. Metric: inter-token-latency (ITL) distribution of lane 0's
# stream, QWISP_TOKEN_BUDGET_SCHED=1 vs unset. GO bar: interleaved p99 ITL
# regresses by no more than ~1 chunk-worth of decode latency (bounded by
# ceil(budget/chunk_size) token-times), not by the other request's entire
# prefill duration (today's unbounded stall).
#
# Doctrine (notes/20's measurement lessons — same discipline as bench_decay_ab.sh):
# same-session paired A/B only, AC power, GPU-exclusive (server counts as the GPU
# process; kill it between passes).
#
# Usage: scripts/bench_lane_budget_ab.sh [bigPromptTokens]   # default 24000
set -euo pipefail

BIG="${1:-24000}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp"
MODEL="${QWISP_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"
PORT="${QWISP_PORT:-8099}"

[ -x "$BIN" ] || { echo "ERROR: build the qwisp scheme first"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: model not found at $MODEL (set QWISP_MODEL)"; exit 1; }
if pgrep -x qwisp >/dev/null; then echo "ERROR: qwisp server already running — stop it first (GPU exclusive)"; exit 1; fi
if ! pmset -g batt | head -1 | grep -q "AC Power"; then
    echo "WARNING: on battery — DVFS makes reps spike; results are diagnostic only"
fi

TS=$(date +%H%M%S)
run_pass() {
    local label="$1"
    local extra_env="$2"
    local out="/tmp/lane-budget-ab-$TS-$label.json"
    echo "== pass: $label (env: ${extra_env:-none}) =="
    # QWISP_LANE_CTX raised above the 16384 shipped default ONLY for this measurement —
    # a $BIG-token admit must fit the lane KV arena or maxTokens ceiling clamps to 0 and
    # the admit becomes a silent no-op (lifting the shipped cap is Stage B; this is a
    # bench-only override, not a default change).
    env $extra_env QWISP_MODEL="$MODEL" QWISP_LANES=2 QWISP_PORT="$PORT" QWISP_LANE_CTX=32768 "$BIN" serve \
        > "/tmp/lane-budget-ab-$TS-$label.server.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        sleep 2
    done
    node "$REPO/tools/lane_budget_probe.mjs" "127.0.0.1:$PORT" "$BIG" > "$out"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 1   # let the GPU/wired pages settle before the next pass
    echo "  -> $out"
    cat "$out"
}

run_pass off ""
run_pass on  "QWISP_TOKEN_BUDGET_SCHED=1"

echo ""
echo "== summary =="
python3 - "/tmp/lane-budget-ab-$TS-off.json" "/tmp/lane-budget-ab-$TS-on.json" << 'EOF'
import json, sys
off = json.load(open(sys.argv[1]))
on = json.load(open(sys.argv[2]))
print(f"{'':>6} {'n':>4} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}  (ms)")
for name, d in (("OFF", off), ("ON", on)):
    print(f"{name:>6} {d['n']:>4} {d['p50']:>8} {d['p90']:>8} {d['p99']:>8} {d['max']:>8}")
if off['p99'] and on['p99']:
    print(f"\np99 ratio (ON/OFF): {on['p99']/off['p99']:.2f}x")
    print(f"max ratio (ON/OFF): {on['max']/off['max']:.2f}x")
EOF
