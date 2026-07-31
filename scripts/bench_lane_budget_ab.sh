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

# BUDGETS: one ON pass per value (sweep in ONE session so all arms share the OFF
# control and the same thermal state — separate invocations would re-pay the ~95s
# OFF pass per arm and compare across sessions, which notes/20's paired-A/B
# discipline forbids). STEADY_TOKENS (read by the probe) sets lane 0's stream length.
BUDGETS="${BUDGETS:-2048}"
FILES=("/tmp/lane-budget-ab-$TS-off.json")
# MUST pass =0 explicitly: QWISP_TOKEN_BUDGET_SCHED defaults to 1 since the 2026-07-25
# GO (LaneServe.swift, notes/21). An empty env here silently ran the ON path as the
# "OFF" control — both arms then measured identical (k=12, sum within 2.5%).
run_pass off "QWISP_TOKEN_BUDGET_SCHED=0"
for b in $BUDGETS; do
    run_pass "on$b" "QWISP_TOKEN_BUDGET_SCHED=1 QWISP_TOKEN_BUDGET=$b"
    FILES+=("/tmp/lane-budget-ab-$TS-on$b.json")
done

echo ""
echo "== summary (steady stream = ${STEADY_TOKENS:-45} tokens, big prompt = $BIG) =="
python3 - "${FILES[@]}" << 'EOF'
import json, sys, os, re
rows = []
for path in sys.argv[1:]:
    label = re.sub(r'^.*?-\d{6}-(.*)\.json$', r'\1', os.path.basename(path))
    rows.append((label, json.load(open(path))))
print(f"{'':>9} {'n':>4} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}  (ms)")
for name, d in rows:
    print(f"{name:>9} {d['n']:>4} {d['p50']:>8} {d['p90']:>8} {d['p99']:>8} {d['max']:>8}")
# Polluted-gap count is the model's real observable: gaps an order of magnitude above
# p50 are the prefill-bearing rounds (k). Predicted k = ceil(promptLen / budget).
print("")
for name, d in rows:
    gaps = d.get('gapsMs') or []
    big = [g for g in gaps if d['p50'] and g > 10 * d['p50']]
    void = "" if d.get('bTokens') else "   <-- VOID: big admit produced 0 tokens (dropped)"
    print(f"{name:>9} polluted gaps k={len(big):>3} / n={d['n']:<4} ({100*len(big)/max(1,d['n']):.0f}%)  sum={sum(big)/1000:.1f}s{void}")
EOF
