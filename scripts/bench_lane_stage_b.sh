#!/usr/bin/env bash
# WS-B Stage B bench (notes/22 "Bench verification"): the three Stage B measurements,
# paired same-session, GPU-exclusive. Mirrors scripts/bench_lane_budget_ab.sh's
# server lifecycle discipline (AC power, kill between passes, refuse if qwisp runs).
#
# Two passes over ONE binary, one model load each:
#   OFF = QWISP_TOKEN_BUDGET_SCHED=0 (legacy atomic path, fixed 16K lane cap)
#   ON  = default (Stage A scheduler + Stage B adaptive sizing)
# Each pass runs:
#   1. e2e   — a ~35K prompt. OFF is the POSITIVE CONTROL: it must come back empty
#              (the silent no-op Stage B fixes). ON must stream real content.
#   2. tps   — B=1..4 concurrent SHORT prompts; per-B mean tok/s must match within
#              ±5% (isolates adaptive-sizing overhead; short prompts never resize).
#   3. conc  — ON pass only: 3 SIMULTANEOUS large admits with `footprint` sampled
#              throughout. 3, not 2: the round-1 reservedBytes TOCTOU guard only
#              bites when budgetedStep peeks at >=3 free slots in one round.
#
# Usage: scripts/bench_lane_stage_b.sh [promptTokens]   # default 35000
set -euo pipefail

BIG="${1:-35000}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp"
MODEL="${QWISP_MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16}"
PORT="${QWISP_PORT:-8099}"
LANES="${QWISP_LANES:-4}"

[ -x "$BIN" ] || { echo "ERROR: build the qwisp scheme first"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: model not found at $MODEL (set QWISP_MODEL)"; exit 1; }
if pgrep -x qwisp >/dev/null; then echo "ERROR: qwisp server already running — stop it first (GPU exclusive)"; exit 1; fi
if ! pmset -g batt | head -1 | grep -q "AC Power"; then
    echo "WARNING: on battery — DVFS makes reps spike; results are diagnostic only"
fi

TS=$(date +%H%M%S)
OUT="/tmp/lane-stage-b-$TS"
mkdir -p "$OUT"
echo "== Stage B bench: prompt=$BIG tok, lanes=$LANES, out=$OUT =="

# Peak "Footprint: N KB" reported by /usr/bin/footprint over the sampling window,
# normalised to MB. `footprint`, never `ps rss` (rss undercounts wired GPU pages).
sample_footprint() {
    local pid
    pid="$1"
    local dst
    dst="$2"
    while kill -0 "$pid" 2>/dev/null; do
        footprint "$pid" 2>/dev/null | grep -m1 "Footprint:" >> "$dst" || true
        sleep 2
    done
}

peak_mb() {
    awk '{
        for (i = 1; i <= NF; i++) if ($i == "Footprint:") { v = $(i+1); u = $(i+2); break }
        if (u == "KB") v /= 1024; else if (u == "GB") v *= 1024;
        if (v > m) m = v
    } END { printf "%.0f", m }' "$1"
}

run_pass() {
    local label
    label="$1"
    local extra_env
    extra_env="$2"
    echo ""
    echo "== pass: $label (env: ${extra_env:-default}) =="
    # No QWISP_LANE_CTX override here, deliberately: Stage B's whole claim is that
    # the SHIPPED default admits a $BIG-token prompt. bench_lane_budget_ab.sh had to
    # raise it to 32768; needing that override again would itself be the failure.
    env $extra_env QWISP_MODEL="$MODEL" QWISP_LANES="$LANES" QWISP_PORT="$PORT" "$BIN" serve \
        > "$OUT/$label.server.log" 2>&1 &
    local pid
    pid=$!
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        sleep 2
    done

    echo "-- e2e ($BIG tok)"
    node "$REPO/tools/lane_stage_b_probe.mjs" e2e "127.0.0.1:$PORT" "$BIG" 32 \
        > "$OUT/$label.e2e.json" || echo '{"error":"probe failed"}' > "$OUT/$label.e2e.json"
    cat "$OUT/$label.e2e.json"

    local b
    for b in 1 2 3 4; do
        echo "-- tps B=$b"
        node "$REPO/tools/lane_stage_b_probe.mjs" tps "127.0.0.1:$PORT" "$b" 64 \
            > "$OUT/$label.tps$b.json"
        cat "$OUT/$label.tps$b.json"
    done

    if [ "$label" = "on" ]; then
        echo "-- conc N=3 ($BIG tok each) + footprint"
        sample_footprint "$pid" "$OUT/footprint.txt" &
        local sampler
        sampler=$!
        node "$REPO/tools/lane_stage_b_probe.mjs" conc "127.0.0.1:$PORT" "$BIG" 3 16 \
            > "$OUT/$label.conc.json" || echo '{"error":"probe failed"}' > "$OUT/$label.conc.json"
        kill "$sampler" 2>/dev/null || true
        cat "$OUT/$label.conc.json"
        echo "peak footprint: $(peak_mb "$OUT/footprint.txt") MB"
    fi

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 2   # let wired GPU pages settle before the next pass
}

run_pass off "QWISP_TOKEN_BUDGET_SCHED=0"
run_pass on  ""

echo ""
echo "== summary =="
python3 - "$OUT" << 'EOF'
import json, os, sys
out = sys.argv[1]
def load(name):
    p = os.path.join(out, name)
    try: return json.load(open(p))
    except Exception: return {}
print("1) E2E 35K admit (OFF is the positive control — empty is EXPECTED there)")
for lbl in ("off", "on"):
    d = load(f"{lbl}.e2e.json")
    print(f"  {lbl:>4}: deltas={d.get('deltas')} ttft={d.get('ttftMs')}ms total={d.get('totalMs')}ms "
          f"status={d.get('status')} sample={d.get('sample','')!r}")
print("\n2) No-regression A/B, short prompts (mean tok/s per stream)")
print(f"  {'B':>2} {'OFF':>8} {'ON':>8} {'Δ%':>7}")
worst = 0.0
for b in (1, 2, 3, 4):
    o, n = load(f"off.tps{b}.json").get("meanTps"), load(f"on.tps{b}.json").get("meanTps")
    if o and n:
        d = (n - o) / o * 100
        worst = max(worst, abs(d))
        print(f"  {b:>2} {o:>8.2f} {n:>8.2f} {d:>+6.1f}%")
    else:
        print(f"  {b:>2} {str(o):>8} {str(n):>8} {'n/a':>7}")
print(f"  worst |Δ| = {worst:.1f}%  (bar: <=5%)")
c = load("on.conc.json")
print(f"\n3) 3 simultaneous {c.get('promptTokens')}-tok admits: ok={c.get('ok')}")
for i, s in enumerate(c.get("streams", [])):
    print(f"  stream {i}: deltas={s['deltas']} ttft={s['ttftMs']}ms total={s['totalMs']}ms status={s['status']}")
EOF
echo ""
echo "artifacts: $OUT"
