#!/usr/bin/env bash
# #154 M1 — shared-prefix KV duplication vs peak footprint (kill criterion 1).
#
# For B in {2,3,4}: warm-capture a shared prefix, then fire B concurrent admits that
# share it; sample /usr/bin/footprint throughout. Kill if duplicated prefix KV is
# <10% of peak footprint (see issue #154).
#
# Usage: scripts/bench_issue154_m1.sh [sharedPrefixTokens]   # default 8192
set -euo pipefail

P="${1:-8192}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/swift/.xcode-build-rel/Build/Products/Release/qwisp"
MODEL="${QWISP_MODEL:-$HOME/models/Ornith-1.5-35B-A3B-MLX-4bit}"
PORT="${QWISP_PORT:-8099}"
LANES="${QWISP_LANES:-4}"

[ -x "$BIN" ] || { echo "ERROR: build the qwisp scheme first"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: model not found at $MODEL (set QWISP_MODEL)"; exit 1; }
if pgrep -x qwisp >/dev/null; then echo "ERROR: qwisp already running — stop it first (GPU exclusive)"; exit 1; fi
if ! pmset -g batt | head -1 | grep -q "AC Power"; then
    echo "WARNING: on battery — DVFS noise; treat numbers as diagnostic"
fi

TS=$(date +%Y%m%d-%H%M%S)
OUT="/tmp/issue154-m1-$TS"
mkdir -p "$OUT"
echo "== #154 M1: sharedPrefix=$P tok, lanes=$LANES, model=$(basename "$MODEL"), out=$OUT =="

sample_footprint() {
    local pid="$1" dst="$2"
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
    } END { printf "%.0f", m+0 }' "$1"
}

env QWISP_MODEL="$MODEL" QWISP_LANES="$LANES" QWISP_PORT="$PORT" \
    QWISP_LANE_PREFIX=1 QWISP_LANE_PREFIX_MB="${QWISP_LANE_PREFIX_MB:-3072}" \
    "$BIN" serve > "$OUT/server.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ERROR: server did not become ready"; tail -40 "$OUT/server.log"; exit 1
fi
echo "server ready pid=$PID"

: > "$OUT/results.jsonl"
for B in 2 3 4; do
    echo ""
    echo "== B=$B =="
    FP="$OUT/footprint.B$B.txt"
    : > "$FP"
    sample_footprint "$PID" "$FP" &
    SAMP=$!
    node "$REPO/tools/lane_issue154_m1_probe.mjs" "127.0.0.1:$PORT" "$P" "$B" 16 \
        > "$OUT/B$B.json" || echo '{"ok":false,"error":"probe failed"}' > "$OUT/B$B.json"
    kill "$SAMP" 2>/dev/null || true
    wait "$SAMP" 2>/dev/null || true
    PEAK=$(peak_mb "$FP")
    python3 - "$OUT/B$B.json" "$PEAK" "$OUT/results.jsonl" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
peak = float(sys.argv[2] or 0)
dup = float(rec.get("duplicatedMB") or 0)
rec["peakFootprintMB"] = peak
rec["dupPctOfPeak"] = round(100.0 * dup / peak, 2) if peak > 0 else None
rec["killCriterion1"] = "KILL(<10%)" if (peak > 0 and dup < 0.10 * peak) else "SURVIVE(>=10%)"
print(json.dumps(rec, indent=2))
open(sys.argv[3], "a").write(json.dumps(rec) + "\n")
PY
    sleep 3
done

echo ""
echo "== #154 M1 summary =="
python3 - "$OUT/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{'B':>3} {'peakMB':>8} {'dupMB':>8} {'arenaMB~':>9} {'dup%':>7}  verdict")
kill_all = True
for r in rows:
    pct = r.get("dupPctOfPeak")
    v = r.get("killCriterion1", "?")
    if v != "KILL(<10%)": kill_all = False
    print(f"{r.get('B',0):>3} {r.get('peakFootprintMB',0):>8.0f} {r.get('duplicatedMB',0):>8.1f} "
          f"{r.get('arenaMBEst',0):>9.1f} {pct if pct is not None else float('nan'):>6.2f}%  {v}")
print()
if kill_all and rows:
    print("VERDICT: Kill criterion 1 TRIPS for all B — duplicated prefix KV is not a material cost.")
    print("Combined with issue #154 mechanism kill (no third path past sdpa_rows / L1), close as not-planned.")
else:
    print("VERDICT: At least one B survives the 10% bar — do not close without revisiting criterion 2.")
print(f"raw: {sys.argv[1]}")
PY
