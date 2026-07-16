#!/bin/bash
# LOOPY-rate vs GPU-pressure sweep (#47). Usage: sweep.sh "0 16 28 34 37 40 44" 3
# Per point: start ballast, N benchtest repeats (16GB tier forced), kill ballast.
# Appends to results.tsv: gpu_gb  run  test  mode  ttft  decode  tokens  stability
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN=/Users/penta2himajin/repos/qwisp/swift/.xcode-build-rel/Build/Products/Release/qwisp
export QWISP_MODEL=$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16
export QWISP_DEVICE_RAM=16
POINTS=${1:?points}; REPEATS=${2:-3}
TSV="$DIR/results.tsv"
[ -f "$TSV" ] || echo -e "gpu_gb\trun\ttest\tmode\tttft_s\tdecode_tps\ttokens\tstability" > "$TSV"

for X in $POINTS; do
  SIMPID=""
  if [ "$X" != "0" ]; then
    "$BIN" simulate --gpu-gb "$X" > "$DIR/sim-$X.log" 2>&1 &
    SIMPID=$!
    until grep -q "GPU ballast resident" "$DIR/sim-$X.log" 2>/dev/null; do sleep 1; done
  fi
  for r in $(seq 1 "$REPEATS"); do
    MD="$DIR/g$X-r$r.md"
    "$BIN" benchtest > "$MD" 2> "$DIR/g$X-r$r.err"
    # parse: | test | mode | TTFTs | decode tok/s | tokens | label (ratio) |
    awk -F'|' -v g="$X" -v r="$r" '
      NF>=7 && $2 !~ /test|---/ {
        gsub(/[ s]/,"",$4); gsub(/ tok\/s/,"",$5); gsub(/ /,"",$5); gsub(/ /,"",$6)
        stab=$7; gsub(/.*\(/,"",stab); gsub(/\).*/,"",stab)
        t=$2; m=$3; gsub(/^ +| +$/,"",t); gsub(/^ +| +$/,"",m)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", g, r, t, m, $4, $5, $6, stab
      }' "$MD" >> "$TSV"
    echo "== g$X r$r done: $(grep -c LOOPY "$MD" || true) LOOPY rows"
  done
  [ -n "$SIMPID" ] && kill "$SIMPID" 2>/dev/null && wait "$SIMPID" 2>/dev/null
done
echo "SWEEP DONE → $TSV"
