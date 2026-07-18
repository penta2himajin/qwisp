#!/bin/bash
# speed round 2: 3 reps, interleaved config order, qs+tcp
set -u
cd "$(dirname "$0")"
BIN=../../../swift/.xcode-build-rel/Build/Products/Release/qwisp
CAL128=$HOME/.mtplx/models/qwisp-experts-2bit-cal128
QS="Write a detailed step-by-step explanation of how quicksort works, with a Python implementation."
TCP="Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery."
secs() { perl -MTime::HiRes=time -e 'print time'; }
run_timed() {
  local envs="$1" prompt="$2" mt="$3" t0 t1
  t0=$(secs); env QWISP_DEVICE_RAM=8 $envs "$BIN" chat --max-tokens "$mt" "$prompt" >/dev/null 2>&1; t1=$(secs)
  perl -e "print $t1-$t0"
}
for rep in 1 2 3; do
  for cfg in generic mixed128; do
    if [ "$cfg" = mixed128 ]; then envs="QWISP_MIXED=1 QWISP_EXPERTS_2BIT=$CAL128"; else envs="QWISP_MIXED=0"; fi
    for pname in qs tcp; do
      [ "$pname" = qs ] && prompt="$QS" || prompt="$TCP"
      t5=$(run_timed "$envs" "$prompt" 5)
      t505=$(run_timed "$envs" "$prompt" 505)
      tps=$(perl -e "printf '%.1f', 500/($t505-$t5)")
      echo "SPEED2 rep$rep $cfg $pname: decode=${tps} tok/s (t5=${t5} t505=${t505})"
    done
  done
done
echo "SPEED2 DONE"
