#!/bin/bash
# lever-A speed A/B + soak (#47 / notes/18). GPU-exclusive.
# Speed: per config, chat with --max-tokens 5 then 505; decode tok/s = 500/(t505-t5)
# (load + one-time mixed calibration cancels in the difference).
# Soak: 8 diverse prompts x 1500 tok + 2 long (3000 tok) on mixed cov128; TOK_DUMP for loop scan.
set -u
cd "$(dirname "$0")"
BIN=../../../swift/.xcode-build-rel/Build/Products/Release/qwisp
CAL128=$HOME/.mtplx/models/qwisp-experts-2bit-cal128
QS="Write a detailed step-by-step explanation of how quicksort works, with a Python implementation."
TCP="Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery."

secs() { perl -MTime::HiRes=time -e 'print time'; }

run_timed() { # config-env-string, prompt, maxtok -> seconds on stdout
  local envs="$1" prompt="$2" mt="$3"
  local t0 t1
  t0=$(secs)
  env QWISP_DEVICE_RAM=8 $envs "$BIN" chat --max-tokens "$mt" "$prompt" >/dev/null 2>&1
  t1=$(secs)
  perl -e "print $t1-$t0"
}

echo "== SPEED A/B (500-token decode window) =="
for cfg in generic mixed128; do
  if [ "$cfg" = mixed128 ]; then envs="QWISP_MIXED=1 QWISP_EXPERTS_2BIT=$CAL128"; else envs="QWISP_MIXED=0"; fi
  for pname in qs tcp; do
    [ "$pname" = qs ] && prompt="$QS" || prompt="$TCP"
    t5=$(run_timed "$envs" "$prompt" 5)
    t505=$(run_timed "$envs" "$prompt" 505)
    tps=$(perl -e "printf '%.1f', 500/($t505-$t5)")
    echo "SPEED $cfg $pname: t5=${t5}s t505=${t505}s decode=${tps} tok/s"
  done
done

echo "== SOAK (mixed cov128) =="
soak() { # tag, prompt, maxtok
  local tag="$1" prompt="$2" mt="$3"
  [ -s "soak-$tag.toks" ] && { echo "[soak] $tag exists, skip"; return; }
  QWISP_MIXED=1 QWISP_DEVICE_RAM=8 QWISP_EXPERTS_2BIT=$CAL128 \
    QWISP_TOK_DUMP="soak-$tag.toks" "$BIN" chat --max-tokens "$mt" "$prompt" \
    >"soak-$tag.txt" 2>"soak-$tag.err"
  echo "[soak] $tag rc=$? toks=$(wc -l < "soak-$tag.toks" 2>/dev/null | tr -d ' ')"
}
soak lru   "Write a Python class implementing an LRU cache with O(1) operations, then explain the design." 1500
soak rev   "Summarize the causes and consequences of the French Revolution in detail." 1500
soak gc    "Explain how garbage collection works in modern JavaScript engines." 1500
soak bash  "Write a bash script that monitors a directory and syncs changes to a backup folder, with comments explaining each part." 1500
soak cap   "Explain the CAP theorem and its practical implications for distributed databases." 1500
soak scifi "Write a short sci-fi story about a translator AI on a first-contact mission." 1500
soak jp    "量子コンピュータの仕組みを、高校生にも分かるように日本語で詳しく説明してください。" 1500
soak bs    "Explain step by step how to implement binary search correctly, including common off-by-one pitfalls, with code." 1500
soak story3000 "Write a short story about a lighthouse keeper who discovers a message in a bottle." 3000
soak qs3000    "$QS" 3000
echo "SPEED_SOAK DONE"
