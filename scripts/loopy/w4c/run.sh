#!/bin/bash
# W4c battery (#47 / notes/18): calibrated 2-bit tail artifact, arms K4=8 (cov 108)
# and K4=0 (cov 115). Free-run + TOK_DUMP -> detlag2/rollstab verdicts -> strict
# TF replay (realized-flip fidelity). bash-3.2-safe. GPU-exclusive: stop qwisp first.
set -u
cd "$(dirname "$0")"
BIN=../../../swift/.xcode-build-rel/Build/Products/Release/qwisp
CAL=$HOME/.mtplx/models/qwisp-experts-2bit-cal
PY=$HOME/.venvs/mlx/bin/python3
NAMES="story tcp qs sky"

prompt_for() {
  case $1 in
    story) echo "Write a short story about a lighthouse keeper who discovers a message in a bottle.";;
    tcp)   echo "Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery.";;
    qs)    echo "Write a detailed step-by-step explanation of how quicksort works, with a Python implementation.";;
    sky)   echo "Explain why the sky appears blue in plain English, in about three paragraphs.";;
  esac
}

for k4 in 8 0; do
  for n in $NAMES; do
    tag="c${k4}-${n}"
    [ -s "$tag.toks" ] && { echo "[w4c] $tag exists, skip"; continue; }
    echo "[w4c] free-run $tag start"
    QWISP_MIXED=1 QWISP_DEVICE_RAM=8 QWISP_EXPERTS_2BIT="$CAL" QWISP_MIX_K4=$k4 \
      QWISP_TOK_DUMP="$tag.toks" "$BIN" chat --max-tokens 1500 "$(prompt_for "$n")" \
      >"$tag.txt" 2>"$tag.err"
    # rc 133/139 tolerated: MLX teardown race at exit, dump already complete
    echo "[w4c] free-run $tag rc=$? toks=$(wc -l < "$tag.toks" 2>/dev/null | tr -d ' ')"
  done
done

for k4 in 8 0; do
  for n in $NAMES; do
    tag="c${k4}-${n}"
    [ -s "$tag.toks" ] || { echo "[w4c] $tag: no toks (early-EOS?), skip TF"; continue; }
    [ -s "$tag.tf.tsv" ] && continue
    echo "[w4c] TF replay $tag"
    QWISP_TF_REPLAY="$tag.toks" QWISP_TF_OUT="$tag.tf.tsv" \
      "$BIN" chat --lossless --max-tokens 1600 "$(prompt_for "$n")" >/dev/null 2>"$tag.tf.err"
    echo "[w4c] TF $tag rc=$?"
  done
done

for k4 in 8 0; do
  echo "=== K4=$k4 ==="
  $PY ../detlag2.py c${k4}-story.toks c${k4}-tcp.toks c${k4}-qs.toks c${k4}-sky.toks
  $PY ../rollstab.py c${k4}-story.txt c${k4}-tcp.txt c${k4}-qs.txt c${k4}-sky.txt
  for n in $NAMES; do
    f="c${k4}-${n}.tf.tsv"
    [ -s "$f" ] && awk -F'\t' -v n="$n" 'NR>1{m+=$4; c++} END{if(c) printf "TF %s: %d/%d = %.2f%%\n", n, m, c, 100*m/c}' "$f"
  done
done
echo "W4C BATTERY DONE"
