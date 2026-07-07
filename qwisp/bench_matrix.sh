#!/usr/bin/env bash
# Deterministic full environment matrix (E1-E7) — no agents, no input variance.
# Replaces the agent-driven qwisp-full-matrix workflow (haiku agents mishandled
# background execution: benches were launched bg, agents returned early, processes
# got killed / overlapped on the GPU — 2026-07-07 incident).
#
# Runs the 7 canonical configs SEQUENTIALLY via bench_batch.sh, then applies the
# deterministic gate (the old sonnet-audit checklist, mechanized):
#   1. completeness  — E1-E6: 8 rows each (2 methods x 4 regimes); E7: 4 rows; EXIT=0
#   2. strict gate   — every suffix-spec fidelity == 100.0%
#   3. determinism   — fidelity identical fast vs slow at same C (per method x regime)
#   4. speed sanity  — strict fast >= slow (WARN only; bolt is io=0 ~throttle-flat)
#   5. correctness   — list any FAIL rows verbatim
#
# Usage: qwisp/bench_matrix.sh [GEN]          run everything + gate  (default GEN=128)
#        qwisp/bench_matrix.sh --check [DIR]  gate existing logs only (default DIR from env)
# Env:   QWISP_MATRIX_DIR — log dir (default /tmp). Logs: $DIR/qwisp-fm-<tag>.log
# Exit:  0 iff gate PASS. Final line: MATRIXGATE PASS|FAIL.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${QWISP_MATRIX_DIR:-/tmp}"

# tag|C|throttle|methods  (E7 resident = strict only)
CONFIGS=(
  "E1-C64-fast|64|0|suffix-spec bolt"
  "E2-C64-slow|64|1.5|suffix-spec bolt"
  "E3-C128-fast|128|0|suffix-spec bolt"
  "E4-C128-slow|128|1.5|suffix-spec bolt"
  "E5-C192-fast|192|0|suffix-spec bolt"
  "E6-C192-slow|192|1.5|suffix-spec bolt"
  "E7-C256-resident|256|0|suffix-spec"
)

if [ "${1:-}" = "--check" ]; then
  DIR="${2:-$DIR}"
else
  GEN="${1:-128}"
  if pgrep -f qwisp-poc > /dev/null; then
    echo "ERROR: qwisp-poc already running (GPU exclusive)"; exit 1
  fi
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag c thr methods <<< "$cfg"
    log="$DIR/qwisp-fm-$tag.log"
    echo "== $tag (C=$c thr=$thr) -> $log =="
    # DEFER only for slow configs' strict batch (bench_batch strips it for raw bolt calls).
    if [ "$thr" != 0 ]; then
      QWISP_THROTTLE_DEFER=1 "$REPO/qwisp/bench_batch.sh" "$c" "$GEN" "$thr" "$methods" > "$log" 2>&1
    else
      "$REPO/qwisp/bench_batch.sh" "$c" "$GEN" "$thr" "$methods" > "$log" 2>&1
    fi
    echo "EXIT=$?" >> "$log"
    tail -n +2 "$log" | grep -E '^  (suffix-spec|bolt) +(code|agentic|longctx|shortnl) ' || true
  done
fi

# ── deterministic gate ─────────────────────────────────────────────────────────
fail=0; warn=0
ROWS="$(mktemp)"   # tag method regime tokps fid
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag c thr methods <<< "$cfg"
  log="$DIR/qwisp-fm-$tag.log"
  [ -f "$log" ] || { echo "GATE FAIL [$tag] missing log $log"; fail=1; continue; }
  grep -q '^EXIT=0$' "$log" || { echo "GATE FAIL [$tag] EXIT=0 absent (crashed or killed)"; fail=1; }
  # data rows (regime whitelist excludes the aggregate 'bolt speed …' lines)
  grep -E '^  (suffix-spec|bolt) +(code|agentic|longctx|shortnl) ' "$log" \
    | awk -v t="$tag" '{ sub(/%$/, "", $4); print t, $1, $2, $3, $4 }' >> "$ROWS"
  want=$(( $(echo "$methods" | wc -w) * 4 ))
  got="$(awk -v t="$tag" '$1 == t' "$ROWS" | wc -l | tr -d ' ')"
  [ "$got" -eq "$want" ] || { echo "GATE FAIL [$tag] rows $got/$want"; fail=1; }
  # correctness FAILs verbatim
  grep -E '^  (suffix-spec|bolt) +(code|agentic|longctx|shortnl) .*FAIL' "$log" \
    | while IFS= read -r l; do echo "GATE FAIL [$tag] correctness: $l"; done
  if grep -qE '^  (suffix-spec|bolt) +(code|agentic|longctx|shortnl) .*FAIL' "$log"; then fail=1; fi
done

# strict fidelity == 100.0 everywhere
while read -r tag m r tokps fid; do
  [ "$m" = suffix-spec ] && [ "$fid" != "100.0" ] \
    && { echo "GATE FAIL [$tag] strict $r fidelity $fid% != 100.0%"; fail=1; }
done < "$ROWS"

# determinism (GATE): fidelity identical fast vs slow at same C, per method x regime
# speed sanity (WARN): strict fast >= slow per regime
for pair in "E1-C64-fast E2-C64-slow" "E3-C128-fast E4-C128-slow" "E5-C192-fast E6-C192-slow"; do
  set -- $pair
  d="$(awk -v f="$1" -v s="$2" '
        $1 == f { ffid[$2 " " $3] = $5 }
        $1 == s { sfid[$2 " " $3] = $5 }
        END { for (k in sfid) if (k in ffid && ffid[k] != sfid[k])
                printf "  %s: fast %s%% != slow %s%%\n", k, ffid[k], sfid[k] }' "$ROWS")"
  [ -z "$d" ] || { echo "GATE FAIL determinism $1 vs $2:"; echo "$d"; fail=1; }
  w="$(awk -v f="$1" -v s="$2" '
        $1 == f && $2 == "suffix-spec" { fast[$3] = $4 }
        $1 == s && $2 == "suffix-spec" { slow[$3] = $4 }
        END { for (r in slow) if (r in fast && fast[r] + 0 < slow[r] + 0)
                printf "  strict %s: fast %s < slow %s tok/s\n", r, fast[r], slow[r] }' "$ROWS")"
  [ -z "$w" ] || { echo "GATE WARN speed sanity $1 vs $2:"; echo "$w"; warn=1; }
done

echo "== matrix summary (tag method regime tok/s fid%) =="
column -t "$ROWS" | sed 's/^/  /'
rm -f "$ROWS"
[ "$warn" -eq 1 ] && echo "(warnings above are non-gating)"
if [ "$fail" -eq 0 ]; then echo "MATRIXGATE PASS"; else echo "MATRIXGATE FAIL"; exit 1; fi
