#!/usr/bin/env bash
# Static gate for the GPU failure firewall (#169).
#
# A bare `waitUntilCompleted()` means the caller does not know whether the GPU
# actually ran. Reading an output buffer after a failed command buffer yields
# its zero-initialised contents, which is how a GPU OOM became an unbounded run
# of token id 0 with no error anywhere. Sites that read results MUST go through
# SeedlessFusedVerify.commitAndWaitChecked.
#
# This is a ratchet, not a clean bill of health: the baseline below is the count
# that existed when the firewall landed. It may only go DOWN. Adding a bare wait
# fails the gate; converting one to a checked commit means lowering the number.
#
# ponytail: a per-file baseline, not a real dataflow analysis. It cannot tell a
# fire-and-forget blit from a decode readback — it only stops the count growing.
# Upgrade path if that stops being enough: mark intentional bare waits with a
# `// cb-unchecked: <reason>` comment and gate on unmarked ones instead.
set -uo pipefail
cd "$(dirname "$0")/.."

BASELINE=89

count=$(grep -rn "waitUntilCompleted" --include='*.swift' swift/Sources \
        | grep -vc "commitAndWaitChecked" || true)

echo "[cb-guard] bare waitUntilCompleted sites: $count (baseline $BASELINE)"

if [ "$count" -gt "$BASELINE" ]; then
  echo "[cb-guard] FAIL: $((count - BASELINE)) new unchecked command buffer wait(s)."
  echo "[cb-guard] Route result-reading commits through SeedlessFusedVerify.commitAndWaitChecked."
  echo "[cb-guard] New sites:"
  grep -rn "waitUntilCompleted" --include='*.swift' swift/Sources | grep -v "commitAndWaitChecked"
  exit 1
fi

if [ "$count" -lt "$BASELINE" ]; then
  echo "[cb-guard] NOTE: down from the baseline — lower BASELINE in this script to $count to keep the ratchet tight."
fi

echo "CBGUARD PASS"
