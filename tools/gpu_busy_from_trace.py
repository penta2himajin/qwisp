#!/usr/bin/env python3
"""issue#3 §5: Metal System Trace から forward の GPU-busy 占有率を算出。

dispatch/sync 律速か GPU-exec 床かを二値判定する。`metal-gpu-intervals`(GPU 実行区間)を
process=qwisp・depth=0 で抽出し、区間 union / span = GPU 占有率を出す。GPU が大きく idle
(≲50%)なら 50ms 床は GPU-exec でなく dispatch/sync(per-layer routing round-trip)律速。

使い方:
  # 1) forward ループを長く回しつつ Metal System Trace を記録
  xcrun xctrace record --template 'Metal System Trace' --output /tmp/qwisp_gpu.trace \
    --env QWISP_RUN=forward-gpu-busy --env QWISP_GPUBUSY_K=1 --env QWISP_FC_REPS=200 \
    --env QWISP_MODEL=... --env QWISP_MTP_REF=/tmp/qwisp_mtp_ref.safetensors \
    --launch -- <qwisp-poc> stream
  # 2) GPU 区間を export
  xcrun xctrace export --input /tmp/qwisp_gpu.trace \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
    --output /tmp/qwisp_gpuint.xml
  # 3) 占有率を算出
  python3 tools/gpu_busy_from_trace.py /tmp/qwisp_gpuint.xml [qwisp]

実測(2026-06-28, M1 Max): GPU-busy ≈ 34.6%(steady median 33.8%) ＝ GPU は ~65% idle
→ dispatch/sync 律速確定。CPU-busy ≈0.51 cores(probe forward-gpu-busy)と整合。
"""
import re, sys, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwisp_gpuint.xml"
needle = sys.argv[2] if len(sys.argv) > 2 else "qwisp"
data = open(path, "r", errors="replace").read()

# xctrace export は値を id で intern し、以降は ref で参照する。先に id->値の表を作る。
def build(tag):
    return {int(i): int(v) for i, v in re.findall(r'<%s id="(\d+)"[^>]*>(\d+)</%s>' % (tag, tag), data)}

startmap = build("start-time")
depthmap = build("metal-nesting-level")
durmap = build("duration")               # col2(exec duration); row 内の最初の <duration>
procmap = {int(i): f for i, f in re.findall(r'<process id="(\d+)" fmt="([^"]*)"', data)}

ivals = []
total = 0
for r in re.finditer(r"<row>(.*?)</row>", data, re.S):
    body = r.group(1); total += 1
    pm = re.search(r'<process (?:id|ref)="(\d+)"', body)
    proc = procmap.get(int(pm.group(1))) if pm else None
    if not proc or needle not in proc:
        continue
    dm = re.search(r'<metal-nesting-level (?:id|ref)="(\d+)"', body)
    if not dm or depthmap.get(int(dm.group(1))) != 0:
        continue
    sm = re.search(r'<start-time (?:id|ref)="(\d+)"', body)
    um = re.search(r'<duration (?:id|ref)="(\d+)"', body)
    s = startmap.get(int(sm.group(1))) if sm else None
    d = durmap.get(int(um.group(1))) if um else None
    if s is None or d is None:
        continue
    ivals.append((s, s + d))

print(f"rows={total}  {needle} depth0 GPU intervals={len(ivals)}")
if not ivals:
    sys.exit("no intervals — wrong process name? try arg2")

ivals.sort()
def union(iv):
    tot = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    return tot + (ce - cs)

span0 = ivals[0][0]; span1 = max(e for _, e in ivals); span = span1 - span0
busy = union(ivals)
print(f"GPU-active span={span/1e6:.0f}ms  union GPU-busy={busy/1e6:.0f}ms  overall occupancy={busy/span*100:.1f}%")

binw = 10_000_000
bins = defaultdict(list)
for s, e in ivals:
    bins[(s - span0) // binw].append((s, e))
occ = sorted(min(union(sorted(v)), binw) / binw for v in bins.values())
busy_bins = [o for o in occ if o > 0.05]
if busy_bins:
    print(f"steady GPU occupancy /10ms: median={st.median(busy_bins)*100:.1f}%  "
          f"p90={busy_bins[int(len(busy_bins)*0.9)]*100:.1f}%  max={max(busy_bins)*100:.1f}%")
print(f"mean interval dur={st.mean(e-s for s,e in ivals)/1e3:.1f}us")
verdict = ("DISPATCH/SYNC 律速 (GPU 大きく idle)" if busy/span < 0.5
           else "GPU-EXEC 床 (GPU ほぼ飽和)" if busy/span >= 0.8 else "混在")
print("判定:", verdict)
