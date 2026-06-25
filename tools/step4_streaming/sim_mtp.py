#!/usr/bin/env python3
"""Qwisp Step 4 Stage A — MTP × streaming の net_tps(D, B) シミュレーション.

問い: MTP 投機デコード（depth D）は expert streaming と同時に使えて、実際に速くなるか。
verify は K=D+1 トークンを1パスで処理 → 各層で (D+1) トークンの expert 和集合に触れる
（AR の top_k=8 に対し最大 (D+1) 倍）。これが cold miss を増やし、MTP の amortize 利得と
綱引きになる。エンジンを作らず trace で net_tps(D,B) を出す。

モデル（OUR マシン単位で一貫）:
- 全載り verify 時間 T_compute(D) = accepted(D) / (baseline_tok_s × mult(D))。
  baseline = OUR 実測 AR（既定 54 tok/s）。mult・acceptance は MTPLX 実測（全載り）から graft。
- streaming ペナルティ T_flash(D,B) = (verify 窓の union-miss 数) × expert_bytes / flash_bw。
  prefill で LRU を温めてから decode を window=D+1 で処理。OUR 実測 flash（既定 4.18GB/s）。
- prefetch 2ブラケット:
    serial (prefetch off): T_verify = T_compute + T_flash
    overlap(prefetch on ideal): T_verify = max(T_compute, T_flash)   # HOBBIT 流上限
- net_tps(D,B) = accepted(D) / T_verify。

使い方:
    python3 sim_mtp.py --trace ../step1_routing_trace/traces.bench.jsonl \
        --budgets 32,48,64,96,128,256 --baseline-tok-s 54 \
        --flash-bw 4179784992 --expert-bytes 1769000
"""

import argparse
import json
import sys
from collections import OrderedDict, defaultdict

# MTPLX 実測（全載り、mtplx_runtime.json）: per-position acceptance と AR比 multiplier。
ACCEPT = {0: [], 1: [0.886], 2: [0.870, 0.641], 3: [0.829, 0.541, 0.278]}
MULT = {0: 1.0, 1: 138.4 / 94.5, 2: 135.7 / 94.5, 3: 107.7 / 94.5}


def accepted_per_verify(D):
    """期待受理トークン数 = 1（bonus）+ Σ 累積受理確率。"""
    acc, cum, total = ACCEPT[D], 1.0, 1.0
    for p in acc:
        cum *= p
        total += cum
    return total


def load_decode_by_layer(path):
    """{prompt: {layer: ([prefill experts...], [decode experts...])}}（token順）。"""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw[r["prompt_id"]][r["layer_idx"]].append(
                (r["token_idx"], r.get("phase", "decode"), tuple(r["expert_ids"])))
    out = {}
    for p, layers in raw.items():
        out[p] = {}
        for L, seq in layers.items():
            seq.sort(key=lambda x: x[0])
            pre = [e for (_t, ph, e) in seq if ph == "prefill"]
            dec = [e for (_t, ph, e) in seq if ph == "decode"]
            out[p][L] = (pre, dec)
    return out


def sim(by_prompt, B, D):
    """verify 窓ごとの union-miss を数える。返り値: (avg_miss_per_verify, n_layers)。"""
    window = D + 1
    total_miss = 0
    n_verifies = 0  # verify forward 数（全層を1回と数える）
    n_layers = 0
    for layers in by_prompt.values():
        n_layers = len(layers)
        # この prompt の verify forward 数（層によらず decode 長で決まる）
        any_dec = next(iter(layers.values()))[1]
        n_win = (len(any_dec) + window - 1) // window
        n_verifies += n_win
        for pre, dec in layers.values():
            cache = OrderedDict()
            for ex in pre:                    # prefill で温める（AR）
                for e in ex:
                    if e in cache:
                        cache.move_to_end(e)
                    else:
                        if len(cache) >= B:
                            cache.popitem(last=False)
                        cache[e] = None
            for i in range(0, len(dec), window):  # decode を窓処理
                union = set()
                for ex in dec[i:i + window]:
                    union.update(ex)
                for e in union:
                    if e in cache:
                        cache.move_to_end(e)
                    else:
                        total_miss += 1
                        if len(cache) >= B:
                            cache.popitem(last=False)
                        cache[e] = None
    return (total_miss / n_verifies if n_verifies else 0.0), n_layers


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step4 MTP×streaming sim")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--budgets", default="32,48,64,96,128,256")
    ap.add_argument("--depths", default="0,1,2,3")
    ap.add_argument("--baseline-tok-s", type=float, default=54.0, help="OUR 実測 AR decode")
    ap.add_argument("--flash-bw", type=float, default=4.179784992e9)
    ap.add_argument("--expert-bytes", type=float, default=1769000.0)
    args = ap.parse_args()

    by_prompt = load_decode_by_layer(args.trace)
    budgets = [int(b) for b in args.budgets.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    print(f"[mtp] baseline={args.baseline_tok_s}tok/s flash={args.flash_bw/1e9:.2f}GB/s "
          f"expert={args.expert_bytes/1e6:.3f}MB", file=sys.stderr)
    print(f"[mtp] full-residency 上限 tok/s: " +
          ", ".join(f"D{d}={args.baseline_tok_s*MULT[d]:.1f}(acc {accepted_per_verify(d):.2f})"
                    for d in depths), file=sys.stderr)

    hdr = (f"{'D':>2} {'B/層':>5} {'DRAM':>6} {'miss/verify':>11} "
           f"{'Tcomp':>7} {'Tflash':>7} {'net_serial':>10} {'net_overlap':>11}")
    print(hdr)
    print("-" * len(hdr))
    best = {}
    for B in budgets:
        for d in depths:
            acc = accepted_per_verify(d)
            t_compute = acc / (args.baseline_tok_s * MULT[d])
            miss_pv, n_layers = sim(by_prompt, B, d)
            t_flash = miss_pv * args.expert_bytes / args.flash_bw
            net_serial = acc / (t_compute + t_flash)
            net_overlap = acc / max(t_compute, t_flash)
            dram = B * n_layers * args.expert_bytes / 1e9
            print(f"{d:>2} {B:>5} {dram:>5.1f}G {miss_pv:>11.1f} "
                  f"{t_compute*1000:>6.1f}ms {t_flash*1000:>6.1f}ms "
                  f"{net_serial:>10.1f} {net_overlap:>11.1f}")
            best.setdefault(B, []).append((d, net_serial, net_overlap))
        print()

    print("[mtp] 各 B での最良 depth（net_serial / net_overlap）:", file=sys.stderr)
    for B in budgets:
        bs = max(best[B], key=lambda x: x[1])
        bo = max(best[B], key=lambda x: x[2])
        print(f"  B={B:>3}: serial→D{bs[0]}({bs[1]:.1f}tps)  overlap→D{bo[0]}({bo[2]:.1f}tps)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
