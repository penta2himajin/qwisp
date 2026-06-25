#!/usr/bin/env python3
"""Qwisp Step 4 Stage B — MTP × mixed-precision streaming の net_tps(D, RAM).

問い: MTP は max-reach(flash-bound) regime では損だった（Stage A, docs/06）。理由は
verify 窓 D+1 の union-miss が accepted より速く増え、**accepted あたり flash 仕事が D で増加**、
flash 帯域が硬い上限になるから。mixed-precision はこの致命点を二重に攻める:
  (1) cold miss が 1.77MB→0.98MB に半減（per-miss IO 削減）
  (2) 同 RAM に ~1.8倍の expert が載る（reach↑→miss 数↓）
→ max-reach regime でも MTP D1 が復活するか？を **byte 予算**の trace-sim で判定。

Stage A（sim_mtp.py）は count 予算・全 4bit・flat expert_bytes。ここは byte 予算で
all4 と mixed(hot 4bit / cold 2bit) を同 RAM(GB) 比較し、各 RAM で最良 depth を出す。

モデルは Stage A と一貫（MTPLX graft の mult/acceptance、serial/overlap 2ブラケット）。
miss コストだけ「数×flat」→「Σ 実バイト(hot1.77/cold0.98)」に置換。

使い方:
    python3 sim_mtp_mixed.py --trace ../step1_routing_trace/traces.bench.jsonl \
        --budgets-gb 3,4,6,8,12 --hot 64 --baseline-tok-s 54 --flash-bw 4.179784992e9
"""

import argparse
import sys
from collections import OrderedDict, defaultdict

from sim_mtp import load_decode_by_layer, accepted_per_verify, MULT

B4 = 1769472   # 4bit per-expert bytes（実測）
B2 = 983040    # 2bit per-expert bytes（実測）


def expert_freq(by_prompt):
    """{layer: {expert: count}}（pre+dec 全 phase）。hot 集合判定用。"""
    freq = defaultdict(lambda: defaultdict(int))
    for layers in by_prompt.values():
        for L, (pre, dec) in layers.items():
            for ex in pre:
                for e in ex:
                    freq[L][e] += 1
            for ex in dec:
                for e in ex:
                    freq[L][e] += 1
    return freq


def sim_bytes(by_prompt, budget_bytes, D, hot_by_layer, mixed):
    """verify 窓ごとの union-miss を **バイト**で集計。返り値: (miss_bytes_per_verify, n_layers).

    budget_bytes = 1層あたりの byte 予算。mixed=False は全 4bit（hot 無視）。
    """
    window = D + 1
    total_miss_bytes = 0
    n_verifies = 0
    n_layers = 0

    def size(e, hot):
        if not mixed:
            return B4
        return B4 if e in hot else B2

    for layers in by_prompt.values():
        n_layers = len(layers)
        any_dec = next(iter(layers.values()))[1]
        n_verifies += (len(any_dec) + window - 1) // window
        for L, (pre, dec) in layers.items():
            hot = hot_by_layer.get(L, set())
            cache = OrderedDict()
            used = 0

            def admit(e):
                nonlocal used
                need = size(e, hot)
                while used + need > budget_bytes and cache:
                    vk, _ = next(iter(cache.items()))
                    used -= size(vk, hot)
                    del cache[vk]
                cache[e] = None
                used += need

            for ex in pre:                    # prefill 温め（AR、protect 無し簡略）
                for e in ex:
                    if e in cache:
                        cache.move_to_end(e)
                    else:
                        admit(e)
            for i in range(0, len(dec), window):  # decode 窓
                union = set()
                for ex in dec[i:i + window]:
                    union.update(ex)
                for e in union:
                    if e in cache:
                        cache.move_to_end(e)
                    else:
                        total_miss_bytes += size(e, hot)
                        admit(e)
    return (total_miss_bytes / n_verifies if n_verifies else 0.0), n_layers


def net_for(by_prompt, budget_bytes, D, hot_by_layer, mixed, base_tps, flash_bw,
            compute_penalty=1.0):
    acc = accepted_per_verify(D)
    # mixed は two-gather の compute overhead を持つ（resident 実測 0.71x ≈ ×1.41）。
    # 融合カーネルなら penalty=1.0。flash-bound では効かないが overlap(compute-bound) で効く。
    t_compute = acc / (base_tps * MULT[D]) * (compute_penalty if mixed else 1.0)
    miss_bytes_pv, n_layers = sim_bytes(by_prompt, budget_bytes, D, hot_by_layer, mixed)
    t_flash = miss_bytes_pv / flash_bw
    return {
        "D": D, "acc": acc, "t_compute": t_compute, "t_flash": t_flash,
        "net_serial": acc / (t_compute + t_flash),
        "net_overlap": acc / max(t_compute, t_flash),
        "miss_mb_pv": miss_bytes_pv / 1e6, "n_layers": n_layers,
    }


def main():
    ap = argparse.ArgumentParser(description="Qwisp MTP × mixed-precision sim")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--budgets-gb", default="3,4,6,8,12", help="expert DRAM 総量 GB")
    ap.add_argument("--hot", type=int, default=64, help="hot(4bit) expert 数（品質 GREEN=64）")
    ap.add_argument("--depths", default="0,1,2,3")
    ap.add_argument("--baseline-tok-s", type=float, default=54.0)
    ap.add_argument("--flash-bw", type=float, default=4.179784992e9)
    ap.add_argument("--mixed-compute-penalty", type=float, default=1.41,
                    help="mixed の two-gather compute 係数（resident 実測 0.71x≈1.41。融合カーネル=1.0）")
    args = ap.parse_args()

    by_prompt = load_decode_by_layer(args.trace)
    freq = expert_freq(by_prompt)
    n_layers = len(next(iter(by_prompt.values())))
    hot_by_layer = {L: set(sorted(freq[L], key=lambda e: -freq[L][e])[:args.hot]) for L in freq}
    budgets = [float(x) for x in args.budgets_gb.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    print(f"[mtp-mix] layers={n_layers} hot={args.hot}@4bit/cold@2bit "
          f"baseline={args.baseline_tok_s} flash={args.flash_bw/1e9:.2f}GB/s "
          f"4bit={B4/1e6:.2f}MB 2bit={B2/1e6:.2f}MB", file=sys.stderr)
    print(f"[mtp-mix] full-res 上限: " +
          ", ".join(f"D{d}={args.baseline_tok_s*MULT[d]:.1f}(acc {accepted_per_verify(d):.2f})"
                    for d in depths), file=sys.stderr)

    hdr = (f"{'cfg':6} {'総GB':>5} {'D':>2} {'miss MB/vf':>10} "
           f"{'Tcomp':>7} {'Tflash':>7} {'net_serial':>10} {'net_overlap':>11} {'best?':>5}")
    print(hdr); print("-" * len(hdr))

    for gb in budgets:
        budget_bytes = gb * 1e9 / n_layers
        for cfg, mixed in (("all4", False), ("mixed", True)):
            rows = [net_for(by_prompt, budget_bytes, d, hot_by_layer, mixed,
                            args.baseline_tok_s, args.flash_bw,
                            args.mixed_compute_penalty) for d in depths]
            best_s = max(rows, key=lambda r: r["net_serial"])["D"]
            best_o = max(rows, key=lambda r: r["net_overlap"])["D"]
            for r in rows:
                tag = ("S" if r["D"] == best_s else " ") + ("O" if r["D"] == best_o else " ")
                print(f"{cfg:6} {gb:>5.1f} {r['D']:>2} {r['miss_mb_pv']:>9.1f} "
                      f"{r['t_compute']*1000:>6.1f}ms {r['t_flash']*1000:>6.1f}ms "
                      f"{r['net_serial']:>10.1f} {r['net_overlap']:>11.1f} {tag:>5}")
            print()


if __name__ == "__main__":
    main()
