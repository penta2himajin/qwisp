#!/usr/bin/env python3
"""Qwisp Step 2.5 — mixed-precision expert キャッシュ go/no-go（制約 RAM regime）。

simulate.py は expert を**個数**で予算化（全 expert 同サイズ＝4bit 前提）。mixed-precision
では hot=4bit / cold=2bit でサイズが違うので、ここは **バイト予算（GB/層）**でシミュレートする。

比較する 2 方策（同じ RAM 予算 GB/層）:
  - all4 : 全 expert 4bit。LRU。miss → 4bit を fetch（1.769MB）。
  - mixed: 静的 hot 集合（trace 頻度 top-B_hot）は 4bit、その他 cold は 2bit でキャッシュ。
           同 RAM により多くの cold が載る（reach↑）。miss → hot は 4bit, cold は 2bit を
           fetch（cold は IO 半減）。品質は別途 GREEN 実証済（hot128@4/cold128@2bit=48/48）。

mixed の狙い＝**reach レバー**: budget<全載り の制約 regime で、
  net_tps = 1/(1/baseline + miss_bytes_per_token ÷ flash_bw)
が all4 を上回るか（=GO）を判定。resident 全載り regime では mixed は overhead だけ（実証済 0.71x）。

注意（適用範囲）: この式は **IO のみ**をモデル化し、mixed の two-gather compute overhead は
含まない。miss が支配的な制約 regime（数GB）では IO 削減が支配し妥当だが、高 RAM 端
（miss≈0）では overhead が効くため mixed の net_tps を**過大評価**する。判定は制約 regime を見る。
予算 `--budgets-gb` は **expert DRAM の総量**（非expert 常駐 ~1.8GB + KV は別途）。

使い方:
  python simulate_mixed.py --trace ../step1_routing_trace/traces.bench.jsonl \
    --budgets-gb 2,3,4,6,8 --hot 64,128 --baseline-tok-s 54 --flash-bw 4.0e9 --target-tok-s 15
"""

import argparse
import sys
from collections import OrderedDict, defaultdict

from simulate import load_trace, infer_dims

INF = float("inf")
B4 = 1769472  # 4bit per-expert bytes（実測, layer3/expert0）
B2 = 983040   # 2bit per-expert bytes（実測; weight 半分・scales/biases 同）


def expert_freq(by_prompt, n_layers):
    """層ごとの expert 出現頻度（全 prompt・全 phase）→ {layer: {expert: count}}。"""
    freq = defaultdict(lambda: defaultdict(int))
    for layers in by_prompt.values():
        for L, seq in layers.items():
            for _t, _p, experts in seq:
                for e in experts:
                    freq[L][e] += 1
    return freq


def hot_set(freq_layer, b_hot):
    """頻度上位 b_hot を hot（4bit）に。"""
    return set(sorted(freq_layer, key=lambda e: -freq_layer[e])[:b_hot])


def sim_layer_bytes(seq, budget_bytes, hot, mixed, tally):
    """1 prompt・1 層を byte 予算 LRU でシミュレート。phase 別 hit/miss と miss バイトを tally。

    mixed=False: 全 expert 4bit。mixed=True: hot=4bit, cold=2bit。
    現トークンの top_k は evict 対象外（同時常駐が必要）。
    """
    def size(e):
        if not mixed:
            return B4
        return B4 if e in hot else B2

    cache = OrderedDict()  # expert -> None（挿入順=LRU）
    used = 0
    for _tok, phase, experts in seq:
        protected = set(experts)
        for e in experts:
            if e in cache:
                tally[(phase, "hit")] += 1
                cache.move_to_end(e)
            else:
                tally[(phase, "miss")] += 1
                tally[(phase, "miss_bytes")] += size(e)
                need = size(e)
                # evict（LRU、protected は除く）まで容量確保
                while used + need > budget_bytes and len(cache) > 0:
                    victim = None
                    for cand in cache:  # 先頭=最古
                        if cand not in protected:
                            victim = cand
                            break
                    if victim is None:
                        break  # 全部 protected（budget が top_k 未満）→ 諦めて入れる
                    used -= size(victim)
                    del cache[victim]
                cache[e] = None
                used += need


def run(by_prompt, budget_bytes, hot_by_layer, mixed):
    tally = defaultdict(int)
    for layers in by_prompt.values():
        for L, seq in layers.items():
            sim_layer_bytes(seq, budget_bytes, hot_by_layer.get(L, set()), mixed, tally)
    return tally


def decode_metrics(tally, active_per_token, flash_bw, base_s):
    h = tally[("decode", "hit")]
    m = tally[("decode", "miss")]
    acc = h + m
    miss_rate = (m / acc) if acc else 0.0
    # miss_bytes_per_token = 総 decode miss バイト / decode トークン数
    #   decode トークン数 = acc / active_per_token
    n_tok = acc / active_per_token if active_per_token else 0
    miss_bytes_per_tok = (tally[("decode", "miss_bytes")] / n_tok) if n_tok else 0.0
    lat = miss_bytes_per_tok / flash_bw
    net = (1.0 / (base_s + lat)) if (base_s + lat) > 0 else INF
    return miss_rate, miss_bytes_per_tok, lat, net


def main():
    ap = argparse.ArgumentParser(description="Qwisp mixed-precision cache go/no-go")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--budgets-gb", default="2,3,4,6,8", help="RAM 予算 GB/層（総 expert DRAM）")
    ap.add_argument("--hot", default="64,128", help="hot(4bit) expert 数のスイープ")
    ap.add_argument("--flash-bw", type=float, default=4.0e9, help="flash 帯域 B/s（Step3 実測 4.0GB/s）")
    ap.add_argument("--baseline-tok-s", type=float, default=54.0, help="素の AR decode tok/s（Step3）")
    ap.add_argument("--target-tok-s", type=float, default=15.0)
    args = ap.parse_args()

    by_prompt = load_trace(args.trace)
    n_layers, top_k = infer_dims(by_prompt)
    active = top_k * n_layers
    base_s = 1.0 / args.baseline_tok_s
    freq = expert_freq(by_prompt, n_layers)
    budgets = [float(x) for x in args.budgets_gb.split(",")]
    hots = [int(x) for x in args.hot.split(",")]

    print(f"[mix-sim] layers={n_layers} top_k={top_k} active/tok={active} prompts={len(by_prompt)} "
          f"baseline={args.baseline_tok_s} flash={args.flash_bw/1e9:.1f}GB/s "
          f"4bit={B4/1e6:.2f}MB 2bit={B2/1e6:.2f}MB", file=sys.stderr)
    hdr = f"{'cfg':18} {'総GB':>6} {'#4b/#2b/層':>11} {'miss(dec)':>9} {'MB/tok':>7} {'+lat':>7} {'net_tps':>8} {'gate':>5}"
    print(hdr); print("-" * len(hdr))

    for gb in budgets:
        budget_bytes = gb * 1e9 / n_layers  # per-layer byte 予算
        # --- all4 baseline ---
        t = run(by_prompt, budget_bytes, {}, mixed=False)
        mr, mbpt, lat, net = decode_metrics(t, active, args.flash_bw, base_s)
        n4 = int(budget_bytes // B4)
        gate = "GO" if net >= args.target_tok_s else "no"
        print(f"{'all4':18} {gb:>6.1f} {n4:>5}/{0:<5} {mr:>9.3f} {mbpt/1e6:>6.2f} {lat*1000:>6.1f}ms {net:>7.1f} {gate:>5}")
        # --- mixed (sweep hot) ---
        for bh in hots:
            hot_by_layer = {L: hot_set(freq[L], bh) for L in freq}
            t = run(by_prompt, budget_bytes, hot_by_layer, mixed=True)
            mr, mbpt, lat, net = decode_metrics(t, active, args.flash_bw, base_s)
            # 概算: hot 全載り後の残予算で cold(2bit)
            n2 = int(max(0, budget_bytes - bh * B4) // B2)
            gate = "GO" if net >= args.target_tok_s else "no"
            print(f"{'mixed h'+str(bh):18} {gb:>6.1f} {bh:>5}/{n2:<5} {mr:>9.3f} {mbpt/1e6:>6.2f} {lat*1000:>6.1f}ms {net:>7.1f} {gate:>5}")
        print()


if __name__ == "__main__":
    main()
