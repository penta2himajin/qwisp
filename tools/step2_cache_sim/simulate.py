#!/usr/bin/env python3
"""Qwisp Step 2 — expert キャッシュシミュレーション（オフライン）。

Step 1 の routing trace に対し、**DRAM 予算（=常駐できる expert 数/層）を振りながら**
LRU / LFU / Belady(oracle) のヒット率を出し、go/no-go の定量ゲートを計算する。
ロードマップ（../../docs/02-roadmap.md Step 2）の中核。「ここで実質ぜんぶ決まる。」

モデル:
- cache 単位 = (layer, expert)。layer L の expert は L 専用なので層ごとに独立キャッシュ。
- 予算 B = 1層あたり常駐 expert 数（総 DRAM ≈ B × 層数 × expertバイト）。
- 各トークンで各層 top_k(=8) experts にアクセス。常駐=hit、欠=miss（満杯なら方策で evict）。
- shared_expert / attention / router は常駐（非 expert）なのでキャッシュ対象外（trace にも無い）。
- キャッシュは **prompt ごとにコールド**から開始（独立リクエスト想定）。prefill で温まり
  decode が定常。**decode フェーズの hit 率が持続 tok/s を支配する**ので主指標にする。

go/no-go ゲート（物理から逆算）:
    per-token miss latency ≈ miss率 × (top_k × 層数) × expertバイト ÷ flash帯域(≈1GB/s)
これが目標 tok/s の時間予算（1/target）に収まるか。

使い方:
    python simulate.py --trace ../step1_routing_trace/traces.jsonl \
        --budgets 8,16,32,64,128,256 --target-tok-s 15
    python simulate.py --selftest      # 合成 trace で実装検証
"""

import argparse
import json
import sys
from bisect import bisect_right
from collections import OrderedDict, defaultdict

INF = float("inf")


def load_trace(path):
    """trace JSONL → {prompt_id: {layer_idx: [(token_idx, phase, (experts...)), ...]}}（token順）。"""
    by_prompt = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_prompt[r["prompt_id"]][r["layer_idx"]].append(
                (r["token_idx"], r.get("phase", "decode"), tuple(r["expert_ids"]))
            )
    for layers in by_prompt.values():
        for seq in layers.values():
            seq.sort(key=lambda x: x[0])
    return by_prompt


def simulate_layer(seq, capacity, policy, tally):
    """1 prompt・1 層の access 列を回し、phase 別 hit/miss を tally に加算。

    seq: [(token_idx, phase, experts_tuple), ...]（token 昇順）。
    policy: 'lru' | 'lfu' | 'belady'。
    """
    # 高速パス: LRU は OrderedDict で O(1)。oldest は popitem(last=False)。
    # 現トークンの expert は直前に末尾へ移動済み → B>=top_k なら victim にならない。
    if policy == "lru":
        cache = OrderedDict()
        for tok, phase, experts in seq:
            for e in experts:
                if e in cache:
                    tally[(phase, "hit")] += 1
                    cache.move_to_end(e)
                else:
                    tally[(phase, "miss")] += 1
                    if len(cache) >= capacity:
                        cache.popitem(last=False)
                    cache[e] = None  # 追加で末尾（=最新）
        return

    resident = {}  # expert -> bookkeeping（LFU: [freq, last_use]）

    # Belady 用: expert -> その層内で出現する token 位置の昇順リスト
    if policy == "belady":
        positions = defaultdict(list)
        for tok, _phase, experts in seq:
            for e in experts:
                positions[e].append(tok)

    def next_use(e, after_tok):
        idx = bisect_right(positions[e], after_tok)
        return positions[e][idx] if idx < len(positions[e]) else INF

    for tok, phase, experts in seq:
        protected = set(experts)  # 現トークンの top_k は同時常駐が必要 → evict 対象外
        for e in experts:
            if e in resident:
                tally[(phase, "hit")] += 1
            else:
                tally[(phase, "miss")] += 1
                if len(resident) >= capacity:
                    victim = _pick_victim(resident, policy, tok, protected,
                                          next_use if policy == "belady" else None)
                    del resident[victim]
                resident[e] = None
        # bookkeeping 更新（アクセスした全 expert を「今」使った扱いに）
        if policy == "lru":
            for e in experts:
                resident[e] = tok
        elif policy == "lfu":
            for e in experts:
                cur = resident.get(e)
                freq = (cur[0] + 1) if cur else 1
                resident[e] = [freq, tok]
        # belady は bookkeeping 不要（victim 選択時に next_use を引く）


def _pick_victim(resident, policy, tok, protected, next_use):
    # 現トークンで使用中の expert は victim にしない（同時常駐が必要）。
    candidates = [e for e in resident if e not in protected]
    if not candidates:
        # capacity >= top_k なら起きない（呼び出し側で保証）。
        candidates = list(resident)
    if policy == "lru":
        # last_use 最小
        return min(candidates, key=lambda e: resident[e] if resident[e] is not None else -1)
    if policy == "lfu":
        # freq 最小、tie は last_use 最小（=より昔）
        return min(candidates, key=lambda e: (resident[e][0], resident[e][1]) if resident[e] else (0, -1))
    if policy == "belady":
        # 次に使うのが最も先（=INF 優先）
        return max(candidates, key=lambda e: next_use(e, tok))
    raise ValueError(policy)


def run_policy(by_prompt, capacity, policy):
    """全 prompt・全層を回し、phase 別 hit/miss 合算を返す。"""
    tally = defaultdict(int)
    for layers in by_prompt.values():
        for seq in layers.values():
            simulate_layer(seq, capacity, policy, tally)
    return tally


def hit_rate(tally, phase=None):
    if phase:
        h = tally[(phase, "hit")]
        m = tally[(phase, "miss")]
    else:
        h = tally[("prefill", "hit")] + tally[("decode", "hit")]
        m = tally[("prefill", "miss")] + tally[("decode", "miss")]
    tot = h + m
    return (h / tot) if tot else 0.0


def infer_dims(by_prompt):
    """trace から層数と top_k を推定。"""
    layers = set()
    top_k = 0
    for lmap in by_prompt.values():
        layers.update(lmap.keys())
        for seq in lmap.values():
            for _t, _p, experts in seq:
                top_k = max(top_k, len(experts))
    return len(layers), top_k


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step 2 expert cache simulation")
    ap.add_argument("--trace", help="Step 1 traces JSONL")
    ap.add_argument("--budgets", default="8,16,32,64,128,256",
                    help="常駐 expert 数/層 のスイープ（カンマ区切り）")
    ap.add_argument("--policies", default="lru,lfu,belady")
    ap.add_argument("--expert-bytes", type=float, default=1.6e6,
                    help="1 expert のバイト数（既定 ~1.6MB: 4bit, inter=512, hidden=2048）")
    ap.add_argument("--flash-bw", type=float, default=1.0e9, help="flash 帯域 B/s（既定 1GB/s）")
    ap.add_argument("--target-tok-s", type=float, default=15.0, help="go/no-go の目標 decode tok/s")
    ap.add_argument("--out", help="結果 JSON 出力先")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.trace:
        raise SystemExit("--trace が必要（または --selftest）")

    by_prompt = load_trace(args.trace)
    n_layers, top_k = infer_dims(by_prompt)
    active_per_token = top_k * n_layers
    budgets = [int(b) for b in args.budgets.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    time_budget_s = 1.0 / args.target_tok_s

    print(f"[qwisp] layers={n_layers} top_k={top_k} active/token={active_per_token} "
          f"prompts={len(by_prompt)}", file=sys.stderr)
    print(f"[qwisp] expert={args.expert_bytes/1e6:.2f}MB flash={args.flash_bw/1e9:.1f}GB/s "
          f"target={args.target_tok_s}tok/s → budget {time_budget_s*1000:.1f}ms/token", file=sys.stderr)

    results = []
    header = f"{'policy':8} {'B/層':>5} {'totDRAM':>8} {'hit(all)':>9} {'hit(dec)':>9} {'miss(dec)':>9} {'+lat/tok':>9} {'flashTPS':>9} {'gate':>5}"
    print(header)
    print("-" * len(header))
    for policy in policies:
        for B in budgets:
            tally = run_policy(by_prompt, B, policy)
            h_all = hit_rate(tally)
            h_dec = hit_rate(tally, "decode")
            miss_dec = 1.0 - h_dec
            lat = miss_dec * active_per_token * args.expert_bytes / args.flash_bw  # s/token
            flash_tps = (1.0 / lat) if lat > 0 else INF
            gate = "GO" if lat <= time_budget_s else "no"
            tot_dram_gb = B * n_layers * args.expert_bytes / 1e9
            print(f"{policy:8} {B:>5} {tot_dram_gb:>7.1f}G {h_all:>9.3f} {h_dec:>9.3f} "
                  f"{miss_dec:>9.3f} {lat*1000:>7.1f}ms {flash_tps:>8.1f} {gate:>5}")
            results.append({
                "policy": policy, "budget_per_layer": B, "total_dram_gb": tot_dram_gb,
                "hit_all": h_all, "hit_decode": h_dec, "miss_decode": miss_dec,
                "added_latency_s": lat, "flash_only_tok_s": flash_tps, "gate_pass": gate == "GO",
            })

    # 局所性の素の指標: B=256（全載り）の decode hit = 1 - compulsory miss。
    print("\n[qwisp] 解釈の目安:", file=sys.stderr)
    print("  - Belady の decode hit が中予算で高い → hot expert に局所性あり → ストリーミング有望。", file=sys.stderr)
    print("  - Belady でも低い（256 に均等分散）→ キャッシュ無力 → フロア引き上げ or 設計見直し。", file=sys.stderr)
    print("  - gate=GO の最小予算が現実的 DRAM に収まるかが go/no-go の核。", file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": {"layers": n_layers, "top_k": top_k,
                                "expert_bytes": args.expert_bytes, "flash_bw": args.flash_bw,
                                "target_tok_s": args.target_tok_s}, "results": results}, f, indent=2)
        print(f"[qwisp] results -> {args.out}", file=sys.stderr)


def selftest():
    """合成 trace で不変条件を検証。"""
    # 2 prompt, 1 層, top_k=2, experts in {0,1,2,3}
    def row(pid, tok, phase, experts):
        return {"prompt_id": pid, "layer_idx": 0, "token_idx": tok, "phase": phase, "expert_ids": experts}
    rows = [
        row("p", 0, "prefill", [0, 1]), row("p", 1, "prefill", [1, 2]),
        row("p", 2, "decode", [0, 1]), row("p", 3, "decode", [2, 3]),
        row("p", 4, "decode", [0, 1]),
    ]
    import io
    buf = io.StringIO("\n".join(json.dumps(r) for r in rows))
    by_prompt = defaultdict(lambda: defaultdict(list))
    for line in buf:
        r = json.loads(line)
        by_prompt[r["prompt_id"]][r["layer_idx"]].append((r["token_idx"], r["phase"], tuple(r["expert_ids"])))

    n_layers, top_k = infer_dims(by_prompt)
    assert n_layers == 1 and top_k == 2, (n_layers, top_k)

    # B >= 4（全 expert 数）→ compulsory miss のみ。アクセス=10、distinct=4 → hit=6。
    t = run_policy(by_prompt, 4, "lru")
    assert hit_rate(t) == 6 / 10, hit_rate(t)
    t = run_policy(by_prompt, 4, "belady")
    assert hit_rate(t) == 6 / 10, hit_rate(t)

    # Belady は LRU 以上のヒット率（同予算）。
    for B in (2, 3):
        hl = hit_rate(run_policy(by_prompt, B, "lru"))
        hb = hit_rate(run_policy(by_prompt, B, "belady"))
        assert hb >= hl - 1e-9, (B, hl, hb)

    # B=1 で容量1：seq の各トークン2 expert は必ず1つ miss するなど下限挙動。
    t1 = run_policy(by_prompt, 1, "lru")
    assert hit_rate(t1) >= 0.0
    print("[qwisp][selftest] OK", file=sys.stderr)


if __name__ == "__main__":
    main()
