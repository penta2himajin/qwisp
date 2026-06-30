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

モデルは Stage A と一貫（MTPLX graft の mult/acceptance）。miss コストは「数×flat」→
「Σ 実バイト(hot1.77/cold0.98)」に置換。overlap は3ブラケット:
  serial(φ=0) / ideal上限(φ=1) / **cutoff(φ=実現可能, SP-MoE arXiv:2510.10302 cutoff-layer を
  Qwisp の MTP窓構造へ再導出)**。cutoff は T_cycle=T_comp+(1−φ)T_flash,
  φ=min(φ_max, (T_draft+verify_overlap)/T_flash)、φ_max=予測 coverage(prev0.66/xlayer0.77)。
  sync(forward内 per-layer)は base_tps=54/T(W1) に内包済ゆえ hideable から差引かない(二重計上回避)。

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
            compute_penalty=1.0, t_draft_frac=2.5 / 40, overlap_eff=1.0,
            cov_prev=0.66, cov_xlayer=0.77):
    """net_tps(D, RAM) を serial(φ=0) / ideal(φ=1, 上限) / cutoff(φ=実現可能) で返す。

    ★cutoff（SP-MoE arXiv:2510.10302 の cutoff-layer を Qwisp 構造へ再導出）:
      T_cycle(φ) = T_comp + (1−φ)·T_flash, φ = min(φ_max, hideable_budget / T_flash)。
      - **sync は hideable から引かない**(二重計上回避): forward内 per-layer sync ~20ms/token は
        base_tps=54 と T(W1)実測に内包済み＝既に t_compute の中。引くと二重。
      - hideable_budget = T_draft + verify_overlap（IO を隠せる wall-clock）:
        ・T_draft = t_draft_frac·T_comp。MTP ヘッド(~1層+head)＝SP-MoE の別 draft-model(L_all層フル)と違い
          窓 ≪ L_all·t_comp。上限 ~(2-3)/40·T_comp。D0 は draft 無し→0。[assumed/sweep]
        ・verify_overlap = overlap_eff·T_comp（40層計算中に次層 expert を先読み）[assumed/sweep]
      - **φ_max = 1 − miss_resync_fraction**: prefetch-exactness の miss検出 sync は miss時のみ＝
        予測 coverage の外側は on-demand 再同期で隠せない。φ_max = 予測 coverage。
        prev-token 再利用 0.66 / cross-layer 予測 0.77(docs/07) の2ケースで算出（分岐点を見る）。
      - t_flash は **miss バイトのみ**(sim_bytes が union-miss を集計＝cache hit 85% は既に除外)
        ゆえ φ は「miss IO のうち隠せた割合」、cache 重複の二重差引は不要。
    """
    acc = accepted_per_verify(D)
    # mixed は two-gather の compute overhead を持つ（resident 実測 0.71x ≈ ×1.41）。
    # 融合カーネルなら penalty=1.0。flash-bound では効かないが overlap(compute-bound) で効く。
    t_compute = acc / (base_tps * MULT[D]) * (compute_penalty if mixed else 1.0)
    miss_bytes_pv, n_layers = sim_bytes(by_prompt, budget_bytes, D, hot_by_layer, mixed)
    t_flash = miss_bytes_pv / flash_bw
    # ★ hideable_budget: draft窓 + verify内 cross-layer overlap（sync は差引かない）。
    #   ★単調性の要請: hidden IO は compute 時間を超えて隠せない（h ≤ t_compute）。T_draft は MULT に
    #   既に含まれる draft コストの一部＝t_compute の外に加算すると h>t_comp となり cutoff>ideal の非物理。
    #   よって hideable = T_draft + overlap_eff·(t_compute − T_draft) で t_compute を上限に保つ
    #   (draft 窓は完全 overlap 可、残り verify compute は overlap_eff で overlap)。
    t_draft = t_draft_frac * t_compute if D >= 1 else 0.0       # D0 は MTP draft 窓を持たない
    hideable = min(t_compute, t_draft + overlap_eff * (t_compute - t_draft))

    budget_phi = float("inf") if t_flash <= 0 else hideable / t_flash   # 時間予算で隠せる上限(coverage 無視)

    def cutoff(phi_max):                                        # φ は coverage(φ_max)と時間予算の両制約
        phi = phi_max if t_flash <= 0 else min(phi_max, hideable / t_flash)
        return acc / (t_compute + (1.0 - phi) * t_flash), phi
    net_prev, phi_prev = cutoff(cov_prev)
    net_xl, phi_xl = cutoff(cov_xlayer)
    return {
        "D": D, "acc": acc, "t_compute": t_compute, "t_flash": t_flash, "budget_phi": budget_phi,
        "net_serial": acc / (t_compute + t_flash),
        "net_overlap": acc / max(t_compute, t_flash),                 # ideal 上限(φ=1)
        "net_cutoff_prev": net_prev, "phi_prev": phi_prev,            # prev-token prefetch(φmax=0.66)
        "net_cutoff_xl": net_xl, "phi_xl": phi_xl,                    # cross-layer 予測(φmax=0.77)
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
    # ★ cutoff ブラケット用（SP-MoE cutoff を Qwisp 構造へ再導出。sync は t_compute 内包ゆえ引かない）
    ap.add_argument("--t-draft-frac", type=float, default=2.5 / 40,
                    help="MTP draft 窓 / T_comp。上限 ~(2-3)/40[assumed/sweep]。D≥1 のみ")
    ap.add_argument("--overlap-eff", type=float, default=1.0,
                    help="verify 40層計算中の cross-layer overlap 時間率[assumed/sweep, 0..1]")
    ap.add_argument("--cov-prev", type=float, default=0.66,
                    help="prev-token prefetch coverage=φmax[measured docs/07, ~66-70%%]")
    ap.add_argument("--cov-xlayer", type=float, default=0.77,
                    help="cross-layer 予測 coverage=φmax[measured docs/07, 77%%]")
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
    print(f"[mtp-mix] cutoff: t_draft_frac={args.t_draft_frac:.3f}[assumed] "
          f"overlap_eff={args.overlap_eff}[assumed] φmax: prev={args.cov_prev}/xlayer={args.cov_xlayer}[measured] "
          f"(sync は t_compute 内包ゆえ非差引)", file=sys.stderr)

    hdr = (f"{'cfg':6} {'総GB':>5} {'D':>2} {'Tcomp':>7} {'Tflash':>7} {'serial':>6} {'ideal':>6} "
           f"{'cutP':>6} {'φp':>4} {'cutXL':>6} {'φx':>4} {'best(S/I/P/X)':>13}")
    print(hdr); print("-" * len(hdr))
    grid = {}   # (gb,cfg) -> rows（Step5 比較用）
    for gb in budgets:
        budget_bytes = gb * 1e9 / n_layers
        for cfg, mixed in (("all4", False), ("mixed", True)):
            rows = [net_for(by_prompt, budget_bytes, d, hot_by_layer, mixed,
                            args.baseline_tok_s, args.flash_bw, args.mixed_compute_penalty,
                            args.t_draft_frac, args.overlap_eff, args.cov_prev, args.cov_xlayer)
                    for d in depths]
            grid[(gb, cfg)] = rows
            bs = max(rows, key=lambda r: r["net_serial"])["D"]
            bi = max(rows, key=lambda r: r["net_overlap"])["D"]
            bp = max(rows, key=lambda r: r["net_cutoff_prev"])["D"]
            bx = max(rows, key=lambda r: r["net_cutoff_xl"])["D"]
            for r in rows:
                tag = "".join([("S" if r["D"] == bs else "-"), ("I" if r["D"] == bi else "-"),
                               ("P" if r["D"] == bp else "-"), ("X" if r["D"] == bx else "-")])
                print(f"{cfg:6} {gb:>5.1f} {r['D']:>2} {r['t_compute']*1000:>6.1f}ms "
                      f"{r['t_flash']*1000:>6.1f}ms {r['net_serial']:>6.1f} {r['net_overlap']:>6.1f} "
                      f"{r['net_cutoff_prev']:>6.1f} {r['phi_prev']:>4.2f} "
                      f"{r['net_cutoff_xl']:>6.1f} {r['phi_xl']:>4.2f} {tag:>13}")
            print()

    # ★ 単調性 self-check: 同一 row で φ=0(serial) ≤ φp(.66) ≤ φx(.77) ≤ φ=1(ideal) ゆえ
    #   net は serial ≤ cutP ≤ cutXL ≤ ideal でなければ T_cycle 実装バグ。
    viol = []
    for (gb, cfg), rows in grid.items():
        for r in rows:
            seq = [r["net_serial"], r["net_cutoff_prev"], r["net_cutoff_xl"], r["net_overlap"]]
            if any(seq[i] > seq[i + 1] + 1e-6 for i in range(3)):
                viol.append(f"{cfg}{gb:.0f}GB D{r['D']}: {[round(x,1) for x in seq]}")
    print(f"\n[mtp-mix] 単調性 self-check(per-row serial≤cutP≤cutXL≤ideal): "
          f"{'PASS（T_cycle 健全）' if not viol else 'FAIL: ' + '; '.join(viol)}", file=sys.stderr)

    # ★(a) overlap の純価値: mixed 構成・固定 depth で serial/cutoff/ideal と Δ=cutoff−serial。
    #   depth argmax も分母 ratio も介さず、overlap だけの寄与を tier 別に見る。
    fixed_d = 1   # MTP D1（本命）。depth 最適化はこの外側の別問題。
    print(f"\n[mtp-mix] (a) overlap 純価値（mixed, D{fixed_d} 固定, 同一 miss バイト）:", file=sys.stderr)
    print(f"  {'GB':>4} {'Tflash':>7} {'serial':>6} {'cutP':>6}(Δ) {'cutXL':>6}(Δ) {'ideal':>6}(Δ)", file=sys.stderr)
    for gb in budgets:
        r = next(x for x in grid[(gb, "mixed")] if x["D"] == fixed_d)
        s = r["net_serial"]
        print(f"  {gb:>4.0f} {r['t_flash']*1000:>6.1f}ms {s:>6.1f} "
              f"{r['net_cutoff_prev']:>6.1f}(+{r['net_cutoff_prev']-s:.1f}) "
              f"{r['net_cutoff_xl']:>6.1f}(+{r['net_cutoff_xl']-s:.1f}) "
              f"{r['net_overlap']:>6.1f}(+{r['net_overlap']-s:.1f})", file=sys.stderr)

    # ★(b) φ_max 到達度: φ が coverage cap(.66/.77)に達したか、draft窓/時間予算で頭打ちか。
    print(f"\n[mtp-mix] (b) φ 到達度（mixed/all4, D{fixed_d}）: budget_phi=hideable/Tflash(時間上限)、"
          f"φ=min(φmax,budget_phi)。budget<φmax なら draft窓律速:", file=sys.stderr)
    for cfg in ("mixed", "all4"):
        for gb in budgets:
            r = next(x for x in grid[(gb, cfg)] if x["D"] == fixed_d)
            bp = r["budget_phi"]   # =hideable/t_flash。hideable=t_draft+overlap_eff·(t_comp−t_draft)
            capP = "cov" if bp >= 0.66 else "予算"   # 予算律速=時間(overlap_eff低なら draft窓主体)で頭打ち
            capX = "cov" if bp >= 0.77 else "予算"
            print(f"  {cfg:5} {gb:>4.0f}GB: budget_phi={bp:>5.2f}  φp={r['phi_prev']:.2f}({capP}) "
                  f"φx={r['phi_xl']:.2f}({capX})", file=sys.stderr)

    # ★ Step5（depth 固定 D1 で比較。分母 all4-D0 も同 bracket overlap を受ける点を明記）
    print(f"\n[mtp-mix] === Step5: mixed+MTP(D{fixed_d}) vs all4-noMTP(D0)（同 bracket）===", file=sys.stderr)
    print("  ※比率は分母 all4-D0 も同 bracket の overlap を受ける（D0 は IO 律速強→overlap 利得大）"
          "ため、cutoff/ideal で比率が serial より縮むのは正常（T_cycle は単調）。", file=sys.stderr)
    for label, key in (("serial", "net_serial"), ("ideal上限", "net_overlap"),
                       ("cutoff-prev(.66)", "net_cutoff_prev"), ("cutoff-xlayer(.77)", "net_cutoff_xl")):
        wins = []
        for gb in budgets:
            a0 = next(r for r in grid[(gb, "all4")] if r["D"] == 0)[key]
            md = next(r for r in grid[(gb, "mixed")] if r["D"] == fixed_d)[key]
            ratio = md / a0 if a0 else 0
            wins.append(f"{gb:.0f}GB:{'✓' if md > a0 else '✗'}{ratio:.2f}x")
        print(f"  {label:18}: " + "  ".join(wins), file=sys.stderr)


if __name__ == "__main__":
    main()
