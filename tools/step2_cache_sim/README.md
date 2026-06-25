# Step 2 — expert キャッシュシミュレーション

Step 1 の routing trace に対し、**DRAM 予算（=常駐 expert 数/層）を振りながら** LRU / LFU / Belady(oracle) のヒット率と go/no-go ゲートを出す。ロードマップ（`../../docs/02-roadmap.md` Step 2）の中核 ―― **「ここで実質ぜんぶ決まる。」**

## モデル

- cache 単位 = `(layer, expert)`。layer L の expert は L 専用なので**層ごとに独立キャッシュ**。
- 予算 `B` = 1層あたり常駐 expert 数（総 expert DRAM ≈ `B × 層数 × expertバイト`）。
- 各トークンで各層 `top_k(=8)` expert にアクセス。常駐=hit、欠=miss（満杯なら方策で evict）。
  **現トークンの top_k は同時常駐が必要**なので evict 対象から除外（`B ≥ top_k` 前提）。
- shared_expert / attention / router は常駐（非 expert）= キャッシュ対象外（trace にも無い）。
- キャッシュは **prompt ごとにコールド**開始。prefill で温まり decode が定常 →
  **decode hit 率が持続 tok/s を支配する**ので主指標。

## go/no-go ゲート（物理から逆算）

```
per-token miss latency ≈ miss率(decode) × (top_k × 層数) × expertバイト ÷ flash帯域(≈1GB/s)
```

これが目標 tok/s の時間予算 `1/target` に収まれば `gate=GO`。

## 使い方

```bash
python simulate.py --selftest        # 合成 trace で実装検証（Belady 最適性など）

python simulate.py \
    --trace ../step1_routing_trace/traces.jsonl \
    --budgets 8,16,32,64,128,256 \
    --expert-bytes 1.6e6 --flash-bw 1e9 --target-tok-s 15 \
    --out results.json
```

パラメータ:
- `--expert-bytes` 既定 ~1.6MB（4bit / moe_intermediate=512 / hidden=2048 の 1 expert）。
- `--flash-bw` 既定 1GB/s（NAND シーケンシャル目安）。
- `--target-tok-s` go/no-go の目標 decode tok/s。

## 出力の読み方

| 列 | 意味 |
| --- | --- |
| `hit(dec)` | **decode フェーズ hit 率（主指標）** |
| `+lat/tok` | flash miss による per-token 追加レイテンシ |
| `flashTPS` | flash 律速だけで決まる tok/s 上限 |
| `gate` | `+lat/tok ≤ 1/target` なら GO |

判定（ロードマップ）:
- **Belady の decode hit が中予算で高い**（hot expert に局所性）→ ストリーミング有望・**GO**。
- **Belady でも低い**（256 に均等分散）→ キャッシュ無力 → フロア引き上げ or 設計見直し・**NO-GO**。
- **`gate=GO` の最小予算が現実的 DRAM に収まるか**が go/no-go の核。Belady=実現可能な上界、
  LRU/LFU=素朴方策の下界。両者の差が「賢いキャッシュ方策（ML/予測）にどれだけ投資価値があるか」。

> 注意: 極小 trace（生成トークン数が少ない）では decode 統計が薄く参考値。Step 1 の本収集
> （`--max-new-tokens 128`＋拡充プロンプト）後の数値で判断する。

## 将来オプション（雛形では未実装）

- 層ごとでなく**グローバル DRAM 予算**で配分する変種。
- **prefetch**（数層先予測）込みのヒット率。HOBBIT の留保（load≫compute 域で利得限定）を併記。
- **混合精度 expert**（hot 高bit常駐 / cold 低bit）でのバイト数可変モデル。
