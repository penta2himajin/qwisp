# 02 — ロードマップ（着手手順）

原則：**安くて「プロジェクト全体を殺せる／活かせる」順**。エンジン実装は最後。Step 1–2 は週末で回せる規模。

---

## Step 1 — ルーティング trace 収集 ★最初の一手

**やること**：Qwen3.6-35B-A3B を `mlx_lm` または `transformers` で動かし、router/gate をフックして「**レイヤ別・トークン別の top-k expert ID 列**」を、実用途の代表プロンプト群でログするだけ。

**重要な気づき**：routing は「モデル × プロンプトの性質」であって**ハードに依存しない**。よって制約デバイスは一切不要。24GB+ のマシンでもクラウドでも普通に流して取れる。**フロア機が手元になくても今日から始められる。**

代表プロンプト群（確定比率 — 主用途を重く）：
- **コーディング 45%**（既存リポジトリ編集・生成）
- **エージェント tool-call 45%**（tool-calling ループ）
- **長文脈 10%**（要約・NIAH 的検索）

比率の根拠：コーディング／agentic が主用途で、長文脈はその上に乗る応用（前者2つが立って初めて要る）。同ドメイン（長いコード・長い tool 履歴）なので活性化 expert の*集合*は重複が大きい。ただし長文脈枠をゼロにしないのは、Step 2 が測るのは expert の種類でなく **DRAM 予算に対するワーキングセット／churn** であり、長い prefill は同じ expert 集合でも瞬間作業集合を膨らませて **no-go を出しうる唯一のケース**だから。10% は「用途」でなく「キャッシュを膨らませるストレス検査」枠。なお agentic ループは履歴が自然に伸びるため、長文脈カバレッジの一部は agentic 枠内で無料で入る。

出力フォーマット（案）：`{prompt_id, token_idx, layer_idx, [expert_ids]}` の JSONL。

**フレームワーク選択**（ここだけ決めれば動ける）：
- Apple 寄り・本番に近い → `mlx_lm`
- フックの手軽さ → `transformers`（`output_router_logits=True` 系 / forward hook）

## Step 2 — キャッシュシミュレーション（オフライン、コード数十行）

Step 1 の trace に対し、**DRAM 予算（＝常駐できる expert 数）を振りながら** LRU / LFU / Belady(oracle) のヒット率を出す。**ここで実質ぜんぶ決まる。**

判定：
- hot な expert 小集合に集中 → 中程度キャッシュで高ヒット → **ストリーミング成立・GO**
- 256 にほぼ均等分散 → キャッシュ無力 → 毎トークンフラッシュ地獄 → **設計やり直し or フロア引き上げ**

**go/no-go の定量ゲート**（物理から逆算）：

```
per-token 追加レイテンシ ≈ miss率 × アクティブexpert数 × expertバイト数 ÷ flash帯域(≈1GB/s)
```

これが目標 tok/s の時間予算に収まるか。FlashMoE 等が成立している以上それなりの局所性はあるはずだが、**モデル固有なので測るしかない**。

参考目標例（要 4 つ組固定）：`35B-A3B 混在Q3/Q4 / decode ≥15 tok/s / 16K ctx / M? 24GB`。

## Step 3 — 素の MLX ベースライン + フロア確定（Step 1 と並行可）

フロア候補機で素の `mlx_lm` の 35B-A3B（と 27B）を 2–3 量子化で回し、記録：

- 載るか / ピークメモリ
- decode tok/s / prefill tok/s
- 上記を 2–3 種の context 長で

→ 「**超えるべき基準値**」と「現実的フロア」が固まる。AFM の top-chip gating を踏まえると、たぶん 24GB 級に落ち着く。

---

## Step 4 以降（Step 2 が GO を出してから）

1. expert / 非 expert 分離ロード（非 expert = attention・router・shared expert を常駐）
2. MLX + mmap-from-NAND でのキャッシュ方策実装（Step 2 で勝った方策を移植・翻訳）
3. 混合精度 expert（hot 高 bit / cold 低 bit）
4. Multi-Token Prediction / 投機デコード統合
5. tool-calling 形状・logit 検証込みのエージェント層

> 注意：C 節の offloading 研究は CUDA/PCIe 前提。**コード移植でなく方策の翻訳**になる（`01-research-notes.md` E-2）。

## 読む順（並行インプット）

1. LLM in a flash（コストモデル・windowing/bundling）
2. FlashMoE（最新・エッジ SSD・expert 分離・ML キャッシュ）
3. ProMoE / OD-MoE（予測器）
4. HOBBIT / EdgeMoE（混合精度 expert）
5. IFPruning（学習を許容する場合のみ）

## 当面の意思決定ポイント

- [x] Step 1 のフック → **MLX で計装・実装済み＆実モデルで検証済み**（transformers 案は撤回）。理由：transformers を選んだ唯一の動機は `output_router_logits` の hook 容易性だけで、routing 自体はフレームワーク非依存。だが手元の実用モデルが MLX 形式（`~/.mtplx/.../Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`, 4bit, 256 experts/top-8, 40層）で **24GB に載って今すぐ動く唯一の道が MLX**。かつ量子化 routing = 出荷忠実で go/no-go に最も効く。実装は `tools/step1_routing_trace/collect_traces.py`：`qwen3_next.Qwen3NextSparseMoeBlock.__call__` を monkeypatch し `argpartition(softmax(gate(x)))[..., -8:]` を捕捉、**AR モード（mlx_lm.load は mtp.* を落とす）**で JSONL 化。smoke 検証で layers=40/top_k=8/max_expert_id=255 が config 一致を確認。
- [x] 代表プロンプト群の確定 → **コーディング45 / エージェント tool-call 45 / 長文脈10**（下記 Step 1 参照）
- [ ] フロア機種の暫定値（手元 or 入手予定の最弱機）
- [ ] 「十分高性能」の 4 つ組定義（Step 3 後に確定）
