# 01 — AFM 3 Core Advanced と関連研究

調査時点：2026-06。数値・年月は一次/二次ソースに基づくが、ブログ等の二次情報も含むため実装前に原典で再確認すること。

## A. AFM 3 Core Advanced（戦略の実体）

- WWDC26（2026-06-08）発表の第 3 世代 Apple Foundation Models。Core Advanced は最上位オンデバイスモデル。
- **20B パラメータのスパースモデルで、リクエストごとに 1–4B だけアクティブ化**。ネイティブマルチモーダル。
- 仕組み：**フル 20B は NAND フラッシュに常駐**させ、プロンプトに応じて選んだ expert 重みだけを DRAM にロード。
- 選択機構 = **Instruction-Following Pruning (IFPruning)**。小さな sparsity 予測器がプロンプトを読み、FFN 行列のどの行・列を活性化するかをリクエストごとに動的決定。
- **重要な制約**：「最も高性能な Apple silicon でのみ解禁・最適化」＝弱いチップには載らない。

原典：
- Apple ML Research, "Introducing the Third Generation of Apple's Foundation Models" — https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models

## ★ 最重要の分析：粒度のズレ

本プロジェクトに AFM 戦略を流用するときの落とし穴。

- **IFP はプロンプト単位**でマスクを固定する。1 リクエスト中アクティブ集合が不変 → スライスを一度ロードして使い回せる → フラッシュアクセスの churn が小さい。**これがフラッシュ常駐を成立させている核心。**
- **Qwen3.6-35B-A3B の MoE ルーティングはトークン単位**。アクティブ expert が毎トークン変わる → フラッシュ局所性が悪い → 「毎トークン stall / NAND 摩耗」の最悪ケースを踏む。

→ **「AFM 戦略を MoE に素朴コピー」すると、Apple が回避している難問の方を引き受ける。** この差を埋めるのが下の C 節（MoE offloading）。

## B. フラッシュ常駐推論の土台

**LLM in a flash** (Apple, arXiv:2312.11514) — https://arxiv.org/abs/2312.11514

- フラッシュに重みを置きオンデマンドで DRAM へ。DRAM の 2 倍サイズのモデルを、素朴ロード比 CPU 4–5×・GPU 20–25× で実行。
- 2 技法：(1) **windowing**（直近アクティブニューロン再利用で転送量削減）、(2) **row-column bundling**（フラッシュのシーケンシャル特性に合わせ大きな連続チャンクで読む）。
- 物理：**フラッシュ ≈ 1GB/s、DRAM ≈ 100GB/s**（約 1/100）。転送量最小化と連続性が全て。
- → ストリーミング層の設計図。コストモデルと windowing/bundling をそのまま下敷きにできる。

## C. MoE expert offloading（Qwen3.6-35B-A3B に最も直接効く線）

| 研究 | 要点 | URL |
| --- | --- | --- |
| **FlashMoE** (2026-01) | expert/非 expert を分離保存し layer/unit 単位で細粒度オンデマンドロード。Belady 最適を ML 近似したキャッシュ置換で LRU/LFU 比ヒット率 +51%・最大 2.6×。**最新・エッジ SSD 前提で最も刺さる。** | arXiv:2601.17063 |
| **ProMoE** | 学習済み予測器による proactive prefetch とインフレ調整 | arXiv:2410.22134 |
| **OD-MoE** (2025-12) | 完全オンデマンド（キャッシュレス）。数レイヤ先の expert 活性を予測 | arXiv:2512.03927 |
| **HOBBIT** | 混合精度 expert offloading。頻用の高精度 expert をキャッシュ優先保持 | arXiv:2411.01433 |
| **EdgeMoE** | オンデバイス MoE。expert 個別ビット幅 + 計算 I/O パイプライン preload | (Yi et al. 2023) |
| **Mixtral-Offloading** | LRU expert キャッシュ + 混合量子化 + 投機 prefetch | (Eliseev & Mazur 2023) |
| **MoE-Infinity** | シーケンス単位の活性認識 prefetch + LFU | (Xue et al. 2024) |
| **PowerInfer-2** | スマホでの高速 LLM 推論（hot/cold split + フラッシュ offload） | (Xue et al. 2024) |

盗める共通テク：

1. **expert / 非 expert を分離**：attention・router・shared expert は常駐、routed expert だけストリーム。
2. **キャッシュ方策が I/O 生値より効く**：LRU/LFU → ML/Belady 近似。expert にはトークン間の時間的局所性がある。
3. **多層先予測で prefetch**：ロードと計算をオーバーラップ。
4. **混合精度 expert**：hot は高 bit 常駐、cold は低 bit フラッシュ。

ただし **HOBBIT の留保**：ロードコストが計算コストを大きく上回る場面では prefetch の利得は限定的。フラッシュ 1GB/s 域はまさにそれ。→ **prefetch 単独では救われず、ヒット率 + 量子化 + skip の合わせ技が要る。**

## D. IFP 路線（モデル再学習を厭わない場合のみ）

- **IFPruning** (Apple/UCSB, arXiv:2501.02086, ICML2025) — https://arxiv.org/abs/2501.02086
  - sparsity 予測器 + LLM の二段階共同学習。9B→3B 刈り込みで dense 3B を 5–8pt 上回り dense 9B に匹敵、TTFT は 3B 並み。
- 27B dense に IFP 的な prompt-conditioned pruning を乗せれば、リクエスト単位でアクティブ集合を固定でき B/C の局所性問題が緩む。
- ただし**学習データ・計算コストが要る本格 ML ワーク**。off-the-shelf で済ませたいなら C 優先。別フェーズ扱い。
- 関連：Probe Pruning、Prompt-prompted Adaptive Structured Pruning (Dong & Chen)、Federici et al. "Dynamic Input Pruning + Cache-Aware Masking"（限定メモリ向けで cache-aware、要チェック）。

## E. Qwisp 固有の批判的留保

1. **フラッシュ 1GB/s の壁**：windowing/bundling は硬い壁を柔らかい壁に変えるだけ。勝負は「hot expert の作業集合を DRAM に高ヒット率で収める」一点。収まらなければ結局数 tok/s。
2. **MLX / unified memory とのトポロジー不整合**：上記 offloading 研究はほぼ CUDA/PCIe 前提（GPU↔ホスト DRAM）。Apple は独立 VRAM が無く、"offload" 先はホスト DRAM でなく **NAND**。→ コード移植はできない。**方策（キャッシュ置換・prefetch・expert 分離・混合精度）を MLX + mmap-from-NAND 環境に翻訳する**ことになる。逆に言えばエッジ MoE 研究の大半が離散 GPU 前提なので、ここは**未踏で新規性を出せるニッチ**。ただし流用コードは減り自前システム実装比重が増える。

## 参考：ハード前提（base ティア）

| SoC | 帯域 | base RAM |
| --- | --- | --- |
| M1 | 68 GB/s | 8 / 16GB |
| M2 | ~100 GB/s | 8 / 16 / 24GB |
| M3 | ~100 GB/s | 8 / 16 / 24GB |
| M4 | 120 GB/s | 16GB〜 |
| M5 | 153 GB/s | 16GB〜 |

decode の tok/s 上限は帯域で決まる。base ティアは世代内で常に最低帯域。
