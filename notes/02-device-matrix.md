# デバイス matrix と device 別最適化（measurement-primary engine の reference 層）

作成: 2026-06-28 / 出典: web 調査エージェント 2 本（Apple Newsroom 一次・Wikipedia・tech press・実測レビュー）。
M5 系・MacBook Neo は 2026-03 発表で日付が新しいため「fresh だが要再確認」。M1–M4＋M5 base は確定。

## 0. 方針: 実測主軸ハイブリッド（議論で決定）

device 別チューニングは **(a) RAM 容量で静的に mode を gate** ＋ **(b) cost model を起動時にオンデバイス実測**。
理由: スペックからの挙動予測は本 repo で繰返し外れた（3-bit=bytes 減で速いはず→実は dispatch-bound、
batch=1=bandwidth-bound→小 expert で 8bit≈4bit。[[greedy-wall-fixed-overhead]] [[nl-speedup-research]]）。
同一 device でもサーマルで 88→55 tok/s 変動。**engine が要るのは生スペックでなく派生決定変数（cost(L) 係数・
実効 SSD 帯域・speculation 損益分岐）で、これらは実測が確実**。本 matrix は reference / fallback / sanity 基準。

---

## 1. compute & memory bandwidth（lowest → highest）

memory-bound decode では **memory BW が最重要**。Max 系は M3 以降 GPU bin で BW が 2 値に分かれるので
**「Max だから」で BW を仮定不可＝GPU コア数を見る**こと。

| Chip | GPU cores(bin) | Mem BW (GB/s) | Max RAM | 主な Mac | 備考 |
|---|---|---|---|---|---|
| **A18 Pro** | 6 | ~60(推定) | 8(固定) | **MacBook Neo**($599, 2026-03) | LPDDR5X-7500。BW は press 推定。8GB 非増設＝最厳しい床 |
| M1 | 7/8 | 68.25 | 16 | Air, mini, 13"MBP | LPDDR4X 128-bit |
| M2 | 8/10 | 100 | 24 | Air, mini | LPDDR5 |
| M3 | 8/10 | 100 | 24 | Air, mini | base RAM=8 |
| M4 | 8/10 | 120 | 32 | Air, mini, base MBP | LPDDR5X。base RAM 16 へ |
| **M5** | 8/10 | 153 | 32 | 14"MBP(2025-10), Air(2026) | LPDDR5X 9600。各 GPU core に Neural Accel |
| M1 Pro | 14/16 | 200 | 32 | 14/16"MBP | 256-bit |
| M2 Pro | 16/19 | 200 | 32 | MBP, mini | |
| M3 Pro | 14/18 | **150** | 36 | 14/16"MBP | ⚠️**回帰**: 200→150(bus 256→192-bit) |
| M4 Pro | 16/20 | 273 | 64 | MBP, mini | |
| M5 Pro | 16/20 | 307 | 64 | 14/16"MBP(2026-03) | |
| M1 Max | 24/32 | 400 | 64 | MBP, Studio | 512-bit |
| M2 Max | 30/38 | 400 | 96 | MBP, Studio | |
| M3 Max | 30/40 | **300(30c)/400(40c)** | 128 | 14/16"MBP | ⚠️bin で 2 値 |
| M4 Max | 32/40 | **410(32c)/546(40c)** | 128 | MBP, Studio | ⚠️bin で 2 値 |
| **M5 Max** | 32/40 | **460(32c)/614(40c)** | 128 | 14/16"MBP(2026-03) | 高性能側ターゲット |
| M1 Ultra | 48/64 | 800 | 128 | Studio | UltraFusion |
| M2 Ultra | 60/76 | 800 | 192 | Studio, Mac Pro | |
| M3 Ultra | 60/80 | 800 | **512** | Studio(2025-03) | 最大 unified memory。M4/M5 Ultra は不在 |

**base 帯域推移**: 68→100→100→120→**153**(M5)。**Pro 回帰**: 200→200→**150(M3 Pro↓)**→273→307。
FP16/FP32 TFLOPS は M3+ 非公開（相対値のみ）＝公称 compute は当てにせず実測。

## 2. SSD read 帯域（streaming tier=8GB 機で律速。256GB 単一 NAND が危険域）

**容量でなく NAND ダイ数が速度を決める**。256GB が単一ダイだと ~半速。1.7MB chunk read は large-block
regime ゆえ random≈sequential（単一 NAND は天井を ~50% 下げるが random 固有の崖は無い）。

| Model | Cap | Read MB/s | NAND | 備考 |
|---|---|---|---|---|
| **A18 Pro MacBook Neo** | 256 | **~1,510–1,735** | 1 | ★真の床(実測, 8GB 固定)。streaming 予算は ~1.5GB/s で組む |
| M1 Air | 256 | ~2,700 | 2 | 2 ダイ＝良い側(床でない) |
| **M2 Air/Pro/mini** | 256 | **~1,450–1,580** | 1 | ⚠️単一 NAND 回帰世代 |
| M3 Air | 256 | ~2,280–2,880 | 2 | read 復帰 |
| M4 Air | 256 | ~2,880 | 2 | read 健全だが write ~1,950(部分単一) |
| 標準(非Pro/Max)512–1TB | | ~3,000 plateau | 2+ | 容量上げても ~3GB/s 頭打ち |
| M4 Pro mini 512 | | ~6,300 | 4+ | Pro/Max controller で跳ねる |
| M5 MBP | | 6,000+ | many | ≈2.5× M4 |
| Ultra Studio | | ~5,900–7,100 | many | burst≠sustained |

## 3. RAM tier → engine mode（静的 gate）

★**RSS 実測(2026-06-28, buddy-no-sync hot-pin top-C, dev 機 64GB=圧迫無の純 footprint)**:
C=64→**6.9GB** / C=128→**11.4GB** / C=192→**15.9GB** / C=256(full-resident)→**20.5GB**。
線形 +64C≈+4.5GB(=64 expert×40層×1.77MB)。base(非expert+overhead)≈2.4GB。**full model full-resident=20.5GB**。
RAM tier は「RSS + macOS/他アプリ headroom(~4-8GB)」で決まる(C=256 を物理 RAM ぎりぎりに置くと memory pressure→
expert ページ eviction/swap で実質 streaming に転落＝full-resident の利得消失):

| RAM | mode | C(hot-pin) | RSS | headroom | 代表機 |
|---|---|---|---|---|---|
| **8GB** | **streaming**(expert を pread demand-load) | C≈64 | 6.9GB | ~1GB(厳しい) | Neo, M1–M3 Air, M1/M2 mini, base M3 MBP |
| 16GB | partial-resident | C≈128 | 11.4GB | ~4.6GB | M4 Air/mini/base MBP |
| **24GB** | **near-full**(partial, miss 数%) | **C≈192–208** | 15.9–17GB | ~7-8GB | M2/M3 Air 上位, M-Pro entry |
| 32–36GB | **full-resident** | **C=256** | 20.5GB | ~11GB(快適) | base M5, Pro |
| 48–64GB+ | full-resident | C=256 | 20.5GB | 余裕大 | Max |
| 128–512GB | full-resident | C=256 | 20.5GB | 余裕大 | Max/Ultra |

★**24GB で C=256(20.5GB)は数値上可だが marginal**: 残り ~3.5GB のみ＝memory pressure リスク。**安全な
full-resident tier は 32GB から**。24GB は near-full(C≈192-208)が現実解(全 expert の 75-81% 常駐、miss 数%)。
maxK は全 mode で **C×3/8**（[[status-8gb-done-16gb-next]]）。8GB は SSD BW が下位の決定変数、16GB+ は
forward 律速で C 非依存（[[greedy-wall-fixed-overhead]] の C=64↔128 で床不変と整合）。

## 4. オンデバイスで実測すべきもの（calibration 計画）

起動時 <1秒 のキャリブで派生決定変数を取る。**計測インフラはセッションで構築済**:
- **cost(L) = a + b·L 係数**: `forward-cost`(QWISP_RUN=forward-cost)。a=forward 床, b=marginal。speculation 損益分岐＝a/b。
- **実効 SSD read 帯域**: `device-probe`(QWISP_RUN=device-probe, F_NOCACHE で 3 経路)。★mmap coalescing リスクは
  **解消済**(下 §5-2)。warm/cold は probe が自己判定(BW>8GB/s=warm)、cold は target 起動時に自然取得。
- **streaming IO 税 / miss penalty**: hot-pin 後の miss-load コスト。C の最適値（miss 0 になる最小 C）を実測。
- **dispatch/launch 特性**: `DispatchBench`/`ExpertBitBench`。bit 幅・融合の損益（device で逆転しうる）。
- 結果は device-ID キーで cache し、既知機は再計測スキップ（fallback に本 matrix の値）。

## 5. engine 設計に効く hardware 洞察（要記憶）

1. **256GB 単一 NAND**: 8GB streaming 機の SSD は ~1.5GB/s（Neo）まで落ちる＝streaming 予算の床。M1 Air(2 ダイ ~2.7)を床と誤認しない。
2. **mmap coalescing リスク=概ね解消**（2026-06-28, DeviceProbe, commit 8fe2d72）。**現状 pread 採用ゆえ元々非該当**＋probe で mmap≥pread・**random≈sequential(0.98-1.12x)=1.7MB は large-block でペナルティ無**を確認(cache 非依存の設計知見)。⚠️ dev 機は RAM 潤沢で warm(14-18GB/s=メモリ, SSD でない)→ cold は target 起動時。cold mmap-fault は未確認だが pread 経路ゆえ engine に無関係。
3. **M3 Pro 帯域回帰(150)**: memory-bound ベンチで M2 Pro(200)より遅い逆転に注意。
4. **Max 系は GPU bin で BW 2 値**（M3+）。device 判定は GPU コア数で。実測すれば自動で正しい値。
5. **公称 compute(TFLOPS)は M3+ 非公開**＝スペック予測でなく実測必須を補強。
6. **2026 RAM 供給逼迫**で上位 RAM tier が一部撤回（M3 Ultra 512→256 等）。tier matrix は時点依存。

## 次アクション
- engine の calibration layer 実装: 起動時 forward-cost＋SSD/mmap probe → cost model → C/maxK/mode/prefetch 自動設定。
- 最優先 probe = mmap vs pread（§4, 8GB streaming の経路選択）。
- 継続調査 agent: compute=aab228f63523fc991 / storage=abea498243dc1a275。
