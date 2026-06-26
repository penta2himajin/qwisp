# Qwisp — Swift PoC (mlx-swift)

Python 版エンジン（`../qwisp/`）の Swift 移植 PoC。最終目標は SwiftUI 付き macOS アプリ。
Python 版は削除せず温存（参照実装・ビット一致比較の基準）。

## 動機

Python 版の実測で 8GB streaming の per-forward ~6-7ms（~8%）が Python 固有オーバーヘッド
（tolist の list 構築 4.3ms + op dispatch 1.1ms + np 演算 1.1ms）と判明。残りは Metal/mlx-core
の仕事で言語非依存。Swift 移植の狙い:
1. ~6-7ms（~10%）の確実な回収
2. **Metal buffer の lifecycle を握ることで、Python mlx で詰んだ手法群を再試行**
   （持続バッファの in-place 更新／no-copy mmap／concat 排除 — SwiftLM が 122B@10.6GB を実現する所以）

## ビルド（重要な前提）

**`swift build`（SwiftPM CLI）は Metal シェーダをコンパイルできない**ため GPU が動かない
（"Failed to load the default metallib"）。`xcodebuild` を使うこと（mlx-swift README 準拠）。

Xcode 26 では Metal Toolchain が分離されたので初回のみ:

```sh
xcodebuild -downloadComponent MetalToolchain   # 一度だけ（~688MB）
```

ビルド & 実行:

```sh
cd swift
xcodebuild build -scheme qwisp-poc -destination 'platform=macOS' \
  -derivedDataPath ./.xcode-build -skipPackagePluginValidation
# 実行（metallib を含む bundle が executable 隣に出る）
$(find .xcode-build -name qwisp-poc -type f -perm +111 | head -1)
```

## マイルストーン

- **M0** ✅ build + smoke（mlx-swift 疎通・GPU 演算）
- **M1** ✅ gatherQuantizedMatmul を Python とビット一致（`qwisp.swift_ref` が参照生成）
- **M2a** ✅ switch_mlp forward（gather_qmm×3+swiglu）を持続 arena で bit一致
- **M2b** 全40層+attention forward → end-to-end decode tok/s（release build, pipeline）
- **M2c** MTP head + 投機ループ
- **M3** ✅ 持続 MTLBuffer の in-place 更新 viable（5µs, Python 1.4ms）= concat 排除の鍵
- **M4** SwiftUI アプリ化

## 構成

- `Sources/QwispCore/` — エンジン（library）
- `Sources/qwisp-poc/` — 検証 CLI
- 参照生成: `PY -m qwisp.swift_ref`（`/tmp/qwisp_ref.safetensors`）
