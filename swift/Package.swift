// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwisp",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "QwispCore", targets: ["QwispCore"]),
        .executable(name: "qwisp-poc", targets: ["qwisp-poc"]),   // bench/gate binary
        .executable(name: "qwisp", targets: ["qwisp"]),           // product CLI + server
    ],
    dependencies: [
        // Python 版 mlx 0.31.2 と整合する mlx-swift 0.31.x
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        // OpenAI 互換 HTTP サーバ（NIO ベース、SSE/keep-alive を正しく扱う）
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        // text↔ids + Qwen chat_template（自作せず既存を使う）
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "QwispCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "qwisp-poc",
            dependencies: ["QwispCore"]
        ),
        .executableTarget(
            name: "qwisp",
            dependencies: [
                "QwispCore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
    ]
)
