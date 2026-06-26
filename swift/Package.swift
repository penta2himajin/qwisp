// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwisp",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "QwispCore", targets: ["QwispCore"]),
        .executable(name: "qwisp-poc", targets: ["qwisp-poc"]),
    ],
    dependencies: [
        // Python 版 mlx 0.31.2 と整合する mlx-swift 0.31.x
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
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
    ]
)
