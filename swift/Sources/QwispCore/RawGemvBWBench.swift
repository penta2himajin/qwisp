import Foundation
import Metal

/// raw-gemv-bw-bench: M=1 量子化 GEMV (qmm4 = MLX qmv_fast 移植) の達成 GB/s を実測し、
/// bandwidth-utilization campaign の go/no-go を計る microbench。モデル不要・production kernel 非改変。
///
/// 質問: 89 tok/s (11.2ms/step) の M=1 decode で、weight-read の主体である qmv が
/// peak(~400GB/s spec)の何% を出しているか? そして bit-exact な load-widening / row-tiling 再調整、
/// もしくは order-change split-K で GB/s は上がるか?
///
/// 方法: production shape (lm_head K=2048 N=248320、MoE gate/up K=2048 N=512、down K=512 N=2048) で
/// weight buffer(private, cold)を確保し、1-CB 内で R 回 dispatch を連鎖(小 shape は毎回別 region を
/// index して SLC を defeat)、gpuEndTime-gpuStartTime から GB/s を算出。
/// variants: (0)prod=MLX qmv_fast そのまま、(A/B/C)row-tiling 再調整(bit-exact)、
/// (V)uint2 幅広 weight load(bit-exact)、(S2)split-K 2-pass(order-change=非 bit-exact)、
/// (R)pure-read ceiling(dequant 無し、この access pattern の memory 上限)。
public enum RawGemvBWBench {

    // production qmv_fast の row-tiling パラメータだけを差し替えた kernel を生成。
    // nsg=num_simdgroups, rps=results_per_simdgroup。per-row の K-walk / 累積順は不変 → bit-exact。
    // vec=true で weight を uint2(8B) 1 発 load に(dequant 結果は同一 → bit-exact)。
    static func qmvSrc(name: String, nsg: Int, rps: Int, vec: Bool) -> String {
        let qdotBody: String
        if vec {
            // 8 bytes(=packs_per_thread*bytes_per_pack)を uint2 で 1 発 load、ushort4 に再解釈。
            qdotBody = """
                uint2 wp = *(const device uint2*)w;         // 8B を 1 発 load
                ushort4 ws = as_type<ushort4>(wp);           // 4×uint16 に再解釈
                accum += (xt[0]*(float)(ws.x & 0x000f) + xt[1]*(float)(ws.x & 0x00f0) + xt[2]*(float)(ws.x & 0x0f00) + xt[3]*(float)(ws.x & 0xf000));
                accum += (xt[4]*(float)(ws.y & 0x000f) + xt[5]*(float)(ws.y & 0x00f0) + xt[6]*(float)(ws.y & 0x0f00) + xt[7]*(float)(ws.y & 0xf000));
                accum += (xt[8]*(float)(ws.z & 0x000f) + xt[9]*(float)(ws.z & 0x00f0) + xt[10]*(float)(ws.z & 0x0f00) + xt[11]*(float)(ws.z & 0xf000));
                accum += (xt[12]*(float)(ws.w & 0x000f) + xt[13]*(float)(ws.w & 0x00f0) + xt[14]*(float)(ws.w & 0x0f00) + xt[15]*(float)(ws.w & 0xf000));
            """
        } else {
            qdotBody = """
                const device uint16_t* ws = (const device uint16_t*)w;
                for (int i = 0; i < 4; i++) {
                    accum += (xt[4*i]   * (float)(ws[i] & 0x000f) +
                              xt[4*i+1] * (float)(ws[i] & 0x00f0) +
                              xt[4*i+2] * (float)(ws[i] & 0x0f00) +
                              xt[4*i+3] * (float)(ws[i] & 0xf000));
                }
            """
        }
        return """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
        inline float ld16_\(name)(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 16; i += 4) {
                sum += x[i] + x[i+1] + x[i+2] + x[i+3];
                xt[i]=x[i]; xt[i+1]=x[i+1]/16.0f; xt[i+2]=x[i+2]/256.0f; xt[i+3]=x[i+3]/4096.0f;
            }
            return sum;
        }
        inline float qd4_\(name)(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f;
            \(qdotBody)
            return scale * accum + sum * bias;
        }
        kernel void \(name)(device const uint32_t* w [[buffer(0)]],
                            device const half* scales [[buffer(1)]],
                            device const half* biases [[buffer(2)]],
                            device const half* x [[buffer(3)]],
                            device half* y [[buffer(4)]],
                            constant int& in_vec_size [[buffer(5)]],
                            constant int& out_vec_size [[buffer(6)]],
                            constant uint& wOff [[buffer(7)]],   // 別 region 選択 (uint32 words)
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
            constexpr int packs_per_thread = 2;
            constexpr int num_simdgroups = \(nsg);
            constexpr int results_per_simdgroup = \(rps);
            constexpr int pack_factor = 8, bytes_per_pack = 4, values_per_thread = 16;
            constexpr int block_size = 512, scale_step_per_thread = 4;
            const device uint8_t* ws = (const device uint8_t*)(w + wOff);
            thread float x_thread[16];
            thread float result[\(rps)] = {0};
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int in_vec_size_g = in_vec_size / 64;
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            x += tid.x * in_vec_size + simd_lid * values_per_thread;
            y += tid.x * out_vec_size + out_row;
            for (int k = 0; k < in_vec_size; k += block_size) {
                float sum = ld16_\(name)(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device half* sl = scales + row * in_vec_size_g;
                    const device half* bl = biases + row * in_vec_size_g;
                    result[row] += qd4_\(name)(wl, x_thread, (float)sl[0], (float)bl[0], sum);
                }
                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64; biases += block_size / 64; x += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                result[row] = simd_sum(result[row]);
                if (simd_lid == 0) y[row] = (half)result[row];
            }
        }
        """
    }

    // pure-read ceiling: qmv と同一 striding で weight を読み simd_sum に落とす(dequant/ALU 最小)。
    static func readSrc() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;
        kernel void raw_read(device const uint32_t* w [[buffer(0)]],
                             device half* y [[buffer(4)]],
                             constant int& in_vec_size [[buffer(5)]],
                             constant int& out_vec_size [[buffer(6)]],
                             constant uint& wOff [[buffer(7)]],
                             uint3 tid [[threadgroup_position_in_grid]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
            constexpr int packs_per_thread = 2;
            constexpr int num_simdgroups = 2, results_per_simdgroup = 4;
            constexpr int pack_factor = 8, bytes_per_pack = 4;
            constexpr int block_size = 512;
            const device uint8_t* ws = (const device uint8_t*)(w + wOff);
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            float acc = 0.0f;
            for (int k = 0; k < in_vec_size; k += block_size) {
                for (int row = 0; row < results_per_simdgroup; row++) {
                    uint2 wp = *(const device uint2*)(ws + row * in_vec_size_w);
                    acc += (float)(wp.x + wp.y);
                }
                ws += block_size * bytes_per_pack / pack_factor;
            }
            acc = simd_sum(acc);
            if (simd_lid == 0 && out_row < out_vec_size) y[out_row] = (half)acc;
        }
        """
    }

    // split-K P-pass: 各 threadgroup が K の 1/P 区間を担当(tid.z=p)、partial[p*N+row] へ。
    // 累積順が変わる(P 個の partial を後段で加算)→ 非 bit-exact。BW headroom 上限測定用。
    static func splitKSrc(P: Int) -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
        inline float ld16s(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 16; i += 4) {
                sum += x[i]+x[i+1]+x[i+2]+x[i+3];
                xt[i]=x[i]; xt[i+1]=x[i+1]/16.0f; xt[i+2]=x[i+2]/256.0f; xt[i+3]=x[i+3]/4096.0f;
            }
            return sum;
        }
        inline float qd4s(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f;
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < 4; i++)
                accum += (xt[4*i]*(float)(ws[i]&0x000f)+xt[4*i+1]*(float)(ws[i]&0x00f0)+xt[4*i+2]*(float)(ws[i]&0x0f00)+xt[4*i+3]*(float)(ws[i]&0xf000));
            return scale*accum + sum*bias;
        }
        kernel void split_k(device const uint32_t* w [[buffer(0)]],
                            device const half* scales [[buffer(1)]],
                            device const half* biases [[buffer(2)]],
                            device const half* x [[buffer(3)]],
                            device float* partial [[buffer(4)]],   // [P, N]
                            constant int& in_vec_size [[buffer(5)]],
                            constant int& out_vec_size [[buffer(6)]],
                            constant uint& wOff [[buffer(7)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
            constexpr int P = \(P);
            constexpr int packs_per_thread = 2, num_simdgroups = 2, results_per_simdgroup = 4;
            constexpr int pack_factor = 8, bytes_per_pack = 4, values_per_thread = 16;
            constexpr int block_size = 512, scale_step_per_thread = 4;
            const device uint8_t* ws = (const device uint8_t*)(w + wOff);
            thread float x_thread[16];
            thread float result[4] = {0};
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int in_vec_size_g = in_vec_size / 64;
            const int kseg = in_vec_size / P;              // 各 p の K 区間長 (512 の倍数)
            const uint p = tid.z;
            const int kstart = p * kseg;
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws     += out_row * in_vec_size_w + (kstart*bytes_per_pack/pack_factor) + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + (kstart/64) + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + (kstart/64) + simd_lid / scale_step_per_thread;
            x += kstart + simd_lid * values_per_thread;
            for (int k = 0; k < kseg; k += block_size) {
                float sum = ld16s(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    result[row] += qd4s(wl, x_thread, (float)scales[row*in_vec_size_g], (float)biases[row*in_vec_size_g], sum);
                }
                ws += block_size*bytes_per_pack/pack_factor;
                scales += block_size/64; biases += block_size/64; x += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                float r = simd_sum(result[row]);
                if (simd_lid == 0) partial[p*out_vec_size + out_row + row] = r;
            }
        }
        kernel void split_k_reduce(device const float* partial [[buffer(0)]],
                                   device half* y [[buffer(1)]],
                                   constant int& out_vec_size [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
            constexpr int P = \(P);
            if ((int)gid >= out_vec_size) return;
            float s = 0.0f;
            for (int p = 0; p < P; p++) s += partial[p*out_vec_size + gid];
            y[gid] = (half)s;
        }
        """
    }

    // 実 MoE path (gqmm4): grid depth=Ktop, 各 expert が別 region。8 experts を 1 dispatch で。
    static func gatherSrc() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
        inline float ld16g(const device half* x, thread float* xt) {
            float sum=0.0f;
            for (int i=0;i<16;i+=4){ sum+=x[i]+x[i+1]+x[i+2]+x[i+3]; xt[i]=x[i];xt[i+1]=x[i+1]/16.0f;xt[i+2]=x[i+2]/256.0f;xt[i+3]=x[i+3]/4096.0f;}
            return sum;
        }
        inline float qd4g(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum){
            float accum=0.0f; const device uint16_t* ws=(const device uint16_t*)w;
            for (int i=0;i<4;i++) accum+=(xt[4*i]*(float)(ws[i]&0x000f)+xt[4*i+1]*(float)(ws[i]&0x00f0)+xt[4*i+2]*(float)(ws[i]&0x0f00)+xt[4*i+3]*(float)(ws[i]&0xf000));
            return scale*accum+sum*bias;
        }
        kernel void qmv_gather(device const uint32_t* w [[buffer(0)]],
                               device const half* scales [[buffer(1)]],
                               device const half* biases [[buffer(2)]],
                               device const half* x [[buffer(3)]],
                               device half* y [[buffer(4)]],
                               constant int& in_vec_size [[buffer(5)]],
                               constant int& out_vec_size [[buffer(6)]],
                               constant uint& expertStrideW [[buffer(7)]],  // uint32 words per expert
                               uint3 tid [[threadgroup_position_in_grid]],
                               uint simd_gid [[simdgroup_index_in_threadgroup]],
                               uint simd_lid [[thread_index_in_simdgroup]]) {
            constexpr int packs_per_thread=2, num_simdgroups=2, results_per_simdgroup=4;
            constexpr int pack_factor=8, bytes_per_pack=4, values_per_thread=16;
            constexpr int block_size=512, scale_step_per_thread=4;
            uint e = tid.z;
            const device uint8_t* ws=(const device uint8_t*)(w + e*expertStrideW);
            thread float x_thread[16]; thread float result[4]={0};
            const int in_vec_size_w=in_vec_size*bytes_per_pack/pack_factor;
            const int in_vec_size_g=in_vec_size/64;
            const int out_row=tid.y*(num_simdgroups*results_per_simdgroup)+simd_gid*results_per_simdgroup;
            ws+=out_row*in_vec_size_w+simd_lid*packs_per_thread*bytes_per_pack;
            scales+=out_row*in_vec_size_g+simd_lid/scale_step_per_thread;
            biases+=out_row*in_vec_size_g+simd_lid/scale_step_per_thread;
            x+=simd_lid*values_per_thread;
            y+=e*out_vec_size+out_row;
            for (int k=0;k<in_vec_size;k+=block_size){
                float sum=ld16g(x,x_thread);
                for (int row=0;row<results_per_simdgroup;row++){
                    auto wl=(const device uint8_t*)(ws+row*in_vec_size_w);
                    result[row]+=qd4g(wl,x_thread,(float)scales[row*in_vec_size_g],(float)biases[row*in_vec_size_g],sum);
                }
                ws+=block_size*bytes_per_pack/pack_factor; scales+=block_size/64; biases+=block_size/64; x+=block_size;
            }
            for (int row=0;row<results_per_simdgroup;row++){ result[row]=simd_sum(result[row]); if(simd_lid==0) y[row]=(half)result[row]; }
        }
        """
    }

    struct Shape { let name: String; let K: Int; let N: Int }

    public static func run() -> String {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return "[gemv-bw] no device" }

        func pso(_ src: String, _ fn: String) -> MTLComputePipelineState? {
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 return try device.makeComputePipelineState(function: lib.makeFunction(name: fn)!)
            } catch { print("[gemv-bw] compile \(fn): \(error)"); return nil }
        }

        // variants (row-tiling / vec / read)
        let variants: [(String, Int, Int, Bool)] = [
            ("prod",  2, 4, false),   // MLX qmv_fast そのまま
            ("nsg4",  4, 2, false),   // 4 simdgroups/TG, 2 rows each (bit-exact)
            ("nsg1",  1, 8, false),   // 1 simdgroup/TG, 8 rows (bit-exact)
            ("rps2",  2, 2, false),   // 4 rows/TG → grid N/4 で 2x threadgroups (bit-exact)
            ("vec",   2, 4, true),    // uint2 幅広 load (bit-exact)
        ]
        var psos: [(String, MTLComputePipelineState, Int, Int)] = []
        for (nm, nsg, rps, vec) in variants {
            guard let p = pso(qmvSrc(name: nm, nsg: nsg, rps: rps, vec: vec), nm) else { continue }
            psos.append((nm, p, nsg, rps))
        }
        guard let readP = pso(readSrc(), "raw_read") else { return "[gemv-bw] raw_read compile fail" }
        var splitPSOs: [(Int, MTLComputePipelineState, MTLComputePipelineState)] = []
        for P in [2, 4] {
            let src = splitKSrc(P: P)
            if let a = pso(src, "split_k"), let b = pso(src, "split_k_reduce") { splitPSOs.append((P, a, b)) }
        }

        let shapes = [
            Shape(name: "lmhead  K2048 N248320", K: 2048, N: 248320),  // cold DRAM: honest ceiling
            Shape(name: "moe_gu  K2048 N512   ", K: 2048, N: 512),     // gate/up
            Shape(name: "moe_dn  K512  N2048  ", K: 512,  N: 2048),    // down
            Shape(name: "attn_o  K2048 N2048  ", K: 2048, N: 2048),    // proj
        ]

        // 大きな weight プール(cold を強制、小 shape も別 region index)。lm_head 単体で 254MB。
        // pool = max(254MB, ...)。単一 shape あたり weightBytes = N*K/2。
        let poolBytes = 320 * 1024 * 1024
        guard let wPool = device.makeBuffer(length: poolBytes, options: .storageModePrivate) else {
            return "[gemv-bw] pool alloc fail"
        }
        // scales/biases/x/y は小さめ確保(最大 shape 基準)
        let maxN = 248320, maxK = 2048
        let scBytes = (maxN * (maxK / 64)) * 2
        guard let scBuf = device.makeBuffer(length: scBytes, options: .storageModePrivate),
              let biBuf = device.makeBuffer(length: scBytes, options: .storageModePrivate),
              let xBuf  = device.makeBuffer(length: maxK * 2, options: .storageModePrivate),
              let yBuf  = device.makeBuffer(length: maxN * 4, options: .storageModePrivate),
              let partBuf = device.makeBuffer(length: maxN * 4 * 4, options: .storageModePrivate)
        else { return "[gemv-bw] aux alloc fail" }

        var lines: [String] = ["[gemv-bw] M=1 quantized GEMV achieved bandwidth (private/cold, gpuEndTime-gpuStartTime)"]
        lines.append(String(format: "  device=%@ peak spec ~400 GB/s (M1 Max) — measure %% of that", device.name))

        func timeDispatch(_ shape: Shape, reps: Int, encode: (MTLComputeCommandEncoder, Int) -> Void, dispatches: Int) -> Double {
            var gpu: [Double] = []
            for r in 0..<(reps+1) {
                guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return -1 }
                for i in 0..<dispatches { encode(enc, i) }
                enc.endEncoding()
                cb.commit(); cb.waitUntilCompleted()
                if r > 0 { gpu.append((cb.gpuEndTime - cb.gpuStartTime) * 1000.0) }  // ms for all dispatches
            }
            gpu.sort(); return gpu[gpu.count/2]
        }

        for shape in shapes {
            let K = shape.K, N = shape.N
            let weightBytes = N * (K / 2)                 // 4-bit
            let scaleBytes = N * (K / 64) * 2 * 2         // scales+biases half
            let bytesPerDispatch = Double(weightBytes + scaleBytes)
            // 別 region 数: pool を weightBytes で割った数(最大 8)。lm_head は 1(既に cold)。
            let regionWords = weightBytes / 4
            let nRegion = max(1, min(8, poolBytes / max(1, weightBytes)))
            let dispatches = max(1, min(8, nRegion))
            var kk = Int32(K), nn = Int32(N)
            lines.append("  --- \(shape.name)  (weight \(weightBytes/1024)KB, \(dispatches) cold dispatches/CB) ---")

            for (nm, p, nsg, rps) in psos {
                let rowsPerTG = nsg * rps
                let gridN = N / rowsPerTG
                let gpuMs = timeDispatch(shape, reps: 8, encode: { enc, i in
                    enc.setComputePipelineState(p)
                    enc.setBuffer(wPool, offset: 0, index: 0)
                    enc.setBuffer(scBuf, offset: 0, index: 1)
                    enc.setBuffer(biBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3)
                    enc.setBuffer(yBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5)
                    enc.setBytes(&nn, length: 4, index: 6)
                    var wOff = UInt32((i % nRegion) * regionWords) ; if UInt64((i % nRegion)) * UInt64(regionWords) + UInt64(regionWords) > UInt64(poolBytes/4) { wOff = 0 }
                    enc.setBytes(&wOff, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: gridN, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
                }, dispatches: dispatches)
                let secPer = (gpuMs / 1000.0) / Double(dispatches)
                let gbps = bytesPerDispatch / secPer / 1e9
                let tag = nm.padding(toLength: 6, withPad: " ", startingAt: 0)
                lines.append(String(format: "    %@ nsg=%d rps=%d : %7.3f ms/disp  %6.1f GB/s  (%4.1f%% of 400)",
                                    tag, nsg, rps, secPer*1000, gbps, gbps/400*100))
            }
            // pure-read ceiling
            do {
                let gpuMs = timeDispatch(shape, reps: 8, encode: { enc, i in
                    enc.setComputePipelineState(readP)
                    enc.setBuffer(wPool, offset: 0, index: 0)
                    enc.setBuffer(yBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5)
                    enc.setBytes(&nn, length: 4, index: 6)
                    var wOff = UInt32((i % nRegion) * regionWords)
                    enc.setBytes(&wOff, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N/8, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }, dispatches: dispatches)
                let secPer = (gpuMs / 1000.0) / Double(dispatches)
                let gbps = Double(weightBytes) / secPer / 1e9
                lines.append(String(format: "    READ            : %7.3f ms/disp  %6.1f GB/s  (%4.1f%% of 400)  [ceiling, no dequant]",
                                    secPer*1000, gbps, gbps/400*100))
            }
            // split-K (2-pass, order-change)
            for (P, sp, rd) in splitPSOs {
                if K % (P*512) != 0 { continue }
                let gpuMs = timeDispatch(shape, reps: 8, encode: { enc, i in
                    enc.setComputePipelineState(sp)
                    enc.setBuffer(wPool, offset: 0, index: 0)
                    enc.setBuffer(scBuf, offset: 0, index: 1)
                    enc.setBuffer(biBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3)
                    enc.setBuffer(partBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5)
                    enc.setBytes(&nn, length: 4, index: 6)
                    var wOff = UInt32((i % nRegion) * regionWords)
                    enc.setBytes(&wOff, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N/8, depth: P),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                    enc.setComputePipelineState(rd)
                    enc.setBuffer(partBuf, offset: 0, index: 0)
                    enc.setBuffer(yBuf, offset: 0, index: 1)
                    enc.setBytes(&nn, length: 4, index: 2)
                    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }, dispatches: dispatches)
                let secPer = (gpuMs / 1000.0) / Double(dispatches)
                let gbps = bytesPerDispatch / secPer / 1e9
                lines.append(String(format: "    splitK P=%d       : %7.3f ms/disp  %6.1f GB/s  (%4.1f%% of 400)  [order-change, 2-pass]",
                                    P, secPer*1000, gbps, gbps/400*100))
            }
        }
        // 実 MoE path: 8 experts を 1 dispatch (gqmm4 と同形状, 4MB working set)
        if let gp = pso(gatherSrc(), "qmv_gather") {
            lines.append("  --- MoE realistic (gqmm4 form: Ktop=8 experts/dispatch) ---")
            for (nm, K, N) in [("gate/up K2048 N512", 2048, 512), ("down   K512 N2048", 512, 2048)] {
                var kk = Int32(K), nn = Int32(N)
                let expertStrideW = (N * (K/2)) / 4    // uint32 words per expert
                let weightBytes = 8 * N * (K/2)        // 8 experts
                var gpu: [Double] = []
                for r in 0..<9 {
                    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
                    enc.setComputePipelineState(gp)
                    enc.setBuffer(wPool, offset: 0, index: 0)
                    enc.setBuffer(scBuf, offset: 0, index: 1)
                    enc.setBuffer(biBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3)
                    enc.setBuffer(yBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5)
                    enc.setBytes(&nn, length: 4, index: 6)
                    var es = UInt32(expertStrideW); enc.setBytes(&es, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N/8, depth: 8),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    if r > 0 { gpu.append((cb.gpuEndTime - cb.gpuStartTime) * 1000.0) }
                }
                gpu.sort(); let ms = gpu[gpu.count/2]
                let gbps = Double(weightBytes) / (ms/1000.0) / 1e9
                lines.append(String(format: "    %@ (8exp, %dMB): %7.3f ms  %6.1f GB/s  (%4.1f%% of 400)",
                                    nm, weightBytes/1024/1024, ms, gbps, gbps/400*100))
            }
        }
        lines.append("[gemv-bw] go/no-go: 最速 bit-exact variant が prod 比 <15% なら NO-GO。READ ceiling が真の上限。")
        return lines.joined(separator: "\n")
    }
}
