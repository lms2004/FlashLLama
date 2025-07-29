#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// 🚀 FlashAttention CUDA kernel
// 该 kernel 实现了 block-wise、tile 化的高效注意力计算，显著降低显存占用，提升长序列推理速度。
__global__ void flash_attention_forward_kernel(const float* Q, const float* K, const float* V, int N, int d,
                                               int Tc, int Tr, int Bc, int Br, float softmax_scale,
                                               float* l, float* m, float* O) {
    int tx = threadIdx.x;              // 🧵 当前线程在线程块中的索引
    int bx = blockIdx.x, by = blockIdx.y;  // 🧱 当前线程块对应的 batch 和 head

    // 📦 计算当前 batch 和 head 对应的 Q/K/V/输出偏移
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); 
    int lm_offset  = (bx * gridDim.y * N) + (by * N);         // 用于 l 和 m 的偏移

    // 🧠 声明共享内存区：用于存 Q, K, V 片段 + 中间结果 S
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;                           // 当前 tile 的 Query
    float* Kj = &sram[tile_size];               // 当前 tile 的 Key
    float* Vj = &sram[tile_size * 2];           // 当前 tile 的 Value
    float* S  = &sram[tile_size * 3];           // 中间乘积结果 QK^T （注意：为 softmax 计算准备）

    // ⬅️ 遍历所有 Key/Value 的列 tile（即 attention 的 j 方向）
    for (int j = 0; j < Tc; j++) {
        // 📥 将当前 Kj 和 Vj 片段 load 到 shared memory
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // ⏸️ 等待所有线程加载完毕再进行下一步

        // ➡️ 遍历当前 Query 的行 tile（即 attention 的 i 方向）
        for (int i = 0; i < Tr; i++)  {
            // 📥 将当前 Qi 片段加载到 shared memory
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }

            // 📚 读取前一轮 l/m 值（用于数值稳定的 softmax）
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // 🧮 计算 attention 分数 S = Q × K^T，找出每行最大值 row_m（为 softmax 做数值稳定）
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                if (sum > row_m)
                    row_m = sum;
            }

            // 🔢 softmax 操作：P = exp(S - row_max)，同时计算每行和 row_l
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // 🔁 更新 row_l 和 row_m（用于累积式 softmax）
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

            // 🎯 写回输出 O：更新 O = 累加 softmax(P) * Vj，带上归一化项
            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                // 🎛️ 注意下面这一行是做数值稳定的累积 weighted sum：
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1 / row_l_new) * (
                        row_l_prev * __expf(row_m_prev - row_m_new) *
                        O[qkv_offset + (tile_size * i) + (tx * d) + x]
                        + __expf(row_m - row_m_new) * pv
                    );
            }

            // 🧾 更新 l 和 m 缓存，用于后续 tile 合并
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }

        __syncthreads();  // 🧱 确保当前 tile 完整处理完，再进入下一个 tile
    }
}

// C++ 封装接口，供 mha_kernel_cu 调用
extern "C" void flash_attention_kernel_cu(const float* Q, const float* K, const float* V, int B, int nh, int N, int d,
                                           float* l, float* m, float* O, cudaStream_t stream) {
    // 🧩 Block 大小（tile 大小），可根据硬件动态调整
    const int Bc = 32, Br = 32;
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;
    float softmax_scale = 1.0f / sqrtf((float)d);
    size_t sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
    dim3 grid_dim(B, nh);    // 每个 (batch, head) 启动一个 block
    dim3 block_dim(Bc);      // 每个 block 启动 Bc 个线程（每个线程处理一个 token）
    // 🚀 启动 FlashAttention CUDA 核函数
    flash_attention_forward_kernel<<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
} 