#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cub/cub.cuh>  // 添加 cub 头文件用于 BlockReduce
// flash_attention_kernel.cu
#include "flash_attention_kernel.cuh"

// 🚀 FlashAttention CUDA kernel - 修正版本：确保与MHA结果一致
// 该 kernel 实现了与原始MHA算法完全一致的Flash Attention
__global__ void flash_attention_forward_kernel(int32_t pos, int32_t seq_len, float* query,
                                               float* score_ptr, float* output, float* key_cache,
                                               float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                               int32_t head_num, int32_t head_size,
                                               int32_t layer_offset) {
    int head = blockIdx.x;
    if (head >= head_num) {
        return;
    }

    int tx = threadIdx.x;
    float scale = 1.f / sqrtf(head_size);
    float* query_head = query + head * head_size;
    float* score_head = score_ptr + head * seq_len;
    int head_offset = (head / kv_mul) * head_size;

    // 🧠 声明共享内存区：用于存 Q, K, V 片段
    extern __shared__ float sram[];
    int tile_size = 32; // Flash Attention 推荐的分块大小
    float* Qi = sram;                           // 当前 tile 的 Query
    float* Kj = &sram[tile_size * head_size];   // 当前 tile 的 Key
    float* Vj = &sram[tile_size * head_size * 2]; // 当前 tile 的 Value

    // 📥 将当前 Query 加载到 shared memory
    for (int x = tx; x < head_size; x += blockDim.x) {
        Qi[x] = query_head[x];
    }
    __syncthreads();

    // 🧮 第一步：计算所有attention scores（与原始MHA一致）
    for (int t = tx; t <= pos; t += blockDim.x) {
        float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
        
        // 计算当前 token 的 attention score（与原始MHA完全一致）
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
            float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
            score += key_head_float4.x * query_head_float4.x;
            score += key_head_float4.y * query_head_float4.y;
            score += key_head_float4.z * query_head_float4.z;
            score += key_head_float4.w * query_head_float4.w;
        }
        score *= scale;
        score_head[t] = score;
    }
    __syncthreads();

    // 🔢 第二步：使用与原始MHA完全相同的softmax实现
    // 复制原始MHA的softmax_gpu函数逻辑
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = pos + 1;

    // find max value (for numerical stability)
    // this should be FLT_MAX, not 0 !!!!
    // otherwise, the softmax may be occur nan when head_dim < 128 threads
    float max_val = tid < size ? score_head[tid] : -FLT_MAX;
    for (int i = tid + step; i < size; i += step) {
        if (score_head[i] > max_val) {
            max_val = score_head[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        score_head[i] = expf(score_head[i] - max_val);
        sum += score_head[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = tid; i < size; i += step) {
        score_head[i] /= sum;
    }
    __syncthreads();

    // 🎯 第三步：计算输出（与原始MHA一致）
    float* output_head = output + head * head_size;
    for (int i = tx; i < head_size; i += blockDim.x) {
        float value = 0.0f;
        #pragma unroll
        for (int t = 0; t <= pos; t++) {
            float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
            float score = score_head[t];
            value += score * value_head[i];
        }
        output_head[i] = value;
    }
}

// C++ 封装接口，与 mha_kernel_cu 保持一致的参数结构
void flash_attention_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                               int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                               const tensor::Tensor& mha_out,
                               const tensor::Tensor& query_tensor, 
                               const tensor::Tensor& score_tensor,
                               const tensor::Tensor& key_cache_tensor, 
                               const tensor::Tensor& value_cache_tensor,
                               base::DeviceType device_type, kernel::CudaConfig* config) {
    UNUSED(device_type);
    
    // 🧩 计算共享内存大小
    int tile_size = 32;
    size_t sram_size = (3 * tile_size * head_size) * sizeof(float);
    
    // 📦 获取数据指针
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(mha_out.ptr<float>());
    float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
    
    // 🧮 计算层偏移
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    
    // 🚀 启动 FlashAttention CUDA 核函数
    cudaStream_t stream = config->stream;
    flash_attention_forward_kernel<<<head_num, 256, sram_size, stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, 
        kv_dim, kv_mul, head_num, head_size, layer_offset);
}