// flash_attention_kernel.cuh
#pragma once
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cub/cub.cuh>
#include <float.h>  // C 标准库头文件
#include <base/cuda_config.h>

// FlashAttention kernel 函数声明 - 与 mha_kernel_cu 保持一致的参数结构
void flash_attention_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                               int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                               const tensor::Tensor& mha_out,
                               const tensor::Tensor& query_tensor, 
                               const tensor::Tensor& score_tensor,
                               const tensor::Tensor& key_cache_tensor, 
                               const tensor::Tensor& value_cache_tensor,
                               base::DeviceType device_type, kernel::CudaConfig* config);