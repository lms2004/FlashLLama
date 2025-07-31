// flash_attention_kernel.cuh
#pragma once
void flash_attention_kernel_cu(const float* Q, const float* K, const float* V, int B, int nh, int N, int d,
                                           float* l, float* m, float* O, cudaStream_t stream);