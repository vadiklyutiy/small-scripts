#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace gemm {

// Matrix dimensions for specific use case
constexpr int M = 8;      // Number of rows in A and C
constexpr int K = 4096;   // Number of columns in A and rows in B
constexpr int N = 28672;  // Number of columns in B and C

// Block sizes - optimized for tensor cores
constexpr int BM = 8;    // Block size for M dimension
constexpr int BN = 128;  // Block size for N dimension
constexpr int BK = 32;   // Block size for K dimension

// Warp-level tile sizes for tensor cores
constexpr int WM = 8;    // Warp tile size for M dimension
constexpr int WN = 32;   // Warp tile size for N dimension
constexpr int WK = 16;   // Warp tile size for K dimension

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

void gemm_tensor_cores(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C);

}  // namespace gemm 