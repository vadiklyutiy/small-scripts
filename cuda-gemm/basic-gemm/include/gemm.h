#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace gemm {

// Matrix dimensions 
constexpr int M = 8;      // Number of rows in A and C
constexpr int K = 4096;   // Number of columns in A and rows in B
constexpr int N = 28672;  // Number of columns in B and C

// Block sizes
constexpr int BM = 2;     // Block size for M dimension
constexpr int BN = 64;    // Block size for N dimension

// Function declaration with __nv_bfloat16 type
void gemm_basic(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C);

} // namespace gemm 