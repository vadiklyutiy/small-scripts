#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace tma_gemm {

// Matrix dimensions as constants instead of macros
constexpr int M_DIM = 8;
constexpr int K_DIM = 4096;
constexpr int N_DIM = 28672;

// Helper function to initialize matrices
void initialize_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C);

// Helper function to check results
bool verify_result(const __nv_bfloat16* C, const __nv_bfloat16* C_ref);

// CPU reference implementation
void cpu_reference(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

// TMA optimized implementation
void gemm_tma(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

#ifdef DEBUG_TMA
// Debug functions
void print_debug_info();
#endif

} // namespace tma_gemm 