#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Matrix dimensions
// A: 8 x 4096 (row major)
// B: 4096 x 28672 (column major)
// C: 8 x 28672 (row major)
#define M 8
#define K 4096
#define N 28672

namespace gemm {

// Common utility functions
// Helper function to initialize matrices
void initialize_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C);

// Helper function to check results
bool verify_result(const __nv_bfloat16* C, const __nv_bfloat16* C_ref);

// CPU reference implementation
void gemm_cpu_reference(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

namespace basic {
    // Basic GEMM implementation with simple tiling
    void gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

    #ifdef DEBUG_GEMM_BASIC
    // Debug functions for basic implementation
    void print_debug_info();
    #endif
}

// Add more namespaces for other implementations here
// namespace shared_mem { ... }
// namespace tensor_core { ... }

} // namespace gemm 