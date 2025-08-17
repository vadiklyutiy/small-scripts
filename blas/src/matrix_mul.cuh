#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

// Constants for matrix dimensions
constexpr int M = 8;        // Rows of A
constexpr int K = 4096;     // Cols of A, Rows of B
constexpr int N = 28672;    // Cols of B

// Result: C = A * B
// A is M x K (row major)
// B is K x N (column major)
// C is M x N (row major)

// Different cuBLAS matrix multiplication implementations
// Each returns execution time in milliseconds

// Implementation using cublasGemmEx
float gemm_cublasGemmEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

// Implementation using cublasGemmBatchedEx
float gemm_cublasGemmBatchedEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

// Implementation using cublasGemmStridedBatchedEx
float gemm_cublasGemmStridedBatchedEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C);

// Utility functions
void check_result(const __nv_bfloat16* C, const __nv_bfloat16* C_ref, const char* name);
void init_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C);
void compute_reference_cpu(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C_ref); 