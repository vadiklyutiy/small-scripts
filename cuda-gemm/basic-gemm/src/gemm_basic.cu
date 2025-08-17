#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/gemm.h"

namespace gemm {

// Basic GEMM kernel with simple tiling
__global__ void gemm_basic_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    constexpr int BM = 2;  // Tile size for M dimension
    constexpr int BN = 64; // Tile size for N dimension
    
    // Calculate global row and column indices
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    
    // Check boundaries
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Accumulate dot product
        for (int k = 0; k < K; ++k) {
            float a_val = float(A[row * K + k]);        // A is row-major
            float b_val = float(B[col * K + k]);        // B is column-major
            sum += a_val * b_val;
        }
        
        // Store result in C (row-major)
        C[row * N + col] = __nv_bfloat16(sum);
    }
}

void gemm_basic(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Define grid and block dimensions
    constexpr int BM = 2;   // Block tile size for M dimension
    constexpr int BN = 64;  // Block tile size for N dimension
    
    dim3 threads(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Launch kernel
    gemm_basic_kernel<<<grid, threads>>>(A, B, C);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

} // namespace gemm 