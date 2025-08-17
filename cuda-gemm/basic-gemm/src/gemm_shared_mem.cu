#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/gemm.h"

namespace gemm {

// GEMM kernel with shared memory tiling
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K>
__global__ void gemm_shared_mem_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Block row and column
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // Thread row and column within tile
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Global row and column indices for output
    int global_row = block_row * BLOCK_SIZE_M + thread_row;
    int global_col = block_col * BLOCK_SIZE_N + thread_col;
    
    // Accumulate result for C(global_row, global_col)
    float sum = 0.0f;
    
    // Loop over all tiles
    int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load A tile (row major) into shared memory
        if (global_row < M && tile * BLOCK_SIZE_K + thread_col < K) {
            As[thread_row][thread_col] = float(A[global_row * K + tile * BLOCK_SIZE_K + thread_col]);
        } else {
            As[thread_row][thread_col] = 0.0f;
        }
        
        // Load B tile (column major) into shared memory
        if (global_col < N && tile * BLOCK_SIZE_K + thread_row < K) {
            Bs[thread_row][thread_col] = float(B[global_col * K + tile * BLOCK_SIZE_K + thread_row]);
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Store result in C (row-major)
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = __nv_bfloat16(sum);
    }
}

void gemm_shared_mem(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Define block sizes
    constexpr int BLOCK_SIZE_M = 8;    // Full M dimension (8)
    constexpr int BLOCK_SIZE_N = 128;  // Block size for N dimension
    constexpr int BLOCK_SIZE_K = 32;   // Block size for K dimension
    
    // Define grid and block dimensions
    dim3 threads(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    // Launch kernel
    gemm_shared_mem_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<grid, threads>>>(A, B, C);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in shared_mem: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

} // namespace gemm 