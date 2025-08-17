#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "../include/gemm.h"

namespace gemm {

// GEMM kernel using Tensor Cores with WMMA
__global__ void gemm_tensor_cores_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // WMMA dimensions
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    // Block dimensions
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    
    // Shared memory for tiling
    __shared__ __nv_bfloat16 As[BLOCK_M][BLOCK_K];
    __shared__ __nv_bfloat16 Bs[BLOCK_K][BLOCK_N];
    
    // Warp and lane identification
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    
    // Compute warp's starting position
    int warp_row = (blockIdx.y * BLOCK_M + warp_id * WMMA_M) / WMMA_M;
    int warp_col = (blockIdx.x * BLOCK_N + threadIdx.y * WMMA_N) / WMMA_N;
    
    // Use nvcuda::wmma
    using namespace nvcuda::wmma;
    
    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize the accumulator fragment
    fill_fragment(c_frag, 0.0f);
    
    // Loop over all K-dimension tiles
    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {
        // Collaboratively load A and B tiles to shared memory
        
        // Calculate A tile indices
        int a_row = blockIdx.y * BLOCK_M + threadIdx.y;
        
        // Load A tile (row major)
        for (int i = threadIdx.x; i < BLOCK_K; i += blockDim.x) {
            if (a_row < M && tile_k + i < K) {
                As[threadIdx.y][i] = A[a_row * K + tile_k + i];
            } else {
                As[threadIdx.y][i] = __nv_bfloat16(0.0f);
            }
        }
        
        // Calculate B tile indices
        int b_col = blockIdx.x * BLOCK_N + threadIdx.y;
        
        // Load B tile (column major)
        for (int i = threadIdx.x; i < BLOCK_K; i += blockDim.x) {
            if (b_col < N && tile_k + i < K) {
                Bs[i][threadIdx.y] = B[b_col * K + tile_k + i];
            } else {
                Bs[i][threadIdx.y] = __nv_bfloat16(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute using tensor cores
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Load the inputs
            load_matrix_sync(a_frag, &As[warp_id * WMMA_M][k], BLOCK_K);
            load_matrix_sync(b_frag, &Bs[k][warp_id * WMMA_N], BLOCK_N);
            
            // Perform the matrix multiplication
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // Store the results
    int c_row = blockIdx.y * BLOCK_M + warp_id * WMMA_M;
    int c_col = blockIdx.x * BLOCK_N + warp_id * WMMA_N;
    
    if (c_row < M && c_col < N) {
        // Compute output address
        __nv_bfloat16* C_warp = &C[c_row * N + c_col];
        
        // Store result
        store_matrix_sync(C_warp, c_frag, N, row_major);
    }
}

void gemm_tensor_cores(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Block dimensions
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 128;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    
    // Define grid and block dimensions
    dim3 threads(32, 8); // 32 threads per warp, 8 warps per block
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    // Launch kernel
    gemm_tensor_cores_kernel<<<grid, threads>>>(A, B, C);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in tensor_cores: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

} // namespace gemm 