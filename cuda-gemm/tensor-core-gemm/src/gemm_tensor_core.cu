#include "gemm_tensor_core.h"
#include <stdio.h>
#include <mma.h>

namespace gemm {

using namespace nvcuda;

// The kernel using WMMA API
__global__ void gemm_tensor_core_kernel(const float *A, const float *B, float *C) {
    // Warp ID and lane ID
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // WMMA fragment to store operands
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Matrix A and B base addresses
    const int aRow = warpM * WM;
    const int bCol = warpN * WN;
    
    // Loop over k dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Convert indices to offset into matrix data
        const int aOffset = aRow * K + k;
        const int bOffset = k * N + bCol;
        
        // Create temporary arrays of half precision values
        half a_half[WMMA_M * WMMA_K];
        half b_half[WMMA_K * WMMA_N];
        
        // Convert and load data into half precision arrays
        for (int i = 0; i < WMMA_M; i++) {
            for (int j = 0; j < WMMA_K; j++) {
                int idx = i * WMMA_K + j;
                if (aRow + i < M && k + j < K) {
                    a_half[idx] = __float2half(A[aOffset + i * K + j]);
                } else {
                    a_half[idx] = __float2half(0.0f);
                }
            }
        }
        
        for (int i = 0; i < WMMA_K; i++) {
            for (int j = 0; j < WMMA_N; j++) {
                int idx = i * WMMA_N + j;
                if (k + i < K && bCol + j < N) {
                    b_half[idx] = __float2half(B[bOffset + i * N + j]);
                } else {
                    b_half[idx] = __float2half(0.0f);
                }
            }
        }
        
        // Load the matrices into fragments
        wmma::load_matrix_sync(a_frag, a_half, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_half, WMMA_N);
        
        // Perform matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result back to global memory
    if (aRow < M && bCol < N) {
        float c_result[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(c_result, c_frag, WMMA_N, wmma::mem_row_major);
        
        // Write back to C
        for (int i = 0; i < WMMA_M; i++) {
            for (int j = 0; j < WMMA_N; j++) {
                if (aRow + i < M && bCol + j < N) {
                    C[(aRow + i) * N + (bCol + j)] = c_result[i * WMMA_N + j];
                }
            }
        }
    }
}

void gemm_tensor_cores(const float *A, const float *B, float *C) {
    // Set the kernel launch configuration parameters
    // Must be multiple of warp size (32)
    dim3 threadsPerBlock(128, 4); // 4 warps per block
    dim3 blocksPerGrid((M + WM - 1) / WM / (threadsPerBlock.x / 32), 
                       (N + WN - 1) / WN / threadsPerBlock.y);
    
    // Launch the CUDA kernel
    gemm_tensor_core_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);
    
    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
}

}  // namespace gemm 