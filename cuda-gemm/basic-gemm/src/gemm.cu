#include "gemm.h"
#include <stdio.h>

namespace gemm {

__global__ void gemm_basic_kernel(const float *A, const float *B, float *C) {
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    const int aBegin = K * BM * by;
    
    // Index of the last sub-matrix of A processed by the block
    const int aEnd = aBegin + K - 1;
    
    // Step size used to iterate through the sub-matrices of A
    const int aStep = BM;
    
    // Index of the first sub-matrix of B processed by the block
    const int bBegin = BN * bx;
    
    // Step size used to iterate through the sub-matrices of B
    const int bStep = BN * K;
    
    // Allocate shared memory for the sub-matrices of A and B
    __shared__ float As[BM][BM];
    __shared__ float Bs[BM][BN];
    
    // C block that this thread computes
    const int cRow = by * BM + ty;
    const int cCol = bx * BN + tx;
    
    // Accumulate results
    float cValue = 0.0f;
    
    // Loop over all the sub-matrices of A and B to compute the block of C
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        // Each thread loads one element of each matrix
        if (ty < BM && tx < BM && a + tx < K * M) {
            As[ty][tx] = A[a + K * ty + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (ty < BM && tx < BN && b + K * ty + tx < K * N) {
            Bs[ty][tx] = B[b + K * ty + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < BM; ++k) {
            cValue += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize to make sure that the preceding computation is done
        __syncthreads();
    }
    
    // Write the block sub-matrix to global memory
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = cValue;
    }
}

void gemm_basic(const float *A, const float *B, float *C) {
    // Set the kernel launch configuration parameters
    dim3 threadsPerBlock(BN, BM);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Launch the CUDA kernel
    gemm_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);
    
    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
}

}  // namespace gemm 