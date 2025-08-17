#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include "../include/gemm.h"

using namespace tma_gemm;

int main() {
    printf("Testing TMA GEMM implementation\n");
    
    // Allocate host memory for A, B, C
    __nv_bfloat16 *h_A, *h_B, *h_C, *h_C_ref;
    cudaMallocHost((void**)&h_A, M_DIM * K_DIM * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_B, K_DIM * N_DIM * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C, M_DIM * N_DIM * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C_ref, M_DIM * N_DIM * sizeof(__nv_bfloat16));
    
    // Allocate device memory for A, B, C
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M_DIM * K_DIM * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_B, K_DIM * N_DIM * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_C, M_DIM * N_DIM * sizeof(__nv_bfloat16));
    
    // Initialize matrices
    initialize_matrices(h_A, h_B, h_C);
    
    // Compute reference result on CPU
    printf("Computing reference result on CPU...\n");
    cpu_reference(h_A, h_B, h_C_ref);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M_DIM * K_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_DIM * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Run TMA GEMM kernel
    printf("Running TMA GEMM kernel...\n");
    cudaMemset(d_C, 0, M_DIM * N_DIM * sizeof(__nv_bfloat16));
    
    // Use the TMA implementation
    gemm_tma(d_A, d_B, d_C);
    
    // Print debug info if available
#ifdef DEBUG_TMA
    print_debug_info();
#endif
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M_DIM * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Verify result
    printf("Verifying result...\n");
    bool correct = verify_result(h_C, h_C_ref);
    
    if (correct) {
        printf("TMA GEMM test PASSED\n");
    } else {
        printf("TMA GEMM test FAILED\n");
    }
    
    // Free memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return correct ? 0 : 1;
} 