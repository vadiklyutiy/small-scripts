#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/gemm_tensor_core.h"

using namespace gemm;

int main() {
    printf("Testing Tensor Core GEMM implementation\n");
    
    // Allocate host memory for A, B, C
    float *h_A, *h_B, *h_C, *h_C_ref;
    cudaMallocHost((void**)&h_A, M * K * sizeof(float));
    cudaMallocHost((void**)&h_B, K * N * sizeof(float));
    cudaMallocHost((void**)&h_C, M * N * sizeof(float));
    cudaMallocHost((void**)&h_C_ref, M * N * sizeof(float));
    
    // Initialize matrices with random values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A[i * K + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B[i * N + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // Allocate device memory for A, B, C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Running tensor core GEMM kernel...\n");
    
    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Run tensor core GEMM
    gemm_tensor_cores(d_A, d_B, d_C);
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tensor Core GEMM execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute reference result
    printf("Computing reference result...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    
    // Verify results
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(h_C[i * N + j] - h_C_ref[i * N + j]);
            // Use a higher tolerance for tensor core operations
            if (diff > 0.2f) {
                errors++;
                if (errors < 10) {
                    printf("Error at (%d, %d): %f vs %f (diff: %f)\n", 
                           i, j, h_C[i * N + j], h_C_ref[i * N + j], diff);
                }
            }
        }
    }
    
    if (errors == 0) {
        printf("All results match within tolerance!\n");
    } else {
        printf("Found %d errors\n", errors);
    }
    
    // Performance metrics
    double ops = 2.0 * M * N * K; // MUL + ADD for each element
    double gflops = (ops / (milliseconds * 1e-3)) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Clean up
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
} 