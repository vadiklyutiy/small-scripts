#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include "../include/gemm.h"

using namespace gemm;

int main() {
    printf("Testing Basic GEMM implementation with bf16\n");
    
    // Allocate host memory for A, B, C 
    float *h_A_float, *h_B_float, *h_C_float, *h_C_ref;
    __nv_bfloat16 *h_A, *h_B, *h_C;
    
    cudaMallocHost((void**)&h_A_float, M * K * sizeof(float));
    cudaMallocHost((void**)&h_B_float, K * N * sizeof(float));
    cudaMallocHost((void**)&h_C_float, M * N * sizeof(float));
    cudaMallocHost((void**)&h_C_ref, M * N * sizeof(float));
    
    cudaMallocHost((void**)&h_A, M * K * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_B, K * N * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C, M * N * sizeof(__nv_bfloat16));
    
    // Initialize matrices with random values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A_float[i * K + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            h_A[i * K + j] = __nv_bfloat16(h_A_float[i * K + j]);
        }
    }
    
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B_float[i * N + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            h_B[j * K + i] = __nv_bfloat16(h_B_float[i * N + j]); // Column-major layout for B
        }
    }
    
    // Allocate device memory for A, B, C
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_C, M * N * sizeof(__nv_bfloat16));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    printf("Running basic GEMM kernel with bf16...\n");
    
    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Run basic GEMM
    gemm_basic(d_A, d_B, d_C);
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Basic GEMM execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Convert results back to float for verification
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_C_float[i * N + j] = float(h_C[i * N + j]);
        }
    }
    
    // Compute reference result
    printf("Computing reference result...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A_float[i * K + k] * h_B_float[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    
    // Verify results
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(h_C_float[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 0.1f) {
                errors++;
                if (errors < 10) {
                    printf("Error at (%d, %d): %f vs %f (diff: %f)\n", 
                           i, j, h_C_float[i * N + j], h_C_ref[i * N + j], diff);
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
    cudaFreeHost(h_A_float);
    cudaFreeHost(h_B_float);
    cudaFreeHost(h_C_float);
    cudaFreeHost(h_C_ref);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
} 