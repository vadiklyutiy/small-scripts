#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include "basic-gemm/include/gemm.h"
#include "tensor-core-gemm/include/gemm_tensor_core.h"

using namespace gemm;

// Run cublas implementation using cublasLt
void run_cublaslt_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    
    // Create matrix descriptors
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    
    // Row-major A: MxK
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, M, K, K);
    
    // Column-major B: KxN
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, K, N, K);
    
    // Row-major C: MxN
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, N);
    
    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    // Set alpha and beta
    float alpha = 1.0f, beta = 0.0f;
    
    // Execute the matrix multiplication
    cublasLtMatmul(handle,
                   operationDesc,
                   &alpha,
                   A,
                   Adesc,
                   B,
                   Bdesc,
                   &beta,
                   C,
                   Cdesc,
                   C,
                   Cdesc,
                   nullptr,
                   nullptr,
                   0,
                   0);
    
    // Clean up
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(handle);
}

// Benchmark function that measures performance
float benchmark_gemm(void (*gemm_func)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*),
                     const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                     int iterations, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_func(A, B, C);
    }
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_func(A, B, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;
    
    double ops = 2.0 * M * N * K; // MUL + ADD for each element
    double gflops = (ops / (avg_time * 1e-3)) / 1e9;
    
    printf("%-20s: %.2f ms (%.2f GFLOPS)\n", name, avg_time, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_time;
}

void verify_results(const __nv_bfloat16* result, const float* reference, int size, const char* name) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float res_float = float(result[i]);
        float diff = fabsf(res_float - reference[i]);
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) {
                printf("%s: Error at index %d: %f vs %f (diff: %f)\n", 
                       name, i, res_float, reference[i], diff);
            }
        }
    }
    
    if (errors == 0) {
        printf("%s: All results match within tolerance\n", name);
    } else {
        printf("%s: Found %d errors out of %d values\n", name, errors, size);
    }
}

int main() {
    printf("CUDA GEMM Benchmark\n");
    printf("Matrix dimensions: A(%d,%d) x B(%d,%d) = C(%d,%d)\n", M, K, K, N, M, N);
    printf("Data type: bfloat16\n\n");
    
    // Allocate host memory
    float *h_A_float, *h_B_float, *h_C_ref;
    __nv_bfloat16 *h_A, *h_B, *h_C_basic, *h_C_tensor, *h_C_cublaslt;
    
    cudaMallocHost((void**)&h_A_float, M * K * sizeof(float));
    cudaMallocHost((void**)&h_B_float, K * N * sizeof(float));
    cudaMallocHost((void**)&h_C_ref, M * N * sizeof(float));
    
    cudaMallocHost((void**)&h_A, M * K * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_B, K * N * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C_basic, M * N * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C_tensor, M * N * sizeof(__nv_bfloat16));
    cudaMallocHost((void**)&h_C_cublaslt, M * N * sizeof(__nv_bfloat16));
    
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
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&d_C, M * N * sizeof(__nv_bfloat16));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Compute reference result on CPU
    printf("Computing reference result on CPU...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A_float[i * K + k] * h_B_float[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    
    printf("\nRunning benchmarks (iterations = 10):\n");
    printf("%-20s: %-10s %-10s\n", "Implementation", "Time (ms)", "GFLOPS");
    printf("--------------------------------------------\n");
    
    // Benchmark implementations
    int iterations = 10;
    
    // Benchmark basic GEMM
    benchmark_gemm(gemm_basic, d_A, d_B, d_C, iterations, "Basic GEMM");
    cudaMemcpy(h_C_basic, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    verify_results(h_C_basic, h_C_ref, M * N, "Basic GEMM");
    
    // Benchmark tensor core GEMM
    benchmark_gemm(gemm_tensor_cores, d_A, d_B, d_C, iterations, "Tensor Core GEMM");
    cudaMemcpy(h_C_tensor, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    verify_results(h_C_tensor, h_C_ref, M * N, "Tensor Core GEMM");
    
    // Benchmark cublasLt GEMM
    benchmark_gemm(run_cublaslt_gemm, d_A, d_B, d_C, iterations, "cuBLASLt GEMM");
    cudaMemcpy(h_C_cublaslt, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    verify_results(h_C_cublaslt, h_C_ref, M * N, "cuBLASLt GEMM");
    
    // Clean up
    cudaFreeHost(h_A_float);
    cudaFreeHost(h_B_float);
    cudaFreeHost(h_C_ref);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_basic);
    cudaFreeHost(h_C_tensor);
    cudaFreeHost(h_C_cublaslt);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
} 