#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

// Constants for matrix dimensions
constexpr int M = 8;        // Rows of A
constexpr int K = 4096;     // Cols of A, Rows of B
constexpr int N = 28672;    // Cols of B

// Number of iterations for benchmarking
constexpr int NUM_ITERATIONS = 10;
constexpr int NUM_WARMUP = 3;

// Result: C = A * B
// A is M x K (row major)
// B is K x N (column major)
// C is M x N (row major)

// Implementation using cublasLt for bf16
float benchmark_cublasLt(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    // Alpha=1.0 and beta=0.0 are special values that cuBLAS/cuBLASLt recognize
    // and automatically optimize for. No additional API calls are needed to
    // enable these optimizations.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Timing events - create early but record just before/after matmul
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Matrix dimensions
    int m = M; // Number of rows in A and C
    int n = N; // Number of columns in B and C
    int k = K; // Number of columns in A and rows in B
    
    // Leading dimensions
    int lda = K; // For row-major A, lda = K
    int ldb = K; // For column-major B, ldb = K
    int ldc = N; // For row-major C, ldc = N
    
    // Create operation descriptor and matrix layouts
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    // Create operation descriptor for matrix multiplication with correct signature
    // cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
    //                                        cublasComputeType_t computeType,
    //                                        cudaDataType_t scaleType);
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cublasLtMatmulDescCreate(&operationDesc, computeType, CUDA_R_32F);
    
    // No transposition needed
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    
    // Set operation attributes
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    // Create matrix descriptors
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc);
    
    // Set matrix layout order - A and C are row-major, B is column-major
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;
    
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder));
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    
    // Create preference for algorithm selection
    cublasLtMatmulPreferenceCreate(&preference);
    
    // Allocate workspace for cublasLt
    size_t workspaceSize = 4 * 1024 * 1024; // 4MB workspace
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspaceSize);
    
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspaceSize, sizeof(workspaceSize));
    
    // Get algorithm for matrix multiplication
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference,
        1, &heuristicResult, &returnedResults);
    
    // Print information about the selected algorithm
    if (returnedResults > 0) {
        std::cout << "\nAlgorithm selected by heuristic:" << std::endl;
        
        // Get and print algorithm information
        unsigned int algoId;
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        
        std::cout << "Algorithm ID: " << algoId << std::endl;
        
        // Get additional algorithm attributes
        int splitK_val = 0;
        int swizzle_val = 0;
        int customOption_val = 0;
        int tile_val = 0;
        int stages_val = 0;
        int reduceMask_val = 0;
        
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle_val, sizeof(swizzle_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption_val, sizeof(customOption_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_val, sizeof(tile_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduceMask_val, sizeof(reduceMask_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_val, sizeof(stages_val), nullptr);
        
        std::cout << "  SplitK: " << splitK_val << std::endl;
        std::cout << "  Swizzle: " << swizzle_val << std::endl;
        std::cout << "  Custom Option: " << customOption_val << std::endl;
        std::cout << "  Tile ID: " << tile_val << std::endl;
        std::cout << "  Stages: " << stages_val << std::endl;
        std::cout << "  Reduction Scheme: " << reduceMask_val << std::endl;
        std::cout << "  Workspace Size: " << workspaceSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << std::endl;
    } else {
        std::cout << "\nNo algorithm found by heuristic!" << std::endl;
    }
    
    // Start timing just before the actual matrix multiplication
    cudaEventRecord(start);
    
    // Perform matrix multiplication
    // When alpha=1 and beta=0, cuBLAS will automatically optimize:
    // - With beta=0, no need to read C before writing the result
    // - With alpha=1, no need to scale the matrix product before writing
    cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        returnedResults > 0 ? &heuristicResult.algo : nullptr,
        workspace,
        workspaceSize,
        0);
    
    // Stop timing immediately after
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Convert to microseconds
    float microseconds = milliseconds * 1000.0f;
    
    // Cleanup
    cudaFree(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
    
    return microseconds;
}

// Initialize matrices with random values
void init_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C) {
    // Initialize A (row major)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            A[i * K + j] = __float2bfloat16(val);
        }
    }
    
    // Initialize B (column major)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            B[j * K + i] = __float2bfloat16(val);
        }
    }
    
    // Initialize C with zeros
    for (int i = 0; i < M * N; i++) {
        C[i] = __float2bfloat16(0.0f);
    }
}

// Calculate GFLOPS (2*M*N*K operations for matrix multiplication)
double calculate_gflops(float microseconds) {
    double operations = 2.0 * M * N * K; // Multiply-add operations
    double gflops = (operations / microseconds) * 1e-3;  // Convert to GFLOPS (microseconds * 1e-3 = milliseconds * 1e-6)
    return gflops;
}

// Function to clear GPU cache by reading a large array
void clear_gpu_cache() {
    const size_t cache_size = 100 * 1024 * 1024; // 100MB
    float* d_cache;
    float* h_cache = new float[cache_size / sizeof(float)];
    
    // Initialize host array with random values
    for (size_t i = 0; i < cache_size / sizeof(float); i++) {
        h_cache[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate and copy to device
    cudaMalloc(&d_cache, cache_size);
    cudaMemcpy(d_cache, h_cache, cache_size, cudaMemcpyHostToDevice);
    
    // Read the array to clear cache
    float sum = 0.0f;
    for (size_t i = 0; i < cache_size / sizeof(float); i++) {
        float val;
        cudaMemcpy(&val, &d_cache[i], sizeof(float), cudaMemcpyDeviceToHost);
        sum += val;
    }
    
    // Cleanup
    cudaFree(d_cache);
    delete[] h_cache;
    
    // Print cache clearing status
    std::cout << "GPU cache cleared by reading " << (cache_size / (1024 * 1024)) 
              << "MB array (sum: " << sum << ")" << std::endl;
}

int main() {
    // Allocate host memory
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;
    
    std::vector<__nv_bfloat16> h_A(sizeA);
    std::vector<__nv_bfloat16> h_B(sizeB);
    std::vector<__nv_bfloat16> h_C(sizeC);
    
    // Initialize matrices
    init_matrices(h_A.data(), h_B.data(), h_C.data());
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, sizeB * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, sizeC * sizeof(__nv_bfloat16));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Print matrix dimensions
    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "A: " << M << " x " << K << " (row-major, bf16)" << std::endl;
    std::cout << "B: " << K << " x " << N << " (column-major, bf16)" << std::endl;
    std::cout << "C: " << M << " x " << N << " (row-major, bf16)" << std::endl;
    
    // Clear GPU cache before warmup
    clear_gpu_cache();
    
    // Warmup
    std::cout << "\nWarmup iterations..." << std::endl;
    for (int i = 0; i < NUM_WARMUP; i++) {
        float us = benchmark_cublasLt(d_A, d_B, d_C);
        std::cout << "Warmup " << i + 1 << ": " << us << " us" << std::endl;
    }
    
    // Benchmarking
    std::cout << "\nBenchmarking cublasLt BF16 implementation..." << std::endl;
    std::vector<float> timings;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float us = benchmark_cublasLt(d_A, d_B, d_C);
        timings.push_back(us);
        std::cout << "Iteration " << i + 1 << ": " << us << " us (" 
                  << calculate_gflops(us) << " GFLOPS)" << std::endl;
    }
    
    // Calculate statistics
    float min_time = *std::min_element(timings.begin(), timings.end());
    float max_time = *std::max_element(timings.begin(), timings.end());
    float avg_time = std::accumulate(timings.begin(), timings.end(), 0.0f) / timings.size();
    
    // Report results
    std::cout << "\nResults:" << std::endl;
    std::cout << "Min time: " << min_time << " us (" << calculate_gflops(min_time) << " GFLOPS)" << std::endl;
    std::cout << "Max time: " << max_time << " us (" << calculate_gflops(max_time) << " GFLOPS)" << std::endl;
    std::cout << "Avg time: " << avg_time << " us (" << calculate_gflops(avg_time) << " GFLOPS)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
} 