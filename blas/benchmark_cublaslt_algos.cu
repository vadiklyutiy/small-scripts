#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <string>

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

// Maximum number of algorithms to test
constexpr int MAX_HEURISTIC_RESULTS = 64;
constexpr int NUM_WORKSPACE_SIZES = 5;

struct AlgoPerformance {
    int algoId;
    std::string algoName;
    bool success;
    float minTime;
    float maxTime;
    float avgTime;
    float flops;
    size_t workspaceSize;
};

// Implementation using cublasLt with a specific algorithm
float benchmark_cublasLt_algo(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, 
                            const cublasLtMatmulAlgo_t* algo, size_t workspaceSize) {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    // Alpha=1.0 and beta=0.0 are special values that cuBLAS/cuBLASLt recognize
    // and automatically optimize for.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Timing events
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
    
    // Create operation descriptor for matrix multiplication
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
    
    // Allocate workspace for cublasLt
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&workspace, workspaceSize);
    }
    
    // Start timing
    cudaEventRecord(start);
    
    // Perform matrix multiplication with the specified algorithm
    cublasStatus_t status = cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        algo,
        workspace,
        workspaceSize,
        0);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Convert to microseconds
    float microseconds = milliseconds * 1000.0f;
    
    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
    
    // Return error value if operation failed
    if (status != CUBLAS_STATUS_SUCCESS) {
        return -1.0f;
    }
    
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
    std::cout << std::endl;
    
    // Create results vector for all algorithms
    std::vector<AlgoPerformance> algoResults;
    
    // Setup for algorithm discovery
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    // Create operation descriptor and matrix layouts for algo discovery
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    // Create operation descriptor
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
    
    // Create matrix descriptors for algo discovery
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, M, K, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, N);
    
    // Set matrix layout order
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;
    
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder));
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    
    // Different workspace sizes to try
    size_t workspaceSizes[NUM_WORKSPACE_SIZES] = {
        0,                    // No workspace
        1 * 1024 * 1024,      // 1 MB
        4 * 1024 * 1024,      // 4 MB
        16 * 1024 * 1024,     // 16 MB
        32 * 1024 * 1024      // 32 MB
    };
    
    int totalAlgos = 0;
    
    // Try different workspace sizes
    for (int i = 0; i < NUM_WORKSPACE_SIZES; i++) {
        size_t workspaceSize = workspaceSizes[i];
        
        std::cout << "\nTrying workspace size: " << (workspaceSize / (1024 * 1024)) 
                  << " MB" << std::endl;
        
        // Create preference for algorithm selection
        cublasLtMatmulPreferenceCreate(&preference);
        
        // Set workspace size
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
            &workspaceSize, sizeof(workspaceSize));
        
        // Get algorithms using heuristics
        cublasLtMatmulHeuristicResult_t heuristicResults[MAX_HEURISTIC_RESULTS];
        int returnedResults = 0;
        
        cublasLtMatmulAlgoGetHeuristic(
            ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference,
            MAX_HEURISTIC_RESULTS, heuristicResults, &returnedResults);
        
        std::cout << "Found " << returnedResults << " algorithms for workspace size "
                  << (workspaceSize / (1024 * 1024)) << " MB" << std::endl << std::endl;
        
        totalAlgos += returnedResults;
        
        // Test each algorithm
        for (int j = 0; j < returnedResults; j++) {
            const cublasLtMatmulAlgo_t* algo = &heuristicResults[j].algo;
            
            // Get algorithm information
            int algoId = j; // Using a sequential ID since we can't easily get algo ID
            
            // Get algorithm attributes
            unsigned int algoIdAttr;
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_ID, &algoIdAttr, sizeof(algoIdAttr), nullptr);
            
            int splitK_val = 0;
            int swizzle_val = 0;
            int customOption_val = 0;
            int tile_val = 0;
            int stages_val = 0;
            int reduceMask_val = 0;
            
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle_val, sizeof(swizzle_val), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption_val, sizeof(customOption_val), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_val, sizeof(tile_val), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_val, sizeof(stages_val), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(algo, 
                CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduceMask_val, sizeof(reduceMask_val), nullptr);
            
            std::string algoName = "Algo ID: " + std::to_string(algoIdAttr) + 
                                 " SplitK: " + std::to_string(splitK_val) +
                                 " Swizzle: " + std::to_string(swizzle_val) +
                                 " Custom: " + std::to_string(customOption_val) +
                                 " Tile: " + std::to_string(tile_val) +
                                 " Stages: " + std::to_string(stages_val) +
                                 " Reduce: " + std::to_string(reduceMask_val) +
                                 " WS: " + std::to_string(workspaceSize / (1024 * 1024)) + "MB";
            
            std::cout << "Testing algorithm " << j+1 << "/" << returnedResults 
                      << ": " << algoName << std::endl;
            
            // Print detailed algorithm information
            std::cout << "  Algorithm ID: " << algoIdAttr << std::endl;
            std::cout << "  SplitK: " << splitK_val << std::endl;
            std::cout << "  Swizzle: " << swizzle_val << std::endl;
            std::cout << "  Custom Option: " << customOption_val << std::endl;
            std::cout << "  Tile ID: " << tile_val << std::endl;
            std::cout << "  Stages: " << stages_val << std::endl;
            
            AlgoPerformance result;
            result.algoId = algoIdAttr; // Use the actual algo ID
            result.algoName = algoName;
            result.workspaceSize = workspaceSize;
            
            // Clear C before each algorithm test
            cudaMemset(d_C, 0, sizeC * sizeof(__nv_bfloat16));
            
            // Warmup iterations
            bool warmupSuccess = true;
            for (int k = 0; k < NUM_WARMUP; k++) {
                float time = benchmark_cublasLt_algo(d_A, d_B, d_C, algo, workspaceSize);
                if (time < 0) {
                    warmupSuccess = false;
                    break;
                }
            }
            
            if (!warmupSuccess) {
                std::cout << "  Algorithm not applicable for this problem configuration" << std::endl;
                result.success = false;
                algoResults.push_back(result);
                continue;
            }
            
            // Benchmarking
            std::vector<float> timings;
            bool benchSuccess = true;
            
            for (int k = 0; k < NUM_ITERATIONS; k++) {
                float us = benchmark_cublasLt_algo(d_A, d_B, d_C, algo, workspaceSize);
                if (us < 0) {
                    benchSuccess = false;
                    break;
                }
                timings.push_back(us);
                std::cout << "  Iteration " << k + 1 << ": " << us << " us (" 
                          << calculate_gflops(us) << " GFLOPS)" << std::endl;
            }
            
            if (!benchSuccess) {
                std::cout << "  Benchmark failed for this algorithm" << std::endl;
                result.success = false;
                algoResults.push_back(result);
                continue;
            }
            
            // Calculate statistics
            float min_time = *std::min_element(timings.begin(), timings.end());
            float max_time = *std::max_element(timings.begin(), timings.end());
            float avg_time = std::accumulate(timings.begin(), timings.end(), 0.0f) / timings.size();
            
            // Report results for this algorithm
            std::cout << "  Results:" << std::endl;
            std::cout << "  Min time: " << min_time << " us (" << calculate_gflops(min_time) << " GFLOPS)" << std::endl;
            std::cout << "  Max time: " << max_time << " us (" << calculate_gflops(max_time) << " GFLOPS)" << std::endl;
            std::cout << "  Avg time: " << avg_time << " us (" << calculate_gflops(avg_time) << " GFLOPS)" << std::endl;
            std::cout << std::endl;
            
            // Store results
            result.success = true;
            result.minTime = min_time;
            result.maxTime = max_time;
            result.avgTime = avg_time;
            result.flops = calculate_gflops(min_time);
            algoResults.push_back(result);
        }
        
        // Clean up preference
        cublasLtMatmulPreferenceDestroy(preference);
    }
    
    // Clean up discovery resources
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
    
    // Report sorted results
    std::cout << "\n====================================================" << std::endl;
    std::cout << "ALGORITHMS RANKED BY PERFORMANCE (BEST FIRST)" << std::endl;
    std::cout << "Found " << totalAlgos << " algorithms total" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Sort algorithms by performance (best first)
    std::sort(algoResults.begin(), algoResults.end(), 
              [](const AlgoPerformance& a, const AlgoPerformance& b) {
                  // Only compare successful algorithms
                  if (a.success && !b.success) return true;
                  if (!a.success && b.success) return false;
                  if (!a.success && !b.success) return false;
                  // Sort by min time (lower is better)
                  return a.minTime < b.minTime;
              });
    
    // Print table header
    std::cout << std::left << std::setw(6) << "Rank" 
              << std::setw(100) << "Algorithm Parameters"
              << std::setw(12) << "Min (μs)" 
              << std::setw(12) << "Avg (μs)" 
              << std::setw(12) << "Max (μs)" 
              << std::setw(12) << "GFLOPS" << std::endl;
    std::cout << std::string(167, '-') << std::endl;
    
    // Print sorted results
    int rank = 1;
    for (const auto& result : algoResults) {
        if (result.success) {
            std::cout << std::left << std::setw(6) << rank++ 
                      << std::setw(100) << result.algoName
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.minTime
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.avgTime
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.maxTime
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.flops << std::endl;
        }
    }
    
    // Print unsuccessful algorithms
    bool hasUnsuccessful = false;
    for (const auto& result : algoResults) {
        if (!result.success) {
            if (!hasUnsuccessful) {
                std::cout << "\nUnsuccessful Algorithms:" << std::endl;
                std::cout << std::string(95, '-') << std::endl;
                hasUnsuccessful = true;
            }
            std::cout << result.algoName << " (Workspace: " << (result.workspaceSize / (1024 * 1024)) << " MB)" << std::endl;
        }
    }
    
    // Check what algorithm the heuristic selects by default with our original workspace size
    std::cout << "\n====================================================" << std::endl;
    std::cout << "DEFAULT ALGORITHM SELECTED BY HEURISTIC (4MB WORKSPACE)" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Create a new preference for 4MB workspace (our original setting)
    cublasLtMatmulPreference_t defaultPreference = nullptr;
    cublasLtMatmulPreferenceCreate(&defaultPreference);
    size_t defaultWorkspaceSize = 4 * 1024 * 1024; // 4MB
    cublasLtMatmulPreferenceSetAttribute(
        defaultPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &defaultWorkspaceSize, sizeof(defaultWorkspaceSize));
    
    // Get heuristic result
    cublasLtMatmulHeuristicResult_t defaultHeuristicResult;
    int returnedResults = 0;
    
    // Create matrices and operation descriptor again
    cublasLtMatmulDesc_t defaultOpDesc = nullptr;
    cublasLtMatrixLayout_t defaultAdesc = nullptr, defaultBdesc = nullptr, defaultCdesc = nullptr;
    
    cublasLtMatmulDescCreate(&defaultOpDesc, computeType, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(
        defaultOpDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(
        defaultOpDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    cublasLtMatrixLayoutCreate(&defaultAdesc, CUDA_R_16BF, M, K, K);
    cublasLtMatrixLayoutCreate(&defaultBdesc, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&defaultCdesc, CUDA_R_16BF, M, N, N);
    
    cublasLtMatrixLayoutSetAttribute(
        defaultAdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(
        defaultBdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder));
    cublasLtMatrixLayoutSetAttribute(
        defaultCdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle, defaultOpDesc, defaultAdesc, defaultBdesc, defaultCdesc, defaultCdesc,
        defaultPreference, 1, &defaultHeuristicResult, &returnedResults);
    
    if (returnedResults > 0) {
        // Get and print algorithm information
        unsigned int algoId;
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        
        std::cout << "Default algorithm selected by heuristic:" << std::endl;
        std::cout << "Algorithm ID: " << algoId << std::endl;
        
        // Try to get more information about the algorithm
        int splitK_val = 0;
        int swizzle_val = 0;
        int customOption_val = 0;
        int tile_val = 0;
        int stages_val = 0;
        int reduceMask_val = 0;
        
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle_val, sizeof(swizzle_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption_val, sizeof(customOption_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_val, sizeof(tile_val), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduceMask_val, sizeof(reduceMask_val), nullptr);
        
        // Hopper GPUs support stages parameter
        cublasLtMatmulAlgoConfigGetAttribute(&defaultHeuristicResult.algo, 
            CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_val, sizeof(stages_val), nullptr);
        
        std::cout << "  SplitK: " << splitK_val << std::endl;
        std::cout << "  Swizzle: " << swizzle_val << std::endl;
        std::cout << "  Custom Option: " << customOption_val << std::endl;
        std::cout << "  Tile ID: " << tile_val << std::endl;
        std::cout << "  Stages: " << stages_val << std::endl;
        std::cout << "  Reduction Scheme: " << reduceMask_val << std::endl;
        std::cout << "  Workspace Size: " << defaultWorkspaceSize / (1024 * 1024) << " MB" << std::endl;
        
        // Benchmark the default algorithm
        cudaMemset(d_C, 0, sizeC * sizeof(__nv_bfloat16));
        
        std::vector<float> defaultTimings;
        
        // Warmup
        for (int i = 0; i < NUM_WARMUP; i++) {
            benchmark_cublasLt_algo(d_A, d_B, d_C, &defaultHeuristicResult.algo, defaultWorkspaceSize);
        }
        
        // Benchmarking
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            float us = benchmark_cublasLt_algo(d_A, d_B, d_C, &defaultHeuristicResult.algo, defaultWorkspaceSize);
            defaultTimings.push_back(us);
            std::cout << "Iteration " << i + 1 << ": " << us << " us (" 
                      << calculate_gflops(us) << " GFLOPS)" << std::endl;
        }
        
        // Calculate statistics
        float min_time = *std::min_element(defaultTimings.begin(), defaultTimings.end());
        float max_time = *std::max_element(defaultTimings.begin(), defaultTimings.end());
        float avg_time = std::accumulate(defaultTimings.begin(), defaultTimings.end(), 0.0f) / defaultTimings.size();
        
        // Print results
        std::cout << "\nDefault Heuristic Algorithm Results:" << std::endl;
        std::cout << "Min time: " << min_time << " us (" << calculate_gflops(min_time) << " GFLOPS)" << std::endl;
        std::cout << "Max time: " << max_time << " us (" << calculate_gflops(max_time) << " GFLOPS)" << std::endl;
        std::cout << "Avg time: " << avg_time << " us (" << calculate_gflops(avg_time) << " GFLOPS)" << std::endl;
    } else {
        std::cout << "No algorithm found by default heuristic!" << std::endl;
    }
    
    // Clean up default preference resources
    cublasLtMatmulPreferenceDestroy(defaultPreference);
    cublasLtMatrixLayoutDestroy(defaultCdesc);
    cublasLtMatrixLayoutDestroy(defaultBdesc);
    cublasLtMatrixLayoutDestroy(defaultAdesc);
    cublasLtMatmulDescDestroy(defaultOpDesc);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
} 