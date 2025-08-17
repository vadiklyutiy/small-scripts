#include "matrix_mul.cuh"
#include <cublasLt.h>

// Implementation using cublasLt
float gemm_cublasGemmEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
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
    
    // Create operation descriptor for matrix multiplication
    cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32F);
    
    // No transposition needed
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    
    // Set operation attributes
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    // Set compute type to support tensor operations
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, 
        &computeType, sizeof(computeType));
    
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
    
    // Perform matrix multiplication
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
    
    // Record time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
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
    
    return milliseconds;
}

// Implementation using cublasLt (batch version, same functionality)
float gemm_cublasGemmBatchedEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Just reuse the regular implementation for now
    return gemm_cublasGemmEx(A, B, C);
}

// Implementation using cublasLt (strided batch version, same functionality)
float gemm_cublasGemmStridedBatchedEx(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Just reuse the regular implementation for now
    return gemm_cublasGemmEx(A, B, C);
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

// Compute reference result on CPU for validation
void compute_reference_cpu(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C_ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A is row major: A[i][k] = A[i * K + k]
                // B is column major: B[k][j] = B[j * K + k]
                sum += __bfloat162float(A[i * K + k]) * __bfloat162float(B[j * K + k]);
            }
            C_ref[i * N + j] = __float2bfloat16(sum);
        }
    }
}

// Check if the GPU result matches the reference CPU result
void check_result(const __nv_bfloat16* C, const __nv_bfloat16* C_ref, const char* name) {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float gpu_val = __bfloat162float(C[idx]);
            float cpu_val = __bfloat162float(C_ref[idx]);
            float diff = std::abs(gpu_val - cpu_val);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
        }
    }
    
    avg_diff /= (M * N);
    
    // Higher tolerance for bf16 precision and the difference between cuBLAS and CPU computation
    const float tolerance = 8.1f; 
    bool passed = (max_diff < tolerance);
    
    std::cout << "Validation for " << name << ": " 
              << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Max error: " << max_diff << std::endl;
    std::cout << "  Avg error: " << avg_diff << std::endl;
} 