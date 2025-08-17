#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>

// Constants for matrix dimensions - using small dims for validation
constexpr int M = 4;        // Rows of A
constexpr int K = 16;       // Cols of A, Rows of B
constexpr int N = 32;       // Cols of B

// Result: C = A * B
// A is M x K (row major)
// B is K x N (column major)
// C is M x N (row major)

// Implementation using cublasLt for bf16
void compute_cublasLt(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
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
    
    // Cleanup
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}

// Initialize matrices with deterministic values
void init_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C) {
    // Initialize A (row major) with some pattern
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float val = static_cast<float>(i * 0.01 + j * 0.1);
            A[i * K + j] = __float2bfloat16(val);
        }
    }
    
    // Initialize B (column major) with some pattern
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            float val = static_cast<float>(i * 0.01 + j * 0.1);
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
bool validate_result(const __nv_bfloat16* C_cublas, const __nv_bfloat16* C_ref) {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_errors = 0;
    
    std::cout << "Validation results:" << std::endl;
    std::cout << "idx\tGPU\t\tCPU\t\tDiff" << std::endl;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float gpu_val = __bfloat162float(C_cublas[idx]);
            float cpu_val = __bfloat162float(C_ref[idx]);
            float diff = std::abs(gpu_val - cpu_val);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (diff > 1e-2) {
                num_errors++;
                // Print some sample differences
                if (num_errors <= 10) {
                    std::cout << idx << "\t" << gpu_val << "\t" << cpu_val << "\t" << diff << std::endl;
                }
            }
        }
    }
    
    avg_diff /= (M * N);
    
    // Higher tolerance for bf16 precision
    const float tolerance = 1e-1; 
    bool passed = (max_diff < tolerance);
    
    std::cout << "Validation: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max error: " << max_diff << std::endl;
    std::cout << "Avg error: " << avg_diff << std::endl;
    std::cout << "Num errors (diff > 1e-2): " << num_errors << " out of " << (M*N) << std::endl;
    
    return passed;
}

// Update main function - force print matrices
int main() {
    // Allocate host memory
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;
    
    std::vector<__nv_bfloat16> h_A(sizeA);
    std::vector<__nv_bfloat16> h_B(sizeB);
    std::vector<__nv_bfloat16> h_C(sizeC);
    std::vector<__nv_bfloat16> h_C_ref(sizeC);
    
    // Initialize matrices
    init_matrices(h_A.data(), h_B.data(), h_C.data());
    
    // Compute reference result on CPU
    compute_reference_cpu(h_A.data(), h_B.data(), h_C_ref.data());
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, sizeB * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, sizeC * sizeof(__nv_bfloat16));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), sizeC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Compute using cublasLt
    compute_cublasLt(d_A, d_B, d_C);
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Print sample matrices
    std::cout << "\nMatrix A (row-major) - sample 4x4:" << std::endl;
    for (int i = 0; i < std::min(4, M); i++) {
        for (int j = 0; j < std::min(4, K); j++) {
            std::cout << __bfloat162float(h_A[i * K + j]) << "\t";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMatrix B (column-major) - sample 4x4:" << std::endl;
    for (int i = 0; i < std::min(4, K); i++) {
        for (int j = 0; j < std::min(4, N); j++) {
            std::cout << __bfloat162float(h_B[j * K + i]) << "\t";
        }
        std::cout << std::endl;
    }
    
    // Detailed validation for a portion of the matrices
    std::cout << "\nDetailed validation of first 4x4 submatrix:" << std::endl;
    for (int i = 0; i < std::min(4, M); i++) {
        for (int j = 0; j < std::min(4, N); j++) {
            int idx = i * N + j;
            float gpu_val = __bfloat162float(h_C[idx]);
            float cpu_val = __bfloat162float(h_C_ref[idx]);
            float diff = std::abs(gpu_val - cpu_val);
            std::cout << "C[" << i << "][" << j << "]: GPU=" << gpu_val 
                      << ", CPU=" << cpu_val << ", diff=" << diff << std::endl;
        }
    }
    
    // Overall validation
    bool valid = validate_result(h_C.data(), h_C_ref.data());
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return valid ? 0 : 1;
} 