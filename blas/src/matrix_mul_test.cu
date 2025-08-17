#include "matrix_mul.cuh"

int main() {
    // Allocate host memory
    __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
    __nv_bfloat16* h_C = new __nv_bfloat16[M * N];
    __nv_bfloat16* h_C_ref = new __nv_bfloat16[M * N];
    
    // Initialize matrices
    std::cout << "Initializing matrices..." << std::endl;
    init_matrices(h_A, h_B, h_C);
    
    // Compute reference CPU result
    std::cout << "Computing reference result on CPU..." << std::endl;
    compute_reference_cpu(h_A, h_B, h_C_ref);
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Test each implementation
    std::cout << "\n----- Testing cuBLAS implementations -----\n" << std::endl;
    
    // --- Test cublasGemmEx ---
    float time_ms_gemmex = gemm_cublasGemmEx(d_A, d_B, d_C);
    float time_us_gemmex = time_ms_gemmex * 1000.0f; // Convert to microseconds
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Check result
    check_result(h_C, h_C_ref, "cublasGemmEx");
    std::cout << "  Execution time: " << time_us_gemmex << " μs" << std::endl << std::endl;
    
    // --- Test cublasGemmBatchedEx ---
    float time_ms_gemmbatchedex = gemm_cublasGemmBatchedEx(d_A, d_B, d_C);
    float time_us_gemmbatchedex = time_ms_gemmbatchedex * 1000.0f; // Convert to microseconds
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Check result
    check_result(h_C, h_C_ref, "cublasGemmBatchedEx");
    std::cout << "  Execution time: " << time_us_gemmbatchedex << " μs" << std::endl << std::endl;
    
    // --- Test cublasGemmStridedBatchedEx ---
    float time_ms_gemmstridedbatchedex = gemm_cublasGemmStridedBatchedEx(d_A, d_B, d_C);
    float time_us_gemmstridedbatchedex = time_ms_gemmstridedbatchedex * 1000.0f; // Convert to microseconds
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Check result
    check_result(h_C, h_C_ref, "cublasGemmStridedBatchedEx");
    std::cout << "  Execution time: " << time_us_gemmstridedbatchedex << " μs" << std::endl << std::endl;
    
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::cout << "All tests completed!" << std::endl;
    
    return 0;
} 