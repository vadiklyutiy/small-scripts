#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include "../include/gemm.h"

namespace tma_gemm {

void initialize_matrices(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize A (row major)
    for (int i = 0; i < M_DIM; ++i) {
        for (int j = 0; j < K_DIM; ++j) {
            A[i * K_DIM + j] = __nv_bfloat16(dist(gen));
        }
    }

    // Initialize B (column major)
    for (int j = 0; j < N_DIM; ++j) {
        for (int i = 0; i < K_DIM; ++i) {
            B[j * K_DIM + i] = __nv_bfloat16(dist(gen));
        }
    }

    // Initialize C to zeros
    for (int i = 0; i < M_DIM * N_DIM; ++i) {
        C[i] = __nv_bfloat16(0.0f);
    }
}

// Helper function to check results
bool verify_result(const __nv_bfloat16* C, const __nv_bfloat16* C_ref) {
    constexpr float tolerance = 1e-1f; // Using a larger tolerance for bf16
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < M_DIM; ++i) {
        for (int j = 0; j < N_DIM; ++j) {
            float c = float(C[i * N_DIM + j]);
            float c_ref = float(C_ref[i * N_DIM + j]);
            float rel_error = std::abs(c - c_ref) / (std::abs(c_ref) + 1e-5f);
            
            if (rel_error > tolerance) {
                if (errors < 10) { // Limit error output
                    printf("Error at position (%d, %d): %f vs %f, relative error: %f\n",
                           i, j, c, c_ref, rel_error);
                }
                errors++;
                correct = false;
            }
        }
    }
    
    if (!correct) {
        printf("Found %d errors\n", errors);
    }
    
    return correct;
}

// Reference CPU implementation for validation
void cpu_reference(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Compute C = A * B with A row-major and B column-major
    for (int i = 0; i < M_DIM; ++i) {
        for (int j = 0; j < N_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K_DIM; ++k) {
                sum += float(A[i * K_DIM + k]) * float(B[j * K_DIM + k]);
            }
            C[i * N_DIM + j] = __nv_bfloat16(sum);
        }
    }
}

} // namespace tma_gemm 