#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using bfloat16_t = __nv_bfloat16;

// Convert float to bfloat16
__device__ __host__ bfloat16_t float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Convert bfloat16 to float
__device__ __host__ float bfloat16_to_float(bfloat16_t bf) {
    return __bfloat162float(bf);
}

// SiLU kernel for tensor [2048, 22016] - Non-vectorized version
// For each j in [0, 11007]: output[j] = silu(input[j]) * input[j + 11008]
// where silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
__global__ void silu_kernel(const bfloat16_t* __restrict__ data, 
                           bfloat16_t* __restrict__ output,
                           int rows, int cols) {
    // Global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total elements to process (only first 11008 elements of each row)
    int total_elements = rows * 11008;
    
    if (i >= total_elements) return;
    
    // Calculate row and column indices
    int row = i / 11008;
    int col = i % 11008;
    
    // Calculate actual indices in the input tensor
    int input_idx = row * cols + col;           // Position of element to apply SiLU
    int mult_idx = row * cols + col + 11008;    // Position of multiplier element
    
    // Load data
    bfloat16_t x = data[input_idx];
    bfloat16_t mult = data[mult_idx];
    
    // Calculate SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    bfloat16_t one = (bfloat16_t)1.0f;
    bfloat16_t sigmoid_x = one / (one + hexp(-x));
    bfloat16_t silu_x = x * sigmoid_x;
    
    // Multiply by the corresponding element
    bfloat16_t result = silu_x * mult;
    
    // Store result
    output[i] = result;
}

// SiLU kernel for tensor [2048, 22016] - Vectorized version using bfloat162 intrinsics
__global__ void silu_kernel_vec2(const bfloat16_t* __restrict__ data, 
                                      bfloat16_t* __restrict__ output,
                                      int rows, int cols) {
    // Global thread index (each thread processes 2 elements using bfloat162)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    // Total elements to process (only first 11008 elements of each row)
    int total_elements = rows * 11008;
    
    if (i >= total_elements) return;
    
    // Since i is always even and total_elements is even, 
    // if i < total_elements then i+1 < total_elements is always true
    // Calculate row and column indices for both elements
    int row1 = i / 11008;
    int col1 = i % 11008;
    int row2 = (i + 1) / 11008;
    int col2 = (i + 1) % 11008;
    
    // Calculate actual indices in the input tensor
    int input_idx1 = row1 * cols + col1;
    int mult_idx1 = row1 * cols + col1 + 11008;
    int input_idx2 = row2 * cols + col2;
    int mult_idx2 = row2 * cols + col2 + 11008;
    
    // Load data as bfloat162 with smart vectorized loads
    __nv_bfloat162 x, mult;
    
    // Check if we can use vectorized load for input data
    if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x = *((const __nv_bfloat162*)&data[input_idx1]);
        mult = *((const __nv_bfloat162*)&data[mult_idx1]);
    } else {
        // Elements not contiguous - use separate loads
        x = __halves2bfloat162(data[input_idx1], data[input_idx2]);
        mult = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
    }
    
    // Calculate SiLU using bfloat162 intrinsics: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);  // Create bfloat162(1.0, 1.0)
    __nv_bfloat162 neg_x = __hneg2(x);               // -x for both elements
    __nv_bfloat162 exp_neg_x = h2exp(neg_x);         // exp(-x) for both elements
    __nv_bfloat162 sigmoid_x = __h2div(one, __hadd2(one, exp_neg_x));  // 1 / (1 + exp(-x))
    __nv_bfloat162 silu_x = __hmul2(x, sigmoid_x);   // x * sigmoid(x)
    
    // Multiply by the corresponding elements
    __nv_bfloat162 result = __hmul2(silu_x, mult);
    
    // Store results using vectorized stores when possible
    // Output is always contiguous, so we can use vectorized stores
    if ((reinterpret_cast<uintptr_t>(&output[i]) & 3) == 0) {
        // Aligned - use single vectorized store
        *(((__nv_bfloat162*)&output[i])) = result;
    } else {
        // Not aligned - use separate stores
        output[i] = __low2bfloat16(result);
        output[i + 1] = __high2bfloat16(result);
    }
}


// SiLU kernel for tensor [2048, 22016] - Vec4 version (4 elements per thread)
__global__ void silu_kernel_vec4(const bfloat16_t* __restrict__ data, 
                                 bfloat16_t* __restrict__ output,
                                 int rows, int cols) {
    // Global thread index (each thread processes 4 elements using 2 bfloat162 vectors)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Total elements to process (only first 11008 elements of each row)
    int total_elements = rows * 11008;
    
    if (i >= total_elements) return;
    
    // Process 4 elements as 2 bfloat162 vectors
    // Since total_elements is divisible by 4 (22544384 / 4 = 5636096), 
    // we don't need boundary checks for this specific case
    
    // Load first pair (elements 0,1)
    int row1 = i / 11008;
    int col1 = i % 11008;
    int row2 = (i + 1) / 11008;
    int col2 = (i + 1) % 11008;
    
    int input_idx1 = row1 * cols + col1;
    int mult_idx1 = row1 * cols + col1 + 11008;
    int input_idx2 = row2 * cols + col2;
    int mult_idx2 = row2 * cols + col2 + 11008;
    
    __nv_bfloat162 x1, mult1;
    
    // Check if we can use vectorized load for first pair
    if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x1 = *((const __nv_bfloat162*)&data[input_idx1]);
        mult1 = *((const __nv_bfloat162*)&data[mult_idx1]);
    } else {
        // Elements not contiguous - use separate loads
        x1 = __halves2bfloat162(data[input_idx1], data[input_idx2]);
        mult1 = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
    }
    
    // Load second pair (elements 2,3)
    int row3 = (i + 2) / 11008;
    int col3 = (i + 2) % 11008;
    int row4 = (i + 3) / 11008;
    int col4 = (i + 3) % 11008;
    
    int input_idx3 = row3 * cols + col3;
    int mult_idx3 = row3 * cols + col3 + 11008;
    int input_idx4 = row4 * cols + col4;
    int mult_idx4 = row4 * cols + col4 + 11008;
    
    __nv_bfloat162 x2, mult2;
    
    // Check if we can use vectorized load for second pair
    if (row3 == row4 && col4 == col3 + 1 && (input_idx3 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x2 = *((const __nv_bfloat162*)&data[input_idx3]);
        mult2 = *((const __nv_bfloat162*)&data[mult_idx3]);
    } else {
        // Elements not contiguous - use separate loads
        x2 = __halves2bfloat162(data[input_idx3], data[input_idx4]);
        mult2 = __halves2bfloat162(data[mult_idx3], data[mult_idx4]);
    }
    
    // Calculate SiLU for both bfloat162 vectors
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    
    // Process first pair
    __nv_bfloat162 neg_x1 = __hneg2(x1);
    __nv_bfloat162 exp_neg_x1 = h2exp(neg_x1);
    __nv_bfloat162 sigmoid_x1 = __h2div(one, __hadd2(one, exp_neg_x1));
    __nv_bfloat162 silu_x1 = __hmul2(x1, sigmoid_x1);
    __nv_bfloat162 result1 = __hmul2(silu_x1, mult1);
    
    // Process second pair
    __nv_bfloat162 neg_x2 = __hneg2(x2);
    __nv_bfloat162 exp_neg_x2 = h2exp(neg_x2);
    __nv_bfloat162 sigmoid_x2 = __h2div(one, __hadd2(one, exp_neg_x2));
    __nv_bfloat162 silu_x2 = __hmul2(x2, sigmoid_x2);
    __nv_bfloat162 result2 = __hmul2(silu_x2, mult2);
    
    // Store results using vectorized stores when possible
    // Output is always contiguous, so we can use vectorized stores
    if ((reinterpret_cast<uintptr_t>(&output[i]) & 3) == 0) {
        // Aligned - use single vectorized stores
        *(((__nv_bfloat162*)&output[i])) = result1;
        *(((__nv_bfloat162*)&output[i + 2])) = result2;
    } else {
        // Not aligned - use separate stores
        output[i] = __low2bfloat16(result1);
        output[i + 1] = __high2bfloat16(result1);
        output[i + 2] = __low2bfloat16(result2);
        output[i + 3] = __high2bfloat16(result2);
    }
}

// SiLU kernel for tensor [2048, 22016] - High vectorization (8 elements per thread)
__global__ void silu_kernel_vec8(const bfloat16_t* __restrict__ data, 
                                 bfloat16_t* __restrict__ output,
                                 int rows, int cols) {
    // Global thread index (each thread processes 8 elements using 4 bfloat162 vectors)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    // Total elements to process (only first 11008 elements of each row)
    int total_elements = rows * 11008;
    
    if (i >= total_elements) return;
    
    // Process 8 elements as 4 bfloat162 vectors
    // Since total_elements is divisible by 8 (22544384 / 8 = 2818048), 
    // we don't need boundary checks for this specific case
    
    // Load 8 elements and create 4 bfloat162 vectors
    __nv_bfloat162 x[4], mult[4];
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx1 = i + j * 2;
        int idx2 = i + j * 2 + 1;
        
        // Calculate row and column indices
        int row1 = idx1 / 11008;
        int col1 = idx1 % 11008;
        int row2 = idx2 / 11008;
        int col2 = idx2 % 11008;
        
        // Calculate actual indices in the input tensor
        int input_idx1 = row1 * cols + col1;
        int mult_idx1 = row1 * cols + col1 + 11008;
        int input_idx2 = row2 * cols + col2;
        int mult_idx2 = row2 * cols + col2 + 11008;
        
        // Check if we can use vectorized load for this pair
        if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
            // Elements are contiguous and aligned - use single load
            x[j] = *((const __nv_bfloat162*)&data[input_idx1]);
            mult[j] = *((const __nv_bfloat162*)&data[mult_idx1]);
        } else {
            // Elements not contiguous - use separate loads
            x[j] = __halves2bfloat162(data[input_idx1], data[input_idx2]);
            mult[j] = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
        }
    }
    
    // Calculate SiLU for all 4 bfloat162 vectors
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 result[4];
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        // Calculate SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        __nv_bfloat162 neg_x = __hneg2(x[j]);
        __nv_bfloat162 exp_neg_x = h2exp(neg_x);
        __nv_bfloat162 sigmoid_x = __h2div(one, __hadd2(one, exp_neg_x));
        __nv_bfloat162 silu_x = __hmul2(x[j], sigmoid_x);
        
        // Multiply by the corresponding elements
        result[j] = __hmul2(silu_x, mult[j]);
    }
    
    // Store results for all 4 bfloat162 vectors (8 elements total)
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int out_idx = i + j * 2;
        // Store results using vectorized stores when possible
        // Output is always contiguous, so we can use vectorized stores
        if ((reinterpret_cast<uintptr_t>(&output[out_idx]) & 3) == 0) {
            // Aligned - use single vectorized store
            *(((__nv_bfloat162*)&output[out_idx])) = result[j];
        } else {
            // Not aligned - use separate stores
            output[out_idx] = __low2bfloat16(result[j]);
            output[out_idx + 1] = __high2bfloat16(result[j]);
        }
    }
}

// ===== DUPLICATE KERNELS FOR BLOCK SIZE 128 TESTING =====

// SiLU kernel - Non-vectorized version (Block Size 128)
__global__ void silu_kernel_bs128(const bfloat16_t* __restrict__ data, 
                                  bfloat16_t* __restrict__ output,
                                  int rows, int cols) {
    // Same logic as silu_kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * 11008;
    if (i >= total_elements) return;
    
    int row = i / 11008;
    int col = i % 11008;
    int input_idx = row * cols + col;
    int mult_idx = row * cols + col + 11008;
    
    bfloat16_t x = data[input_idx];
    bfloat16_t mult = data[mult_idx];
    
    bfloat16_t one = (bfloat16_t)1.0f;
    bfloat16_t sigmoid_x = one / (one + hexp(-x));
    bfloat16_t silu_x = x * sigmoid_x;
    bfloat16_t result = silu_x * mult;
    
    output[i] = result;
}

// SiLU kernel - Vectorized version (Block Size 128)
__global__ void silu_kernel_vec2_bs128(const bfloat16_t* __restrict__ data, 
                                             bfloat16_t* __restrict__ output,
                                             int rows, int cols) {
    // Same logic as silu_kernel_vec2
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int total_elements = rows * 11008;
    if (i >= total_elements) return;
    
    int row1 = i / 11008;
    int col1 = i % 11008;
    int row2 = (i + 1) / 11008;
    int col2 = (i + 1) % 11008;
    
    int input_idx1 = row1 * cols + col1;
    int mult_idx1 = row1 * cols + col1 + 11008;
    int input_idx2 = row2 * cols + col2;
    int mult_idx2 = row2 * cols + col2 + 11008;
    
    // Load data as bfloat162 with smart vectorized loads
    __nv_bfloat162 x, mult;
    
    // Check if we can use vectorized load for input data
    if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x = *((const __nv_bfloat162*)&data[input_idx1]);
        mult = *((const __nv_bfloat162*)&data[mult_idx1]);
    } else {
        // Elements not contiguous - use separate loads
        x = __halves2bfloat162(data[input_idx1], data[input_idx2]);
        mult = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
    }
    
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 neg_x = __hneg2(x);
    __nv_bfloat162 exp_neg_x = h2exp(neg_x);
    __nv_bfloat162 sigmoid_x = __h2div(one, __hadd2(one, exp_neg_x));
    __nv_bfloat162 silu_x = __hmul2(x, sigmoid_x);
    __nv_bfloat162 result = __hmul2(silu_x, mult);
    
    // Store results using vectorized stores when possible
    // Output is always contiguous, so we can use vectorized stores
    if ((reinterpret_cast<uintptr_t>(&output[i]) & 3) == 0) {
        // Aligned - use single vectorized store
        *(((__nv_bfloat162*)&output[i])) = result;
    } else {
        // Not aligned - use separate stores
        output[i] = __low2bfloat16(result);
        output[i + 1] = __high2bfloat16(result);
    }
}



// SiLU kernel - Vec4 version (Block Size 128)
__global__ void silu_kernel_vec4_bs128(const bfloat16_t* __restrict__ data, 
                                       bfloat16_t* __restrict__ output,
                                       int rows, int cols) {
    // Same logic as silu_kernel_vec4
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total_elements = rows * 11008;
    if (i >= total_elements) return;
    
    // Load first pair
    int row1 = i / 11008;
    int col1 = i % 11008;
    int row2 = (i + 1) / 11008;
    int col2 = (i + 1) % 11008;
    
    int input_idx1 = row1 * cols + col1;
    int mult_idx1 = row1 * cols + col1 + 11008;
    int input_idx2 = row2 * cols + col2;
    int mult_idx2 = row2 * cols + col2 + 11008;
    
    __nv_bfloat162 x1, mult1;
    
    // Check if we can use vectorized load for first pair
    if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x1 = *((const __nv_bfloat162*)&data[input_idx1]);
        mult1 = *((const __nv_bfloat162*)&data[mult_idx1]);
    } else {
        // Elements not contiguous - use separate loads
        x1 = __halves2bfloat162(data[input_idx1], data[input_idx2]);
        mult1 = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
    }
    
    // Load second pair
    int row3 = (i + 2) / 11008;
    int col3 = (i + 2) % 11008;
    int row4 = (i + 3) / 11008;
    int col4 = (i + 3) % 11008;
    
    int input_idx3 = row3 * cols + col3;
    int mult_idx3 = row3 * cols + col3 + 11008;
    int input_idx4 = row4 * cols + col4;
    int mult_idx4 = row4 * cols + col4 + 11008;
    
    __nv_bfloat162 x2, mult2;
    
    // Check if we can use vectorized load for second pair
    if (row3 == row4 && col4 == col3 + 1 && (input_idx3 & 3) == 0) {
        // Elements are contiguous and aligned - use single load
        x2 = *((const __nv_bfloat162*)&data[input_idx3]);
        mult2 = *((const __nv_bfloat162*)&data[mult_idx3]);
    } else {
        // Elements not contiguous - use separate loads
        x2 = __halves2bfloat162(data[input_idx3], data[input_idx4]);
        mult2 = __halves2bfloat162(data[mult_idx3], data[mult_idx4]);
    }
    
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    
    // Process first pair
    __nv_bfloat162 neg_x1 = __hneg2(x1);
    __nv_bfloat162 exp_neg_x1 = h2exp(neg_x1);
    __nv_bfloat162 sigmoid_x1 = __h2div(one, __hadd2(one, exp_neg_x1));
    __nv_bfloat162 silu_x1 = __hmul2(x1, sigmoid_x1);
    __nv_bfloat162 result1 = __hmul2(silu_x1, mult1);
    
    // Process second pair
    __nv_bfloat162 neg_x2 = __hneg2(x2);
    __nv_bfloat162 exp_neg_x2 = h2exp(neg_x2);
    __nv_bfloat162 sigmoid_x2 = __h2div(one, __hadd2(one, exp_neg_x2));
    __nv_bfloat162 silu_x2 = __hmul2(x2, sigmoid_x2);
    __nv_bfloat162 result2 = __hmul2(silu_x2, mult2);
    
    // Store results using vectorized stores when possible
    // Output is always contiguous, so we can use vectorized stores
    if ((reinterpret_cast<uintptr_t>(&output[i]) & 3) == 0) {
        // Aligned - use single vectorized stores
        *(((__nv_bfloat162*)&output[i])) = result1;
        *(((__nv_bfloat162*)&output[i + 2])) = result2;
    } else {
        // Not aligned - use separate stores
        output[i] = __low2bfloat16(result1);
        output[i + 1] = __high2bfloat16(result1);
        output[i + 2] = __low2bfloat16(result2);
        output[i + 3] = __high2bfloat16(result2);
    }
}

// SiLU kernel - Vec8 version (Block Size 128)
__global__ void silu_kernel_vec8_bs128(const bfloat16_t* __restrict__ data, 
                                       bfloat16_t* __restrict__ output,
                                       int rows, int cols) {
    // Same logic as silu_kernel_vec8
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int total_elements = rows * 11008;
    if (i >= total_elements) return;
    
    __nv_bfloat162 x[4], mult[4];
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx1 = i + j * 2;
        int idx2 = i + j * 2 + 1;
        
        int row1 = idx1 / 11008;
        int col1 = idx1 % 11008;
        int row2 = idx2 / 11008;
        int col2 = idx2 % 11008;
        
        int input_idx1 = row1 * cols + col1;
        int mult_idx1 = row1 * cols + col1 + 11008;
        int input_idx2 = row2 * cols + col2;
        int mult_idx2 = row2 * cols + col2 + 11008;
        
        // Check if we can use vectorized load for this pair
        if (row1 == row2 && col2 == col1 + 1 && (input_idx1 & 3) == 0) {
            // Elements are contiguous and aligned - use single load
            x[j] = *((const __nv_bfloat162*)&data[input_idx1]);
            mult[j] = *((const __nv_bfloat162*)&data[mult_idx1]);
        } else {
            // Elements not contiguous - use separate loads
            x[j] = __halves2bfloat162(data[input_idx1], data[input_idx2]);
            mult[j] = __halves2bfloat162(data[mult_idx1], data[mult_idx2]);
        }
    }
    
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 result[4];
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        __nv_bfloat162 neg_x = __hneg2(x[j]);
        __nv_bfloat162 exp_neg_x = h2exp(neg_x);
        __nv_bfloat162 sigmoid_x = __h2div(one, __hadd2(one, exp_neg_x));
        __nv_bfloat162 silu_x = __hmul2(x[j], sigmoid_x);
        result[j] = __hmul2(silu_x, mult[j]);
    }
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int out_idx = i + j * 2;
        // Store results using vectorized stores when possible
        // Output is always contiguous, so we can use vectorized stores
        if ((reinterpret_cast<uintptr_t>(&output[out_idx]) & 3) == 0) {
            // Aligned - use single vectorized store
            *(((__nv_bfloat162*)&output[out_idx])) = result[j];
        } else {
            // Not aligned - use separate stores
            output[out_idx] = __low2bfloat16(result[j]);
            output[out_idx + 1] = __high2bfloat16(result[j]);
        }
    }
}

// Cache warming kernel to preload tensors into L2 cache
__global__ void cache_warming_kernel(const bfloat16_t* __restrict__ input_data,
                                    int input_size) {
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Read input tensor to warm L2 cache
    if (input_idx < input_size) {
        volatile bfloat16_t dummy = input_data[input_idx];
        (void)dummy;  // Prevent optimization
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)



// Initialize random data
void initialize_data(std::vector<bfloat16_t>& data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = float_to_bfloat16(dis(gen));
    }
}

int main() {
    const int rows = 2048;
    const int cols = 22016;
    const int input_size = rows * cols;
    const int output_size = rows * 11008;  // Only processing first 11008 elements per row
    
    std::cout << "SiLU Kernel Test" << std::endl;
    std::cout << "Input tensor shape: [" << rows << ", " << cols << "]" << std::endl;
    std::cout << "Output tensor shape: [" << rows << ", " << 11008 << "]" << std::endl;
    std::cout << "Total input elements: " << input_size << std::endl;
    std::cout << "Total output elements: " << output_size << std::endl;
    
    // Allocate host memory
    std::vector<bfloat16_t> h_input(input_size);
    std::vector<bfloat16_t> h_output_gpu(output_size);
    
    // Initialize input data
    std::cout << "Initializing random input data..." << std::endl;
    initialize_data(h_input, input_size);
    
    // Allocate device memory
    bfloat16_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(bfloat16_t)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(bfloat16_t)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                         input_size * sizeof(bfloat16_t), 
                         cudaMemcpyHostToDevice));
    
    // Test both block sizes
    const int num_iterations = 100;
    
    // Create arrays of CUDA events for precise GPU timing (one pair per iteration)
    cudaEvent_t start_events[num_iterations], stop_events[num_iterations];
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaEventCreate(&start_events[i]));
        CUDA_CHECK(cudaEventCreate(&stop_events[i]));
    }

    float total_time_ms; // Declare here for reuse across measurements
    
    // Calculate grid size for cache warming
    const int cache_block_size = 256;
    const int cache_grid_size_input = (input_size + cache_block_size - 1) / cache_block_size;
    const int cache_grid_size_output = (output_size + cache_block_size - 1) / cache_block_size;
    const int cache_grid_size = max(cache_grid_size_input, cache_grid_size_output);
    
    // Test both block sizes
    const int block_size_512 = 512;
    const int block_size_128 = 128;
    
    std::cout << "\n==============================================" << std::endl;
    std::cout << "          BLOCK SIZE 512 TESTING             " << std::endl;
    std::cout << "==============================================" << std::endl;
    
    // === Test Non-Vectorized Kernel (Block Size 512) ===
    const int grid_size = (output_size + block_size_512 - 1) / block_size_512;
    
    std::cout << "\n=== Non-Vectorized Kernel ===" << std::endl;
    std::cout << "Block size: " << block_size_512 << std::endl;
    std::cout << "Grid size: " << grid_size << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel<<<grid_size, block_size_512>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays - minimal overhead approach
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel<<<grid_size, block_size_512>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_nonvec = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_nonvec << " us" << std::endl;
    
    // Calculate throughput
    double elements_per_sec = output_size / (avg_time_us_nonvec / 1e6);
    double gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_nonvec / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    
    // === Test Vectorized Kernel ===
    const int grid_size_vec = ((output_size + 1) / 2 + block_size_512 - 1) / block_size_512;  // Half the threads since each processes 2 elements
    
    std::cout << "\n=== Vect2 Kernel ===" << std::endl;
    std::cout << "Block size: " << block_size_512 << std::endl;
    std::cout << "Grid size: " << grid_size_vec << " (half threads, 2 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec2<<<grid_size_vec, block_size_512>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec2<<<grid_size_vec, block_size_512>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    


    // === Test Vec4 Kernel ===
    const int grid_size_vec4 = ((output_size + 3) / 4 + block_size_512 - 1) / block_size_512;  // 1/4 threads since each processes 4 elements
    
    std::cout << "\n=== Vec4 Kernel (4 elements/thread) ===" << std::endl;
    std::cout << "Block size: " << block_size_512 << std::endl;
    std::cout << "Grid size: " << grid_size_vec4 << " (1/4 threads, 4 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec4<<<grid_size_vec4, block_size_512>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with CUDA events
    total_time_ms = 0.0f;
    
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec4<<<grid_size_vec4, block_size_512>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec4 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec4 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec4 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec4 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    // === Test Vec8 Kernel ===
    const int grid_size_vec8 = ((output_size + 7) / 8 + block_size_512 - 1) / block_size_512;  // 1/8 threads since each processes 8 elements
    
    std::cout << "\n=== Vec8 Kernel (8 elements/thread) ===" << std::endl;
    std::cout << "Block size: " << block_size_512 << std::endl;
    std::cout << "Grid size: " << grid_size_vec8 << " (1/8 threads, 8 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec8<<<grid_size_vec8, block_size_512>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec8<<<grid_size_vec8, block_size_512>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec8 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec8 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec8 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec8 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    // ===================================================================
    std::cout << "\n==============================================" << std::endl;
    std::cout << "          BLOCK SIZE 128 TESTING             " << std::endl;
    std::cout << "==============================================" << std::endl;

    // === Test Non-Vectorized Kernel (Block Size 128) ===
    const int grid_size_128 = (output_size + block_size_128 - 1) / block_size_128;
    
    std::cout << "\n=== Non-Vectorized Kernel (BS128) ===" << std::endl;
    std::cout << "Block size: " << block_size_128 << std::endl;
    std::cout << "Grid size: " << grid_size_128 << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_bs128<<<grid_size_128, block_size_128>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_bs128<<<grid_size_128, block_size_128>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_nonvec_128 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_nonvec_128 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_nonvec_128 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_nonvec_128 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    // === Test Vectorized Kernel (Block Size 128) ===
    const int grid_size_vec_128 = ((output_size + 1) / 2 + block_size_128 - 1) / block_size_128;
    
    std::cout << "\n=== Vec2 Kernel (BS128) ===" << std::endl;
    std::cout << "Block size: " << block_size_128 << std::endl;
    std::cout << "Grid size: " << grid_size_vec_128 << " (half threads, 2 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec2_bs128<<<grid_size_vec_128, block_size_128>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec2_bs128<<<grid_size_vec_128, block_size_128>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec_128 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec_128 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec_128 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec_128 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;



    // === Test Vec4 Kernel (Block Size 128) ===
    const int grid_size_vec4_128 = ((output_size + 3) / 4 + block_size_128 - 1) / block_size_128;
    
    std::cout << "\n=== Vec4 Kernel (BS128) ===" << std::endl;
    std::cout << "Block size: " << block_size_128 << std::endl;
    std::cout << "Grid size: " << grid_size_vec4_128 << " (1/4 threads, 4 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec4_bs128<<<grid_size_vec4_128, block_size_128>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec4_bs128<<<grid_size_vec4_128, block_size_128>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec4_128 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec4_128 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec4_128 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec4_128 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    // === Test Vec8 Kernel (Block Size 128) ===
    const int grid_size_vec8_128 = ((output_size + 7) / 8 + block_size_128 - 1) / block_size_128;
    
    std::cout << "\n=== Vec8 Kernel (BS128) ===" << std::endl;
    std::cout << "Block size: " << block_size_128 << std::endl;
    std::cout << "Grid size: " << grid_size_vec8_128 << " (1/8 threads, 8 elements each)" << std::endl;
    
    // Warm up with cache warming
    for (int i = 0; i < 10; i++) {
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        silu_kernel_vec8_bs128<<<grid_size_vec8_128, block_size_128>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement with optimized event arrays
    for (int i = 0; i < num_iterations; i++) {
        // Warm L2 cache before each measurement
        cache_warming_kernel<<<cache_grid_size, cache_block_size>>>(d_input, input_size);
        
        // Record start event for this iteration
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        
        silu_kernel_vec8_bs128<<<grid_size_vec8_128, block_size_128>>>(d_input, d_output, rows, cols);
        
        // Record stop event for this iteration
        CUDA_CHECK(cudaEventRecord(stop_events[i]));
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate total elapsed time outside the measurement loop
    total_time_ms = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start_events[i], stop_events[i]));
        total_time_ms += iter_time_ms;
    }
    
    double avg_time_us_vec8_128 = (total_time_ms * 1000.0) / num_iterations;
    
    std::cout << "Time: " << avg_time_us_vec8_128 << " us" << std::endl;
    
    // Calculate throughput
    elements_per_sec = output_size / (avg_time_us_vec8_128 / 1e6);
    gb_per_sec = (input_size * sizeof(bfloat16_t) + output_size * sizeof(bfloat16_t)) / (avg_time_us_vec8_128 / 1e6) / 1e9;
    
    std::cout << "Throughput: " << elements_per_sec / 1e9 << " Gelements/s" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    // === Performance Comparison ===
    std::cout << "\n=== Performance Comparison (Block Size 512) ===" << std::endl;
    std::cout << "Non-vectorized: " << avg_time_us_nonvec << " us" << std::endl;
    std::cout << "Vec2:           " << avg_time_us_vec << " us" << std::endl;
    std::cout << "Vec4:           " << avg_time_us_vec4 << " us" << std::endl;
    std::cout << "Vec8:           " << avg_time_us_vec8 << " us" << std::endl;

    std::cout << "\n=== Performance Comparison (Block Size 128) ===" << std::endl;
    std::cout << "Non-vectorized: " << avg_time_us_nonvec_128 << " us" << std::endl;
    std::cout << "Vec2:           " << avg_time_us_vec_128 << " us" << std::endl;
    std::cout << "Vec4:           " << avg_time_us_vec4_128 << " us" << std::endl;
    std::cout << "Vec8:           " << avg_time_us_vec8_128 << " us" << std::endl;

    std::cout << "\n=== Block Size 128 vs 512 Comparison ===" << std::endl;
    std::cout << "Non-vectorized 128/512 ratio: " << avg_time_us_nonvec_128 / avg_time_us_nonvec << "x" << std::endl;
    std::cout << "Vec2           128/512 ratio: " << avg_time_us_vec_128 / avg_time_us_vec << "x" << std::endl;
    std::cout << "Vec4           128/512 ratio: " << avg_time_us_vec4_128 / avg_time_us_vec4 << "x" << std::endl;
    std::cout << "Vec8           128/512 ratio: " << avg_time_us_vec8_128 / avg_time_us_vec8 << "x" << std::endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, 
                         output_size * sizeof(bfloat16_t), 
                         cudaMemcpyDeviceToHost));
    
    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    // Destroy event arrays
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaEventDestroy(start_events[i]));
        CUDA_CHECK(cudaEventDestroy(stop_events[i]));
    }
    
    return 0;
} 