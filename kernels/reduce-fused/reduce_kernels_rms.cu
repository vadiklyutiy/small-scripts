#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum_optimized(float val) {
    int32_t mask = __activemask();
    val += __shfl_down_sync(mask, val, 16, 32);
    val += __shfl_down_sync(mask, val, 8, 32);
    val += __shfl_down_sync(mask, val, 4, 32);
    val += __shfl_down_sync(mask, val, 2, 32);
    val += __shfl_down_sync(mask, val, 1, 32);
    return __shfl_sync(mask, val, 0, 32);  // Broadcast result to all threads in warp
}

// Kernel 1: 32 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_32_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads 64 elements (2048/32 = 64) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 64; i++) {
        int idx = tid + i * 32;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Write result (only thread 0): sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((rv / width) + 1e-06f);
    }
}

// Kernel 2: 32 threads, separate (same as atomic for single warp)
__global__ void reduce_rms_32_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads 64 elements (2048/32 = 64) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 64; i++) {
        int idx = tid + i * 32;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Write result (only thread 0): sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((rv / width) + 1e-06f);
    }
}

// Kernel 3: 64 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_64_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads 32 elements (2048/64 = 32) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = tid + i * 64;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Atomic add from each warp leader
    if ((tid % 32) == 0) {
        atomicAdd(&shared_sum, rv);
    }
    __syncthreads();
    
    // Write final result: sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((shared_sum / width) + 1e-06f);
    }
}

// Kernel 4: 64 threads, separate (RMS with epsilon)
__global__ void reduce_rms_64_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[2];
    
    // Each thread loads 32 elements (2048/64 = 32) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = tid + i * 64;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Store warp result to shared memory
    if ((tid % 32) == 0) {
        warp_sums[warp_id] = rv;
    }
    __syncthreads();
    
    // Final reduction by thread 0: sqrt(mean + epsilon)
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            total_sum += warp_sums[i];
        }
        output[row] = sqrtf((total_sum / width) + 1e-06f);
    }
}

// Kernel 5: 128 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_128_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads 16 elements (2048/128 = 16) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 128;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Atomic add from each warp leader
    if ((tid % 32) == 0) {
        atomicAdd(&shared_sum, rv);
    }
    __syncthreads();
    
    // Write final result: sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((shared_sum / width) + 1e-06f);
    }
}

// Kernel 6: 128 threads, separate (RMS with epsilon)
__global__ void reduce_rms_128_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[4];
    
    // Each thread loads 16 elements (2048/128 = 16) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 128;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Store warp result to shared memory
    if ((tid % 32) == 0) {
        warp_sums[warp_id] = rv;
    }
    __syncthreads();
    
    // Final reduction by thread 0: sqrt(mean + epsilon)
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            total_sum += warp_sums[i];
        }
        output[row] = sqrtf((total_sum / width) + 1e-06f);
    }
}

// Kernel 7: 256 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_256_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads 8 elements (2048/256 = 8) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Atomic add from each warp leader
    if ((tid % 32) == 0) {
        atomicAdd(&shared_sum, rv);
    }
    __syncthreads();
    
    // Write final result: sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((shared_sum / width) + 1e-06f);
    }
}

// Kernel 8: 256 threads, separate (RMS with epsilon)
__global__ void reduce_rms_256_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[8];
    
    // Each thread loads 8 elements (2048/256 = 8) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Store warp result to shared memory
    if ((tid % 32) == 0) {
        warp_sums[warp_id] = rv;
    }
    __syncthreads();
    
    // Final reduction by thread 0: sqrt(mean + epsilon)
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            total_sum += warp_sums[i];
        }
        output[row] = sqrtf((total_sum / width) + 1e-06f);
    }
}

// Kernel 9: 512 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_512_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads 4 elements (2048/512 = 4) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 512;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Atomic add from each warp leader
    if ((tid % 32) == 0) {
        atomicAdd(&shared_sum, rv);
    }
    __syncthreads();
    
    // Write final result: sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((shared_sum / width) + 1e-06f);
    }
}

// Kernel 10: 512 threads, separate (RMS with epsilon)
__global__ void reduce_rms_512_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[16];
    
    // Each thread loads 4 elements (2048/512 = 4) and squares them
    float rv = 0.0f;
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 512;
        if (idx < width) {
            float val = input[row * width + idx];
            rv += val * val;  // Square the input
        }
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Store warp result to shared memory
    if ((tid % 32) == 0) {
        warp_sums[warp_id] = rv;
    }
    __syncthreads();
    
    // Final reduction by thread 0: sqrt(mean + epsilon)
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            total_sum += warp_sums[i];
        }
        output[row] = sqrtf((total_sum / width) + 1e-06f);
    }
}

// Kernel 11: 1024 threads, atomic (RMS with epsilon)
__global__ void reduce_rms_1024_atomic(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads 2 elements (2048/1024 = 2) and squares them
    float rv = 0.0f;
    
    // Load first element
    if (tid < width) {
        float val = input[row * width + tid];
        rv += val * val;
    }
    
    // Load second element (offset by 1024)
    if (tid + 1024 < width) {
        float val = input[row * width + tid + 1024];
        rv += val * val;
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Atomic add from each warp leader
    if ((tid % 32) == 0) {
        atomicAdd(&shared_sum, rv);
    }
    __syncthreads();
    
    // Write final result: sqrt(mean + epsilon)
    if (tid == 0) {
        output[row] = sqrtf((shared_sum / width) + 1e-06f);
    }
}

// Kernel 12: 1024 threads, separate (RMS with epsilon)
__global__ void reduce_rms_1024_separate(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[32];
    
    // Each thread loads 2 elements (2048/1024 = 2) and squares them
    float rv = 0.0f;
    
    // Load first element
    if (tid < width) {
        float val = input[row * width + tid];
        rv += val * val;
    }
    
    // Load second element (offset by 1024)
    if (tid + 1024 < width) {
        float val = input[row * width + tid + 1024];
        rv += val * val;
    }
    
    // Immediate warp-level shuffle reduction
    int32_t mask = __activemask();
    rv += __shfl_down_sync(mask, rv, 16, 32);
    rv += __shfl_down_sync(mask, rv, 8, 32);
    rv += __shfl_down_sync(mask, rv, 4, 32);
    rv += __shfl_down_sync(mask, rv, 2, 32);
    rv += __shfl_down_sync(mask, rv, 1, 32);
    rv = __shfl_sync(mask, rv, 0, 32);
    
    // Store warp result to shared memory
    if ((tid % 32) == 0) {
        warp_sums[warp_id] = rv;
    }
    __syncthreads();
    
    // Final reduction by thread 0: sqrt(mean + epsilon)
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            total_sum += warp_sums[i];
        }
        output[row] = sqrtf((total_sum / width) + 1e-06f);
    }
} 