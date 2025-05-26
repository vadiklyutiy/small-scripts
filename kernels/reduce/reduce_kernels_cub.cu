#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// Kernel 1: 32 threads, CUB BlockReduce
__global__ void reduce_mean_32_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/32 = 64)
    float thread_sum = 0.0f;
    for (int i = 0; i < 64; i++) {
        int idx = tid + i * 32;
        if (idx < width) {
            thread_sum += input[row * width + idx];
        }
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 32>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
}

// Kernel 2: 64 threads, CUB BlockReduce
__global__ void reduce_mean_64_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/64 = 32)
    float thread_sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = tid + i * 64;
        if (idx < width) {
            thread_sum += input[row * width + idx];
        }
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 64>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
}

// Kernel 3: 128 threads, CUB BlockReduce
__global__ void reduce_mean_128_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/128 = 16)
    float thread_sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 128;
        if (idx < width) {
            thread_sum += input[row * width + idx];
        }
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
}

// Kernel 4: 256 threads, CUB BlockReduce
__global__ void reduce_mean_256_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/256 = 8)
    float thread_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        if (idx < width) {
            thread_sum += input[row * width + idx];
        }
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
}

// Kernel 5: 512 threads, CUB BlockReduce
__global__ void reduce_mean_512_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/512 = 4)
    float thread_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 512;
        if (idx < width) {
            thread_sum += input[row * width + idx];
        }
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
}

// Kernel 6: 1024 threads, CUB BlockReduce
__global__ void reduce_mean_1024_cub(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread loads multiple elements (2048/1024 = 2)
    float thread_sum = 0.0f;
    
    // Load first element
    if (tid < width) {
        thread_sum += input[row * width + tid];
    }
    
    // Load second element (offset by 1024)
    if (tid + 1024 < width) {
        thread_sum += input[row * width + tid + 1024];
    }
    
    // CUB BlockReduce
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // Write result (only thread 0)
    if (tid == 0) {
        output[row] = block_sum / width;
    }
} 