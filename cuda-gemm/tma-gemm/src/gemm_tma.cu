#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdio.h>
#include <mma.h>
#include "../include/gemm.h"

namespace tma_gemm {

// Debug info structure
struct DebugInfo {
    dim3 grid;
    dim3 block;
    float executionTimeMs;
    int sharedMemBytes;
} debugInfo;

// Tensor Core GEMM implementation using the WMMA API
// This uses the proper H100 Tensor Core operations
__global__ void gemm_tma_tensor_kernel(
    const __nv_bfloat16* A,    // M_DIM x K_DIM (row major)
    const __nv_bfloat16* B,    // K_DIM x N_DIM (column major)
    __nv_bfloat16* C           // M_DIM x N_DIM (row major)
) {
    // WMMA fragment shapes - H100 supports different configs
    // Use 16x16x16 for bfloat16
    const int M_TILE = 16;
    const int N_TILE = 16;
    const int K_TILE = 16;
    
    // Each warp processes one 16x16 output tile
    // Calculate warp location in output matrix
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;
    
    // Matrix coordinates
    int outputM = warpM * M_TILE;
    int outputN = warpN * N_TILE;
    
    // Use accumulator fragments, which store temporary results
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> c_frag;
    
    // Initialize the accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over the K dimension in steps of K_TILE
    for (int k = 0; k < K_DIM; k += K_TILE) {
        // Load tiles from global memory
        if (outputM < M_DIM && k < K_DIM) {
            wmma::load_matrix_sync(a_frag, A + outputM * K_DIM + k, K_DIM);
        } else {
            wmma::fill_fragment(a_frag, 0.0f);
        }
        
        if (k < K_DIM && outputN < N_DIM) {
            wmma::load_matrix_sync(b_frag, B + outputN * K_DIM + k, K_DIM);
        } else {
            wmma::fill_fragment(b_frag, 0.0f);
        }
        
        // Perform matrix multiplication using tensor cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store the result back to global memory
    if (outputM < M_DIM && outputN < N_DIM) {
        wmma::store_matrix_sync(C + outputM * N_DIM + outputN, c_frag, N_DIM, wmma::mem_row_major);
    }
}

// Use a generic tiled implementation when tensor cores are not needed
// This is our stable fallback version
__global__ void gemm_tma_kernel(
    const __nv_bfloat16* A,    // M_DIM x K_DIM (row major)
    const __nv_bfloat16* B,    // K_DIM x N_DIM (column major)
    __nv_bfloat16* C           // M_DIM x N_DIM (row major)
) {
    // Block size
    const int TILE_SIZE = 16;
    
    // Create shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Register to accumulate results
    float sum = 0.0f;

    // Loop over tiles
    #pragma unroll 4
    for (int t = 0; t < (K_DIM + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile - convert to float immediately for better precision
        if (row < M_DIM && t * TILE_SIZE + threadIdx.x < K_DIM) {
            As[threadIdx.y][threadIdx.x] = (float)A[row * K_DIM + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile - convert to float immediately (note column-major layout)
        if (col < N_DIM && t * TILE_SIZE + threadIdx.y < K_DIM) {
            Bs[threadIdx.y][threadIdx.x] = (float)B[col * K_DIM + t * TILE_SIZE + threadIdx.y];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Ensure shared memory loads are complete
        __syncthreads();
        
        // Compute partial dot product for this tile with loop unrolling
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Ensure computations are complete before loading next tile
        __syncthreads();
    }
    
    // Write result if within bounds
    if (row < M_DIM && col < N_DIM) {
        C[row * N_DIM + col] = __nv_bfloat16(sum);
    }
}

// Helper function to check if the device supports tensor cores
bool deviceSupportsTensorCores() {
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    // Tensor core support was introduced in compute capability 7.0
    return (deviceProp.major > 7 || (deviceProp.major == 7 && deviceProp.minor >= 0));
}

void gemm_tma(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C) {
    // Check if device supports tensor cores
    bool use_tensor_cores = deviceSupportsTensorCores();
    
    if (use_tensor_cores) {
        // Configuration for tensor core implementation
        const int M_TILE = 16;
        const int N_TILE = 16;
        
        // Calculate grid dimensions - each block processes multiple tiles
        int warpSize = 32;
        dim3 blockDim(4, 8 * warpSize); // 8 warps per block
        dim3 gridDim((N_DIM + N_TILE - 1) / N_TILE, 
                     (M_DIM + M_TILE - 1) / M_TILE * warpSize / blockDim.y);
        
        // Launch tensor core kernel
        gemm_tma_tensor_kernel<<<gridDim, blockDim>>>(A, B, C);
    } else {
        // Fall back to our regular implementation
        const int TILE_SIZE = 16;
        
        // Set up grid and blocks
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N_DIM + TILE_SIZE - 1) / TILE_SIZE, (M_DIM + TILE_SIZE - 1) / TILE_SIZE);
        
        // Calculate shared memory size
        const int sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
        
        // Save configuration for debugging
        debugInfo.grid = gridDim;
        debugInfo.block = blockDim;
        debugInfo.sharedMemBytes = sharedMemSize;
        
        // Set L1 cache preference for shared memory
        cudaFuncSetCacheConfig(gemm_tma_kernel, cudaFuncCachePreferShared);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Record start event
        cudaEventRecord(start);
        
        // Launch kernel
        gemm_tma_kernel<<<gridDim, blockDim, sharedMemSize>>>(A, B, C);
        
        // Record stop event and synchronize
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate execution time
        cudaEventElapsedTime(&debugInfo.executionTimeMs, start, stop);
        
        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

#ifdef DEBUG_TMA
void print_debug_info() {
    printf("=== GEMM Tensor Core Debug Info ===\n");
    printf("Matrix dimensions: A[%d x %d] * B[%d x %d] = C[%d x %d]\n", 
           M_DIM, K_DIM, K_DIM, N_DIM, M_DIM, N_DIM);
    printf("Grid dimensions: (%d, %d, %d)\n", 
           debugInfo.grid.x, debugInfo.grid.y, debugInfo.grid.z);
    printf("Block dimensions: (%d, %d, %d)\n", 
           debugInfo.block.x, debugInfo.block.y, debugInfo.block.z);
    printf("Total threads: %d\n", 
           debugInfo.grid.x * debugInfo.grid.y * debugInfo.grid.z * 
           debugInfo.block.x * debugInfo.block.y * debugInfo.block.z);
    printf("Shared memory: %d bytes\n", debugInfo.sharedMemBytes);
    printf("Execution time: %.3f ms\n", debugInfo.executionTimeMs);
    printf("Device supports tensor cores: %s\n", deviceSupportsTensorCores() ? "Yes" : "No");
    
    // Performance metrics
    double ops = 2.0 * M_DIM * N_DIM * K_DIM; // MUL + ADD for each element
    double gflops = (ops / (debugInfo.executionTimeMs * 1e-3)) / 1e9;
    double bandwidth = (M_DIM*K_DIM + K_DIM*N_DIM + M_DIM*N_DIM) * sizeof(__nv_bfloat16) / (debugInfo.executionTimeMs * 1e-3) / 1e9;
    
    printf("Arithmetic intensity: %.2f FLOP/byte\n", 
           ops / ((M_DIM*K_DIM + K_DIM*N_DIM + M_DIM*N_DIM) * sizeof(__nv_bfloat16)));
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("============================\n");
}
#endif

} // namespace tma_gemm 