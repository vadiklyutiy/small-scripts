# SiLU CUDA Kernel Implementation

This directory contains a comprehensive CUDA kernel implementation for the SiLU (Sigmoid Linear Unit) activation function applied to bfloat16 tensors with multiple vectorization optimizations.

## Problem Description

Given an input tensor of shape [2048, 22016] with bfloat16 data type:
- For each j in [0, 11007]: compute `silu(input[j]) * input[j + 11008]`
- Where `silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))`
- Output tensor shape: [2048, 11008]
- Total elements processed: 22,544,384

## Files

- `silu_kernel.cu`: Complete CUDA kernel implementation with multiple vectorization approaches
- `Makefile`: Build configuration using optimized nvcc compilation flags  
- `README.md`: This documentation file

## Kernel Implementations

### 1. Non-Vectorized Kernel
- **1 element per thread**: Standard scalar implementation
- **Block sizes tested**: 512 and 128

### 2. Vectorized Kernel (Vec2)
- **2 elements per thread**: Uses `__nv_bfloat162` intrinsics
- **Smart vectorized loads**: Automatic detection of contiguous/aligned data
- **Block sizes tested**: 512 and 128

### 3. Vec4 Kernel ⭐ **OPTIMAL**
- **4 elements per thread**: Uses 2×`__nv_bfloat162` vectors
- **Best overall performance**: Optimal with block size 128 (48.0 μs)
- **Block sizes tested**: 512 and 128

### 4. Vec8 Kernel ⭐ **COMPETITIVE**
- **8 elements per thread**: Uses 4×`__nv_bfloat162` vectors
- **Excellent performance**: Very close to Vec4 (48.8-48.9 μs)
- **Block sizes tested**: 512 and 128

## Performance Results (H100 GPU)

### Block Size 512 Results

| Kernel Type | Time (μs) | Throughput (Gelements/s) | Bandwidth (GB/s) | Speedup |
|-------------|-----------|--------------------------|------------------|---------|
| Non-Vectorized | 81.4 | 276.9 | 1,661.6 | 1.00x (baseline) |
| Vec2 | 58.8 | 383.4 | 2,300.5 | 1.38x |
| Vec4 | 50.0 | 450.7 | 2,704.4 | 1.63x |
| **Vec8** | **48.9** | **461.0** | **2,766.1** | **1.66x** ⭐ |

### Block Size 128 Results

| Kernel Type | Time (μs) | Throughput (Gelements/s) | Bandwidth (GB/s) | Speedup vs BS512 |
|-------------|-----------|--------------------------|------------------|------------------|
| Non-Vectorized | 110.4 | 204.2 | 1,225.5 | 0.74x (worse) |
| Vec2 | 58.1 | 387.9 | 2,327.3 | 1.01x |
| **Vec4** | **48.0** | **469.7** | **2,818.1** | **1.04x** ⭐ |
| Vec8 | 48.8 | 462.2 | 2,773.2 | 1.00x |

### Block Size Analysis

**Optimal Configurations:**
1. **Vec4 + Block Size 128**: **48.0 μs** - Overall best performance
2. **Vec8 + Block Size 512**: **48.9 μs** - Close second  
3. **Vec8 + Block Size 128**: **48.8 μs** - Very close third

**Key Insights:**
- **Block Size 512 is better for non-vectorized** kernels (81.4 vs 110.4 μs)
- **Block Size 128 gives slight edge** for highly vectorized kernels (Vec4/Vec8)
- **Vec8 performance significantly improved** - now competitive with Vec4
- **Memory bandwidth peaks at ~2.82 TB/s** (Vec4 + BS128)

## Key Technical Insights

### Register Usage vs Performance Trade-off
- **Non-Vectorized**: 12 registers → Good occupancy, but less compute efficiency
- **Vec2**: 14 registers → Good balance, solid improvement  
- **Vec4**: 21 registers → High performance, manageable register pressure
- **Vec8**: 32 registers → **Now optimized** - competitive performance despite high register usage

### Memory Bandwidth Analysis
- **Theoretical Peak**: ~3.35 TB/s (H100 HBM3)
- **Achieved Peak**: ~2.82 TB/s (Vec4 + BS128)
- **Efficiency**: ~84% of theoretical peak
- **Bottleneck**: Memory-bound workload with excellent optimization

### Vectorization Performance Scaling
- **1 element/thread**: Baseline performance
- **2 elements/thread**: 1.38x improvement, excellent efficiency
- **4 elements/thread**: 1.63x improvement, strong scaling continues
- **8 elements/thread**: 1.66x improvement, **optimal with right block size**

### Block Size Impact Analysis
- **Small Block Size (128)**: 
  - **Bad for simple kernels** (non-vectorized: 0.74x performance)
  - **Good for complex kernels** (Vec4: 1.04x, Vec8: 1.00x)
  - Better register utilization per SM
- **Large Block Size (512)**:
  - **Good for simple kernels** (baseline performance)
  - **Competitive for complex kernels** (Vec8 actually peaks here)
  - Better occupancy for low-register kernels

## Building and Running

```bash
# Compile the kernel
make

# Run comprehensive performance test
./silu_test

# Clean build artifacts  
make clean
```

## Sample Output

```
SiLU Kernel Test
Input tensor shape: [2048, 22016]
Output tensor shape: [2048, 11008]
Total input elements: 45088768
Total output elements: 22544384
Initializing random input data...

==============================================
          BLOCK SIZE 512 TESTING
==============================================

=== Non-Vectorized Kernel ===
Block size: 512
Grid size: 44032
Time: 81.4074 us
Throughput: 276.933 Gelements/s
Bandwidth: 1661.6 GB/s

=== Vec4 Kernel (4 elements/thread) ===
Block size: 512  
Grid size: 11008 (1/4 threads, 4 elements each)
Time: 50.0179 us
Throughput: 450.726 Gelements/s
Bandwidth: 2704.36 GB/s

=== Vec8 Kernel (8 elements/thread) ===
Block size: 512
Grid size: 5504 (1/8 threads, 8 elements each)
Time: 48.9024 us
Throughput: 461.008 Gelements/s
Bandwidth: 2766.05 GB/s

=== Vec4 Kernel (BS128) ===
Block size: 128
Grid size: 44032 (1/4 threads, 4 elements each)
Time: 48 us
Throughput: 469.675 Gelements/s
Bandwidth: 2818.05 GB/s

=== Performance Comparison (Block Size 512) ===
Non-vectorized: 81.4074 us
Vectorized:     58.7984 us
Vec4:           50.0179 us
Vec8:           48.9024 us

=== Block Size 128 vs 512 Comparison ===
Vec4           128/512 ratio: 0.959656x
Vec8           128/512 ratio: 0.997409x
```

## Architecture Details

### Memory Layout
- **Input tensor**: [2048, 22016] = 45,088,768 elements
- **Processing scope**: First 11,008 elements of each row  
- **Output tensor**: [2048, 11008] = 22,544,384 elements
- **Memory access pattern**: Coalesced loads with vectorization

### Thread Mapping (Vec4 Example)
```cuda
thread_id = blockIdx.x * blockDim.x + threadIdx.x
elements_per_thread = 4
base_idx = thread_id * elements_per_thread

// Process 4 elements as 2 bfloat162 vectors
for (element in [0,1,2,3]) {
    global_idx = base_idx + element
    row = global_idx / 11008
    col = global_idx % 11008
    input_idx = row * 22016 + col
    mult_idx = row * 22016 + col + 11008
}
```

### Optimized Computation (using bfloat162 intrinsics)
```cuda
// Load 2 elements as bfloat162
__nv_bfloat162 x = load_vectorized_data(input_idx);
__nv_bfloat162 mult = load_vectorized_data(mult_idx);

// Vectorized SiLU computation
__nv_bfloat162 one = __float2bfloat162_rn(1.0f);
__nv_bfloat162 neg_x = __hneg2(x);
__nv_bfloat162 exp_neg_x = h2exp(neg_x);
__nv_bfloat162 sigmoid_x = __h2div(one, __hadd2(one, exp_neg_x));
__nv_bfloat162 silu_x = __hmul2(x, sigmoid_x);
__nv_bfloat162 result = __hmul2(silu_x, mult);

// Store vectorized result
store_vectorized_data(output_idx, result);
```

## Compilation Flags

Optimized for H100 GPU architecture:
```bash
nvcc -O3 -march=sapphirerapids -funroll-loops -ffast-math \
     -gencode arch=compute_90a,code=sm_90a \
     -ftz=true -prec-div=false \
     silu_kernel.cu -o silu_test
```

## Optimization Techniques Used

1. **Vectorization**: bfloat162 intrinsics for SIMD operations
2. **Memory coalescing**: Aligned vectorized loads/stores where possible  
3. **Register optimization**: Balanced register usage vs parallelism
4. **Cache warming**: L2 cache preloading for consistent measurements
5. **Block size tuning**: Tested multiple configurations
6. **Boundary condition elimination**: Leveraged even tensor dimensions

## Lessons Learned

1. **Memory bandwidth is the primary bottleneck** - but we achieved 84% efficiency (2.82 TB/s)
2. **Vec8 can be competitive** when properly optimized - now matches Vec4 performance  
3. **Block size impact varies by complexity**:
   - Simple kernels (non-vectorized): Prefer larger blocks (512)
   - Complex kernels (Vec4/Vec8): Slight advantage with smaller blocks (128)
4. **Vectorization scales excellently** up to 8 elements per thread
5. **Register pressure is manageable** on modern GPUs like H100
6. **Optimal configuration**: **Vec4 + Block Size 128** = **48.0 μs** (best overall)
7. **Memory access patterns matter** - vectorized loads/stores provide significant gains

## Future Optimization Opportunities

1. **Tensor Core utilization**: Explore MMA instructions for computation
2. **Multi-streaming**: Pipeline multiple kernel launches
3. **Persistent kernels**: Reduce launch overhead
4. **Custom vectorization**: Beyond standard bfloat162 intrinsics
5. **Memory layout optimization**: Row-major vs column-major considerations 