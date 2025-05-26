# CUDA Reduce Mean Kernels

High-performance CUDA kernels for computing row-wise mean reduction on a 2048×2048 matrix, producing a 2048×1 output vector.

## Implementation Overview

This project implements **18 different CUDA kernels** using two approaches:

### Hand-Optimized Shuffle Kernels (12 kernels)
- **Thread Configurations**: 32, 64, 128, 256, 512, 1024 threads per block
- **Synchronization Strategies**:
  - **Atomic**: Inter-warp reduction via `atomicAdd(&shared_sum, rv)`
  - **Separate**: Individual `warp_sums[warp_id]` locations per warp
- **Optimization**: Warp-level shuffle instructions for maximum performance

### CUB Library Kernels (6 kernels)
- **Thread Configurations**: 32, 64, 128, 256, 512, 1024 threads per block
- **Implementation**: NVIDIA's CUB (CUDA Unbound) BlockReduce primitives
- **Advantage**: Highly optimized library implementation

## Performance Results

**Test Environment**: H100 80GB, CUDA 12.8, Matrix size: 2048×2048

```
         Kernel Name   Threads    Type        Time    Status
------------------------------------------------------------
           32_atomic        32 Shuffle        5.29 μs      PASS
         32_separate        32 Shuffle        5.28 μs      PASS
           64_atomic        64 Shuffle        5.38 μs      PASS
         64_separate        64 Shuffle        5.26 μs      PASS
          128_atomic       128 Shuffle        4.99 μs      PASS
        128_separate       128 Shuffle        4.94 μs      PASS  ← Best Hand-Optimized
          256_atomic       256 Shuffle        5.26 μs      PASS
        256_separate       256 Shuffle        4.98 μs      PASS
          512_atomic       512 Shuffle        7.55 μs      PASS
        512_separate       512 Shuffle        5.40 μs      PASS
         1024_atomic      1024 Shuffle       16.01 μs      PASS
       1024_separate      1024 Shuffle        8.37 μs      PASS
              32_cub        32     CUB        5.29 μs      PASS
              64_cub        64     CUB        5.26 μs      PASS
             128_cub       128     CUB        4.97 μs      PASS
             256_cub       256     CUB        4.83 μs      PASS  ← Best Overall
             512_cub       512     CUB        5.56 μs      PASS
            1024_cub      1024     CUB        8.82 μs      PASS
```

### Key Performance Insights

- **🏆 Best Overall**: `256_cub` - **4.83 μs**
- **🏆 Best Hand-Optimized**: `128_separate` - **4.94 μs**
- **📈 Performance Range**: 4.83 μs to 16.01 μs (3.3x difference)
- **📊 Average Time**: 6.30 μs across all 18 kernels
- **🎯 CUB vs Hand-Optimized**: CUB wins by only **0.11 μs** (2.2% advantage)

### Atomic vs Separate Strategy Analysis

| Threads | Atomic | Separate | Speedup |
|---------|--------|----------|---------|
| 32-256  | ~5.0 μs | ~5.0 μs | Minimal difference |
| 512     | 7.55 μs | 5.40 μs | **1.40x faster** |
| 1024    | 16.01 μs | 8.37 μs | **1.91x faster** |

**Conclusion**: Separate strategy significantly outperforms atomic for high thread counts due to reduced contention.

## Files

- `reduce_kernels.cu` - 12 hand-optimized shuffle kernels
- `reduce_kernels_cub.cu` - 6 CUB library kernels  
- `test_reduce.cu` - Comprehensive test program
- `Makefile` - Build configuration with H100 optimizations
- `run_tests.sh` - Automated test runner with multiple modes

## Building and Running

### Prerequisites
- NVIDIA H100 GPU (compute capability 9.0+)
- CUDA Toolkit 12.8+
- GCC/G++ compiler

### Quick Start
```bash
# Build and run
make run

# Clean build
make clean && make

# Run automated tests
./run_tests.sh
```

### Build Targets
```bash
make           # Build main test program
make clean     # Remove build artifacts
make run       # Build and execute test
make help      # Show all targets
```

## Implementation Details

### Hand-Optimized Kernels
All kernels use optimized warp shuffle instructions:
```cuda
// Immediate warp-level reduction
int32_t mask = __activemask();
rv += __shfl_down_sync(mask, rv, 16, 32);
rv += __shfl_down_sync(mask, rv, 8, 32);
rv += __shfl_down_sync(mask, rv, 4, 32);
rv += __shfl_down_sync(mask, rv, 2, 32);
rv += __shfl_down_sync(mask, rv, 1, 32);
rv = __shfl_sync(mask, rv, 0, 32);
```

### Memory Access Pattern
Strided access for coalesced memory reads:
- Thread `tid` processes elements: `tid, tid + block_size, tid + 2*block_size, ...`
- Ensures optimal memory bandwidth utilization

### Compilation Optimizations
```bash
-march=sapphirerapids    # Intel Sapphire Rapids optimizations
-funroll-loops          # Aggressive loop unrolling
-ffast-math            # Fast math optimizations
-ftz=true              # Flush-to-zero for denormals
-gencode arch=compute_90a,code=sm_90a  # H100-specific optimizations
```

## Architecture Notes

- **Grid Configuration**: Always 2048 blocks (one per row)
- **Block Sizes**: Variable (32-1024 threads)
- **Data Type**: float32
- **Optimized for**: NVIDIA H100 Hopper architecture
- **Memory Pattern**: Coalesced global memory access
- **Synchronization**: Warp-level primitives for maximum efficiency

## Conclusion

This implementation demonstrates that:

1. **CUB library achieves near-optimal performance** (4.83 μs)
2. **Hand-optimized kernels are highly competitive** (4.94 μs, only 2.2% slower)
3. **Separate memory strategy scales better** than atomic for high thread counts
4. **Warp shuffle instructions are crucial** for high-performance reductions
5. **Proper memory access patterns** enable excellent bandwidth utilization

Both approaches achieve excellent performance on modern hardware, with the choice depending on development time vs. fine-grained control requirements. 