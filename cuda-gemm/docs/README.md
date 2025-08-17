# CUDA GEMM Implementations

This repository contains various CUDA implementations of General Matrix Multiplication (GEMM) operations, optimized for different GPU architectures and performance characteristics.

## Implementations Overview

### 1. Basic GEMM (`basic-gemm/`)

Simple implementations of matrix multiplication in CUDA:

- `gemm_basic.cu`: Simple row/column-based implementation
- `gemm_shared_mem.cu`: Basic shared memory tiling implementation

These implementations are good starting points for understanding GEMM operations on GPUs, but are not optimized for performance.

### 2. TMA GEMM (`tma-gemm/`)

Implementations related to Tensor Memory Accelerator (TMA) operations:

- `src/gemm_tma.cu`: Main implementation using tiled approach for matrix multiplication
- `tests/test_tma_gemm.cu`: Test driver for the TMA implementation
- Uses shared memory optimizations and tiling for improved performance

### 3. Tensor Core GEMM (`tensor-core-gemm/`)

Implementation that leverages NVIDIA Tensor Cores:

- `src/gemm_tensor_cores.cu`: Uses the WMMA (Warp Matrix Multiply Accumulate) API to offload matrix operations to Tensor Cores
- Designed for high performance on GPUs with Tensor Core support (Volta, Turing, Ampere, Hopper architectures)

## Performance Characteristics

The implementations are ordered from simplest to most complex/performant:

1. Basic GEMM: Simple, but least efficient
2. TMA GEMM: Better performance with tiling and shared memory optimization
3. Tensor Core GEMM: Highest performance on supported hardware, leveraging specialized matrix operations

## Compilation and Usage

Each implementation directory contains its own build files. Generally, you can build them using:

```bash
cd [implementation-directory]
./build.sh
```

## Notes

- The implementations assume BF16 data type
- Matrix A is in row-major format, Matrix B is in column-major format
- The test cases use large matrices (8 x 4096 * 4096 x 28672 = 8 x 28672) to demonstrate performance characteristics 