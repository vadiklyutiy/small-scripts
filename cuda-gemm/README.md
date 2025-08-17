# CUDA GEMM Implementations

This repository contains various CUDA implementations of General Matrix Multiplication (GEMM) operations, optimized for different GPU architectures.

## Directory Structure

- `basic-gemm/`: Simple GEMM implementations with basic optimizations
- `tma-gemm/`: Implementations using tiled matrix approach with shared memory
- `tensor-core-gemm/`: Implementations leveraging Tensor Cores via WMMA API
- `docs/`: Detailed documentation and performance analysis

## Getting Started

For detailed documentation, see [docs/README.md](docs/README.md).

## Performance Summary

| Implementation | Execution Time (ms) | GFLOPS | Optimization Level |
|----------------|---------------------|--------|-------------------|
| Basic GEMM     | ~300                | ~6     | Low               |
| Tiled GEMM     | ~17                 | ~108   | Medium            |
| Tensor Core    | ~1.85 (potential)   | ~1000+ | High              |

## Requirements

- CUDA Toolkit 11.0+
- GPU with Compute Capability 7.0+ for Tensor Core implementations
- CMake 3.10+ 