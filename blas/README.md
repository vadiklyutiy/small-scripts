# Matrix Multiplication with cuBLASLt for BF16

This project implements high-performance matrix multiplication for BF16 matrices with mixed layouts using NVIDIA's cuBLASLt library, optimized for NVIDIA H100 GPUs.

## Matrix Specifications
- Matrix A: 8 x 4096 (row major format, bf16 data type)
- Matrix B: 4096 x 28672 (column major format, bf16 data type)
- Result C: 8 x 28672 (row major format, bf16 data type)

## Implementation
The project uses cuBLASLt to perform efficient matrix multiplication while handling mixed matrix layouts:
- cublasLt supports both row-major and column-major formats natively
- BF16 data type is used for reduced memory footprint and higher throughput
- Tensor cores are leveraged for maximum performance
- The implementation handles alpha=1, beta=0 efficiently

## Building the Project

```bash
# Build the benchmark and validation programs
make
```

## Running the Benchmark

```bash
./benchmark_cublaslt
```

The benchmark:
- Performs warmup iterations to prime the GPU
- Runs multiple timed iterations of the matrix multiplication
- Reports statistics including:
  - Min/Max/Avg execution time (in microseconds)
  - Performance in GFLOPS
  - Automatically optimizes for alpha=1, beta=0 case

## Validating Results

```bash
./validate_cublaslt
```

The validation program:
- Compares GPU results with CPU reference implementation
- Verifies correctness of the implementation
- Handles precision differences between CPU and GPU calculations

## Performance

On an NVIDIA H100 GPU, this implementation achieves over 22 TFLOPS of performance, with:
- Average execution time of ~85 microseconds
- Efficient use of GPU memory bandwidth
- Proper utilization of tensor cores for BF16

## Requirements
- CUDA Toolkit 12.0 or later with support for SM 9.0 (H100)
- cuBLASLt library
- C++14 or later compiler with CUDA support 