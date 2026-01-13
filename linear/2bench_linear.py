#!/usr/bin/env python3
"""
Benchmark for torch.nn.functional.linear on NVIDIA GPU
Compares five variants:
1. Original torch.nn.functional.linear
2. With torch.compile
3. With torch.compile(mode="max-autotune-no-cudagraphs")
4. TGV GEMM (FlashInfer SM100)
5. TGV GEMM (FlashInfer SM100) with pdl=True

Uses bfloat16 precision
Measures performance using CUDA events with 100 repetitions
Uses CUDA graphs to capture and replay the 100 iterations for accurate timing
"""

import torch
import torch.nn.functional as F
import nvtx
from flashinfer import tgv_gemm_sm100


def run_benchmark(linear_func, name, x, weight, bias, num_iterations=100, warmup=10, use_cuda_graph=True, verbose=False):
    """Run benchmark for a given linear function"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")
    
    # Warm-up iterations
    if verbose:
        print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = linear_func(x, weight, bias)
    
    torch.cuda.synchronize()
    
    if use_cuda_graph:
        # Capture CUDA graph for the iterations
        if verbose:
            print(f"Capturing CUDA graph for {num_iterations} iterations...")
        
        # Create a static buffer for output to ensure graph can be captured
        static_output = None
        
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(num_iterations):
                static_output = linear_func(x, weight, bias)
        
        torch.cuda.synchronize()
        if verbose:
            print(f"Replaying CUDA graph...")
        
        # Benchmark with CUDA events - replay the graph
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        graph.replay()
        end_event.record()
        
        torch.cuda.synchronize()
        output = static_output
        
    else:
        # Benchmark with CUDA events (without CUDA graph)
        if verbose:
            print(f"Running {num_iterations} iterations...")
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start event
        start_event.record()
        
        # Run iterations
        for _ in range(num_iterations):
            output = linear_func(x, weight, bias)
        
        # Record end event
        end_event.record()
        
        # Wait for all operations to complete
        torch.cuda.synchronize()
    
    # Calculate elapsed time
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_us = elapsed_time_ms * 1000  # Convert to microseconds
    avg_time_us = elapsed_time_us / num_iterations
    
    # Calculate FLOPs
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = weight.shape[0]
    flops_per_iteration = 2 * batch_size * out_features * in_features
    gflops = (flops_per_iteration / (avg_time_us * 1e-6)) / 1e9
    
    # Calculate Memory Bandwidth
    # Bytes read/written per iteration:
    # - Read x: batch_size * in_features * dtype_size
    # - Read weight: out_features * in_features * dtype_size
    # - Write output: batch_size * out_features * dtype_size
    # - Read bias (if present): out_features * dtype_size
    dtype_size = x.element_size()  # bytes per element
    bytes_read_x = batch_size * in_features * dtype_size
    bytes_read_weight = out_features * in_features * dtype_size
    bytes_write_output = batch_size * out_features * dtype_size
    bytes_read_bias = out_features * dtype_size if bias is not None else 0
    
    total_bytes = bytes_read_x + bytes_read_weight + bytes_write_output + bytes_read_bias
    bandwidth_gb_s = (total_bytes / (avg_time_us * 1e-6)) / 1e9
    
    # Print results
    if verbose:
        print(f"\nResults:")
        print(f"  CUDA Graph: {'Yes' if use_cuda_graph else 'No'}")
        print(f"  Total time for {num_iterations} iterations: {elapsed_time_us:.4f} us")
        print(f"  Average time per iteration: {avg_time_us:.6f} us")
        print(f"  Throughput: {1e6 / avg_time_us:.2f} iterations/second")
        print(f"\nPerformance:")
        print(f"  FLOPs per iteration: {flops_per_iteration:,}")
        print(f"  GFLOPS: {gflops:.2f}")
        print(f"  Memory per iteration: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
        print(f"  Memory Bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"  Output shape: {output.shape}")
    
    return avg_time_us, gflops, bandwidth_gb_s


def benchmark_linear():
    """Benchmark torch.nn.functional.linear with different batch sizes"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires an NVIDIA GPU.")
        return
    
    device = torch.device("cuda")
    in_features = 4096
    out_features = 2048
    dtype = torch.bfloat16
    
    # Create weight matrix (shared across all batch sizes)
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias = None
    
    # Define functions
    def linear_original(x, weight, bias):
        return F.linear(x, weight, bias)
    
    # Compile functions once
    linear_compiled = torch.compile(linear_original)
    linear_max_autotune = torch.compile(linear_original, mode="max-autotune-no-cudagraphs")
    
    # Prepare TGV GEMM weight
    weight_tgv = weight.clone().contiguous().t()
    bias_tgv = None
    
    def linear_tgv(x, weight, bias):
        return tgv_gemm_sm100(x, weight_tgv, bias_tgv, pdl=False)
    
    def linear_tgv_pdl(x, weight, bias):
        return tgv_gemm_sm100(x, weight_tgv, bias_tgv, pdl=True)
    
    # Initial warmup
    x_warmup = torch.randn(8192, in_features, device=device, dtype=dtype)
    torch._dynamo.mark_dynamic(x_warmup, 0)

    for func in [linear_original, linear_compiled, linear_max_autotune, linear_tgv, linear_tgv_pdl]:
        _ = func(x_warmup, weight, bias)
    torch.cuda.synchronize()
    del x_warmup
    
    # Batch sizes to test (powers of 2 from 1 to 4096)
    batch_sizes = [2**i for i in range(6)]  # 1, 2, 4, 8, ..., 4096
    
    for batch_size in batch_sizes:
        # Create input tensor for this batch size
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        
        # Run benchmarks silently
        time1, gflops1, bw1 = run_benchmark(
            linear_original, 
            "1. Original",
            x, weight, bias,
            verbose=False
        )
        
        time2, gflops2, bw2 = run_benchmark(
            linear_compiled,
            "2. torch.compile()",
            x, weight, bias,
            verbose=False
        )
        
        time3, gflops3, bw3 = run_benchmark(
            linear_max_autotune,
            "3. max-autotune ncg",
            x, weight, bias,
            verbose=False
        )
        
        time4, gflops4, bw4 = run_benchmark(
            linear_tgv,
            "4. TGV GEMM pdl=False",
            x, weight, bias,
            verbose=False
        )
        
        time5, gflops5, bw5 = run_benchmark(
            linear_tgv_pdl,
            "5. TGV GEMM pdl=True",
            x, weight, bias,
            verbose=False
        )
        
        # Print summary
        print(f"batch={batch_size}")
        print(f"{'='*90}")
        print("SUMMARY COMPARISON")
        print(f"{'='*90}")
        print(f"{'Variant':<35} {'Time (us)':<15} {'GFLOPS':<12} {'BW (GB/s)':<12} {'Speedup'}")
        print(f"{'-'*90}")
        print(f"{'1. Original':<35} {time1:<15.6f} {gflops1:<12.2f} {bw1:<12.2f} {1.0:.2f}x")
        print(f"{'2. torch.compile()':<35} {time2:<15.6f} {gflops2:<12.2f} {bw2:<12.2f} {time1/time2:.2f}x")
        print(f"{'3. max-autotune ncg':<35} {time3:<15.6f} {gflops3:<12.2f} {bw3:<12.2f} {time1/time3:.2f}x")
        print(f"{'4. TGV GEMM pdl=False':<35} {time4:<15.6f} {gflops4:<12.2f} {bw4:<12.2f} {time1/time4:.2f}x")
        print(f"{'5. TGV GEMM pdl=True':<35} {time5:<15.6f} {gflops5:<12.2f} {bw5:<12.2f} {time1/time5:.2f}x")
        print()  # Empty line between batch sizes
        
        del x


if __name__ == "__main__":
    benchmark_linear()