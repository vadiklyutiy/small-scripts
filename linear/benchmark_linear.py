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
Uses CUDA graphs to capture and replay the 100 iterations
Each benchmark runs 5 times and reports median and std
By default, weight buffers are cloned for each iteration to simulate different memory reads
(can be disabled with use_separate_weight_buffers=False)
"""

import torch
import torch.nn.functional as F
import nvtx
from flashinfer import tgv_gemm_sm100
import statistics


def run_benchmark(linear_func, name, x, weight, bias, num_iterations=100, warmup=10, use_separate_weight_buffers=True):
    """Run benchmark for a given linear function
    
    Args:
        use_separate_weight_buffers: If True, creates separate weight buffer for each iteration in CUDA graph.
                                      If False, uses the same weight buffer for all iterations.
    """
    
    # Warm-up iterations
    for _ in range(warmup):
        _ = linear_func(x, weight, bias)
    
    torch.cuda.synchronize()
    
    if use_separate_weight_buffers:
        # Create multiple weight buffers - one for each iteration
        weight_buffers = []
        for _ in range(num_iterations):
            # Create a new weight buffer with the same data but different memory location
            weight_buffer = weight.clone().detach()
            weight_buffers.append(weight_buffer)
    else:
        # Use the same weight for all iterations
        weight_buffers = [weight] * num_iterations
    
    # Create a static buffer for output to ensure graph can be captured
    static_output = None
    
    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(num_iterations):
            # Use weight buffer for each iteration
            static_output = linear_func(x, weight_buffers[i], bias)
    
    torch.cuda.synchronize()
    
    # Warmup CUDA graph - replay 3 times
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events - replay the graph 5 times
    num_benchmark_runs = 5
    elapsed_times_us = []
    
    for _ in range(num_benchmark_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        graph.replay()
        end_event.record()
        
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times_us.append(elapsed_time_ms * 1000)  # Convert to microseconds
    
    output = static_output
    
    # Clean up weight buffers (only if we created separate ones)
    if use_separate_weight_buffers:
        del weight_buffers
    
    # Calculate median and std of elapsed times
    median_time_us = statistics.median(elapsed_times_us)
    std_time_us = statistics.stdev(elapsed_times_us) if len(elapsed_times_us) > 1 else 0.0
    avg_time_us = median_time_us / num_iterations
    
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
    
    # Calculate std per iteration
    std_time_per_iter_us = std_time_us / num_iterations
    
    return avg_time_us, std_time_per_iter_us, gflops, bandwidth_gb_s


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
    batch_sizes = [2**i for i in range(7)]  # 1, 2, 4, 8, ..., 4096
    
    for batch_size in batch_sizes:
        # Create input tensor for this batch size
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        
        # Run benchmarks
        # Note: Add use_separate_weight_buffers=False to use same weight buffer for all iterations
        time1, std1, gflops1, bw1 = run_benchmark(
            linear_original, 
            "1. Original",
            x, weight, bias
        )
        
        time2, std2, gflops2, bw2 = run_benchmark(
            linear_compiled,
            "2. torch.compile()",
            x, weight, bias
        )
        
        time3, std3, gflops3, bw3 = run_benchmark(
            linear_max_autotune,
            "3. max-autotune ncg",
            x, weight, bias
        )
        
        time4, std4, gflops4, bw4 = run_benchmark(
            linear_tgv,
            "4. TGV GEMM pdl=False",
            x, weight, bias
        )
        
        time5, std5, gflops5, bw5 = run_benchmark(
            linear_tgv_pdl,
            "5. TGV GEMM pdl=True",
            x, weight, bias
        )
        
        # Print summary
        print(f"batch={batch_size}")
        print(f"{'='*110}")
        print("SUMMARY COMPARISON")
        print(f"{'='*110}")
        print(f"{'Variant':<35} {'Median (us)':<15} {'Std (us)':<12} {'GFLOPS':<12} {'BW (GB/s)':<12} {'Speedup'}")
        print(f"{'-'*110}")
        print(f"{'1. Original':<35} {time1:<15.6f} {std1:<12.6f} {gflops1:<12.2f} {bw1:<12.2f} {1.0:.2f}x")
        print(f"{'2. torch.compile()':<35} {time2:<15.6f} {std2:<12.6f} {gflops2:<12.2f} {bw2:<12.2f} {time1/time2:.2f}x")
        print(f"{'3. max-autotune ncg':<35} {time3:<15.6f} {std3:<12.6f} {gflops3:<12.2f} {bw3:<12.2f} {time1/time3:.2f}x")
        print(f"{'4. TGV GEMM pdl=False':<35} {time4:<15.6f} {std4:<12.6f} {gflops4:<12.2f} {bw4:<12.2f} {time1/time4:.2f}x")
        print(f"{'5. TGV GEMM pdl=True':<35} {time5:<15.6f} {std5:<12.6f} {gflops5:<12.2f} {bw5:<12.2f} {time1/time5:.2f}x")
        print()  # Empty line between batch sizes
        
        del x


if __name__ == "__main__":
    benchmark_linear()

