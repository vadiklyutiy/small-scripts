"""
Combined performance benchmark comparing:
- FP16 torch.nn.functional.linear
- FP8 gemm_fp8_nt_groupwise (cutlass backend)
- FP8 gemm_fp8_nt_groupwise (trtllm backend)

Tests various shapes and batch sizes with unified reporting.
"""

import os
import sys
import torch
import numpy as np

# Set environment variables
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

from flashinfer.testing.utils import bench_gpu_time_with_cupti, quantize_fp8

# Import FlashInfer after path setup
sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')
from flashinfer.gemm import gemm_fp8_nt_groupwise

# Import DeepGEMM
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8


def is_sm100() -> bool:
    """Check if running on SM100 (Blackwell B200) GPU."""
    capability = torch.cuda.get_device_capability()
    return capability[0] == 10


def get_gpu_info() -> str:
    """Get GPU name and compute capability."""
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability()
    return f"{gpu_name} (SM{capability[0]}{capability[1]})"


def create_fp8_tensors(m, n, k, scale_major_mode="MN"):
    """Create FP8 tensors with proper scaling for gemm_fp8_nt_groupwise."""
    block_size = 128
    
    # Create input tensors in bfloat16
    a_bf16 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    
    a_scale_shape = (k // block_size, m)
    a_tile_shape = (1, block_size)
    
    b_scale_shape = (k // block_size, n // block_size)
    b_tile_shape = (block_size, block_size)
    
    a_fp8, a_scale = quantize_fp8(a_bf16, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_bf16, b_scale_shape, b_tile_shape, scale_major_mode)
    
    # Create output tensor
    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    
    return a_fp8, b_fp8, a_scale, b_scale, out


def create_vllm_blockwise_fp8_tensors(m, n, k):
    """Create FP8 tensors with vLLM-style blockwise quantization:
    - Activation: grouped quantization with group size 128 -> scales shape [m, k/128]
    - Weight: blockwise 128x128 quantization -> scales shape [k/128, n/128]
    """
    from vllm._custom_ops import scaled_fp8_quant
    
    # Create input tensors
    a_bf16 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    # Create weight as [n, k] (like PyTorch Linear layer weight)
    b_bf16_nk = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    
    # Quantize activations with grouped scaling (group size 128)
    block_size = 128
    assert k % block_size == 0
    
    # Reshape to [m * k/128, 128] so each row is one group
    a_reshaped = a_bf16.view(m * (k // block_size), block_size)
    
    # Quantize with per-token (per-row) quantization -> one scale per group
    a_fp8_flat, a_scale_flat = scaled_fp8_quant(
        a_reshaped,
        use_per_token_if_dynamic=True
    )
    
    # Reshape back to original dimensions
    a_fp8 = a_fp8_flat.view(m, k)
    a_scale = a_scale_flat.view(m, k // block_size)
    
    # Quantize weights with blockwise 128x128 scaling
    block_size = 128
    assert k % block_size == 0 and n % block_size == 0
    
    # Reshape for blockwise quantization on [n, k] tensor
    b_reshaped = b_bf16_nk.view(n // block_size, block_size, k // block_size, block_size)
    b_blocks = b_reshaped.permute(0, 2, 1, 3).contiguous()
    b_blocks_flat = b_blocks.view(-1, block_size * block_size)
    
    # Quantize each block
    b_fp8_blocks = []
    b_scales_blocks = []
    for i in range(b_blocks_flat.shape[0]):
        block = b_blocks_flat[i].view(block_size, block_size)
        block_fp8, block_scale = scaled_fp8_quant(block)
        b_fp8_blocks.append(block_fp8)
        b_scales_blocks.append(block_scale)
    
    # Reconstruct blockwise quantized weight [n, k]
    b_fp8_blocks_tensor = torch.stack(b_fp8_blocks).view(
        n // block_size, k // block_size, block_size, block_size
    )
    b_fp8_nk = b_fp8_blocks_tensor.permute(0, 2, 1, 3).contiguous().view(n, k)
    b_scale_nk = torch.stack(b_scales_blocks).view(n // block_size, k // block_size)
    
    # Transpose to [k, n] for cutlass_scaled_mm (this creates column-major layout)
    b_fp8 = b_fp8_nk.t()
    b_scale = b_scale_nk.t()
    
    return a_fp8, b_fp8, a_scale, b_scale  # b is [k, n] transposed for cutlass_scaled_mm


def benchmark_fp16_linear(batch_size, out_features, in_features):
    """Benchmark FP16 torch.nn.functional.linear."""
    device = 'cuda'
    dtype = torch.float16
    
    # Create tensors
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)
    input_tensor = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.linear(input_tensor, weight, bias)
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return torch.nn.functional.linear(input_tensor, weight, bias)
    
    # Run benchmark with CUPTI using CUDA graphs
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        cold_l2_cache=True,
        use_cuda_graph=True,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * batch_size * in_features * out_features + batch_size * out_features
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def benchmark_fp8_gemm(batch_size, out_features, in_features, backend='cutlass'):
    """Benchmark FP8 gemm_fp8_nt_groupwise."""
    m, n, k = batch_size, out_features, in_features
    scale_major_mode = 'MN'
    
    # Create tensors
    a_fp8, b_fp8, a_scale, b_scale, out = create_fp8_tensors(m, n, k, scale_major_mode)
    
    # Warmup
    for _ in range(5):
        gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale,
            scale_major_mode=scale_major_mode,
            mma_sm=1,
            out=out,
            backend=backend,
        )
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale,
            scale_major_mode=scale_major_mode,
            mma_sm=1,
            out=out,
            backend=backend,
        )
    
    # Run benchmark with CUPTI using CUDA graphs
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        cold_l2_cache=True,
        use_cuda_graph=True,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * m * n * k
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def benchmark_deepgemm(batch_size, out_features, in_features):
    """Benchmark DeepGEMM FP8 fp8_gemm_nt."""
    m, n, k = batch_size, out_features, in_features
    
    # Create tensors in bfloat16 for conversion
    a_bf16 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    
    # Convert to FP8 using DeepGEMM utilities
    # Use UE8M0 for SM100 (Blackwell B200), regular FP32 scales for SM90 (Hopper)
    use_ue8m0 = is_sm100()
    a_fp8, a_scale = per_token_cast_to_fp8(a_bf16, use_ue8m0=use_ue8m0)
    b_fp8, b_scale = per_block_cast_to_fp8(b_bf16, use_ue8m0=use_ue8m0)
    
    # Create output tensor
    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    
    # AGGRESSIVE Warmup - DeepGEMM JIT compiles on first call
    # Do many more warmup iterations to ensure compilation is complete
    for _ in range(50):
        deep_gemm.fp8_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        deep_gemm.fp8_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
        return out
    
    # Run benchmark with CUPTI using CUDA graphs to eliminate launch overhead
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        cold_l2_cache=True,
        use_cuda_graph=True,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * m * n * k
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def benchmark_vllm_cutlass(batch_size, out_features, in_features):
    """Benchmark vLLM's cutlass_scaled_mm with blockwise quantization."""
    from vllm._custom_ops import cutlass_scaled_mm
    
    m, n, k = batch_size, out_features, in_features
    
    # Create tensors with vLLM-style quantization
    a_fp8, b_fp8, a_scale, b_scale = create_vllm_blockwise_fp8_tensors(m, n, k)
    
    # Warmup
    for _ in range(5):
        _ = cutlass_scaled_mm(a_fp8, b_fp8, a_scale, b_scale, torch.bfloat16)
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return cutlass_scaled_mm(a_fp8, b_fp8, a_scale, b_scale, torch.bfloat16)
    
    # Run benchmark with CUPTI using CUDA graphs
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        cold_l2_cache=True,
        use_cuda_graph=True,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * m * n * k
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def run_combined_benchmark():
    """Run combined benchmark for all configurations."""
    
    # Print GPU information
    gpu_info = get_gpu_info()
    sm100 = is_sm100()
    
    print("=" * 200)
    print("Combined Performance Benchmark: FP8 GEMM vs FP16 Linear (vLLM as baseline)")
    print("=" * 200)
    print(f"GPU: {gpu_info}")
    print(f"SM100 Mode: {'ENABLED (UE8M0 scales)' if sm100 else 'DISABLED (FP32 scales)'}")
    print()
    print("Configurations:")
    print("  - FP8 vLLM (BASELINE): cutlass_scaled_mm with grouped activations (group=128) + 128x128 blockwise weights")
    print("  - FP8 FI Cutlass: gemm_fp8_nt_groupwise (cutlass backend)")
    print("  - FP8 FI TRT-LLM: gemm_fp8_nt_groupwise (trtllm backend)")
    print(f"  - FP8 DeepGEMM: deep_gemm.fp8_gemm_nt {'with UE8M0 scales' if sm100 else 'with FP32 scales'}")
    print("  - FP16: torch.nn.functional.linear")
    print("  - ALL methods: CUPTI with CUDA Graphs and cold L2 cache")
    print("=" * 200)
    print()
    
    # Test configurations
    shapes = [
        [2048, 1024],
        [2048, 128],
        [256, 2048],
        [2560, 2048],
        [3072, 2048],
    ]
    
    batch_sizes = [128, 256, 512, 1024]
    
    results = []
    
    for out_features, in_features in shapes:
        for batch_size in batch_sizes:
            print(f"\nTesting: Batch={batch_size}, Out={out_features}, In={in_features}")
            
            result = {
                'batch': batch_size,
                'out': out_features,
                'in': in_features,
            }
            
            # Benchmark FP16
            try:
                print("  Running FP16 linear...")
                fp16_result = benchmark_fp16_linear(batch_size, out_features, in_features)
                result['fp16_median'] = fp16_result['median_us']
                result['fp16_std_pct'] = fp16_result['std_pct']
                result['fp16_tflops'] = fp16_result['tflops_per_sec']
            except Exception as e:
                print(f"  FP16 ERROR: {e}")
                result['fp16_median'] = None
                result['fp16_std_pct'] = None
                result['fp16_tflops'] = None
            
            # Benchmark vLLM cutlass_scaled_mm (BASELINE)
            try:
                print("  Running vLLM cutlass...")
                vllm_result = benchmark_vllm_cutlass(batch_size, out_features, in_features)
                result['vllm_median'] = vllm_result['median_us']
                result['vllm_std_pct'] = vllm_result['std_pct']
                result['vllm_tflops'] = vllm_result['tflops_per_sec']
            except Exception as e:
                print(f"  vLLM ERROR: {e}")
                result['vllm_median'] = None
                result['vllm_std_pct'] = None
                result['vllm_tflops'] = None
            
            # Benchmark FP8 Cutlass
            try:
                print("  Running FP8 Cutlass...")
                cutlass_result = benchmark_fp8_gemm(batch_size, out_features, in_features, backend='cutlass')
                result['cutlass_median'] = cutlass_result['median_us']
                result['cutlass_std_pct'] = cutlass_result['std_pct']
                result['cutlass_tflops'] = cutlass_result['tflops_per_sec']
                
                # Calculate ratio relative to vLLM
                if result['vllm_median'] is not None:
                    result['cutlass_ratio'] = result['vllm_median'] / result['cutlass_median']
                else:
                    result['cutlass_ratio'] = None
            except Exception as e:
                print(f"  Cutlass ERROR: {e}")
                result['cutlass_median'] = None
                result['cutlass_std_pct'] = None
                result['cutlass_tflops'] = None
                result['cutlass_ratio'] = None
            
            # Benchmark FP8 TRT-LLM
            try:
                print("  Running FP8 TRT-LLM...")
                trtllm_result = benchmark_fp8_gemm(batch_size, out_features, in_features, backend='trtllm')
                result['trtllm_median'] = trtllm_result['median_us']
                result['trtllm_std_pct'] = trtllm_result['std_pct']
                result['trtllm_tflops'] = trtllm_result['tflops_per_sec']
                
                # Calculate ratio relative to vLLM
                if result['vllm_median'] is not None:
                    result['trtllm_ratio'] = result['vllm_median'] / result['trtllm_median']
                else:
                    result['trtllm_ratio'] = None
            except Exception as e:
                print(f"  TRT-LLM ERROR: {e}")
                result['trtllm_median'] = None
                result['trtllm_std_pct'] = None
                result['trtllm_tflops'] = None
                result['trtllm_ratio'] = None
            
            # Benchmark DeepGEMM
            try:
                print("  Running DeepGEMM...")
                deepgemm_result = benchmark_deepgemm(batch_size, out_features, in_features)
                result['deepgemm_median'] = deepgemm_result['median_us']
                result['deepgemm_std_pct'] = deepgemm_result['std_pct']
                result['deepgemm_tflops'] = deepgemm_result['tflops_per_sec']
                
                # Calculate ratio relative to vLLM
                if result['vllm_median'] is not None:
                    result['deepgemm_ratio'] = result['vllm_median'] / result['deepgemm_median']
                else:
                    result['deepgemm_ratio'] = None
            except Exception as e:
                print(f"  DeepGEMM ERROR: {e}")
                result['deepgemm_median'] = None
                result['deepgemm_std_pct'] = None
                result['deepgemm_tflops'] = None
                result['deepgemm_ratio'] = None
            
            # Benchmark FP16 (for comparison)
            # Calculate ratio relative to vLLM
            if result['fp16_median'] is not None and result['vllm_median'] is not None:
                result['fp16_ratio'] = result['vllm_median'] / result['fp16_median']
            else:
                result['fp16_ratio'] = None
            
            results.append(result)
    
    # Print summary table
    print("\n" + "=" * 200)
    print("SUMMARY TABLE (vLLM FP8 as baseline)")
    print("=" * 200)
    # First header line with column categories - must align "|" with data rows
    # Data format: {5} {5} {6} | {10} {7} | {10} {7} {14} | {10} {7} {14} | {10} {7} {14} | {10} {7} {14}
    # Section widths after "|": 18 | 33 | 33 | 33 | 33
    print(f"{'Out':>5} {'In':>5} {'Batch':>6} | "
          f"{'FP8 vLLM':^18} | "
          f"{'FP8 Cutlass':^33} | "
          f"{'FP8 TRT-LLM':^33} | "
          f"{'FP8 DeepGEMM':^33} | "
          f"{'FP16':^33}")
    # Second header line with specific metrics - must match exact spacing of data rows
    print(f"{'':>5} {'':>5} {'':>6} | "
          f"{'Median(us)':>10} {'Std(%)':>7} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to vLLM':>14} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to vLLM':>14} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to vLLM':>14} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to vLLM':>14}")
    print("-" * 200)
    
    for r in results:
        # vLLM values (baseline)
        vllm_med = f"{r['vllm_median']:10.2f}" if r['vllm_median'] is not None else "         -"
        vllm_std = f"{r['vllm_std_pct']:7.2f}" if r['vllm_std_pct'] is not None else "      -"
        
        # Cutlass values
        cutlass_med = f"{r['cutlass_median']:10.2f}" if r['cutlass_median'] is not None else "         -"
        cutlass_std = f"{r['cutlass_std_pct']:7.2f}" if r['cutlass_std_pct'] is not None else "      -"
        cutlass_ratio = f"{r['cutlass_ratio']:14.2f}" if r['cutlass_ratio'] is not None else "             -"
        
        # TRT-LLM values
        trtllm_med = f"{r['trtllm_median']:10.2f}" if r['trtllm_median'] is not None else "         -"
        trtllm_std = f"{r['trtllm_std_pct']:7.2f}" if r['trtllm_std_pct'] is not None else "      -"
        trtllm_ratio = f"{r['trtllm_ratio']:14.2f}" if r['trtllm_ratio'] is not None else "             -"
        
        # DeepGEMM values
        deepgemm_med = f"{r['deepgemm_median']:10.2f}" if r['deepgemm_median'] is not None else "         -"
        deepgemm_std = f"{r['deepgemm_std_pct']:7.2f}" if r['deepgemm_std_pct'] is not None else "      -"
        deepgemm_ratio = f"{r['deepgemm_ratio']:14.2f}" if r['deepgemm_ratio'] is not None else "             -"
        
        # FP16 values
        fp16_med = f"{r['fp16_median']:10.2f}" if r['fp16_median'] is not None else "         -"
        fp16_std = f"{r['fp16_std_pct']:7.2f}" if r['fp16_std_pct'] is not None else "      -"
        fp16_ratio = f"{r['fp16_ratio']:14.2f}" if r['fp16_ratio'] is not None else "             -"
        
        print(f"{r['out']:5d} {r['in']:5d} {r['batch']:6d} | "
              f"{vllm_med} {vllm_std} | "
              f"{cutlass_med} {cutlass_std} {cutlass_ratio} | "
              f"{trtllm_med} {trtllm_std} {trtllm_ratio} | "
              f"{deepgemm_med} {deepgemm_std} {deepgemm_ratio} | "
              f"{fp16_med} {fp16_std} {fp16_ratio}")
    
    print("=" * 200)
    print("Notes:")
    print("  - Median: Median execution time in microseconds")
    print("  - Std(%): Standard deviation as percentage of median")
    print("  - Ratio to vLLM: vLLM time / method time (>1.0 means faster than vLLM baseline, <1.0 means slower)")
    print(f"  - DeepGEMM uses {'UE8M0 (power-of-2) scales' if sm100 else 'FP32 scales'} on {gpu_info}")
    print("  - vLLM (baseline) uses grouped activation quantization (group=128) + 128x128 blockwise weight quantization")
    print("  - '-' indicates unsupported configuration")
    print("=" * 200)
    
    return results


if __name__ == "__main__":
    results = run_combined_benchmark()