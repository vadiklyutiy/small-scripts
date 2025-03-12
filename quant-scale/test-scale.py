import os
import torch
from torch.utils.cpp_extension import load
from hidet.testing.torch_utils import bench_model

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

# Load the extension
ops = load(
    name="scale_ops",
    sources=["./scale_ops.cpp", "./scale_ops.cu"],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-prec-div=false', '-ftz=true'],
    build_directory="build",
    verbose=True
)


def test_correctness(fn, x, block_size, inverse):
    fn_ref = ops.scale_ref
    out_ref = torch.empty(x.shape, device='cuda', dtype=torch.float8_e4m3fn)
    scales_ref = torch.empty((x.shape[0]), device='cuda', dtype=torch.float32)                 
    fn_ref(out_ref, scales_ref, x, 1024, False)
    
    out = torch.empty(x.shape, device='cuda', dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0]), device='cuda', dtype=torch.float32)
    fn(out, scales, x, block_size, inverse)

    if inverse:
        scales = scales.reciprocal()
    
    assert torch.allclose(scales.to(dtype=torch.bfloat16), scales_ref.to(dtype=torch.bfloat16), atol=0.125, rtol=0.125), f"scales: {scales} scales_ref: {scales_ref}"
    assert torch.allclose(out.to(dtype=torch.bfloat16), out_ref.to(dtype=torch.bfloat16), atol=0.125, rtol=0.125), f"out: {out} out_ref: {out_ref}"


def test_block_size(fn, args):
    for block_size in [32, 64, 128, 256, 512, 1024]:
        for inverse in [False, True]:
            test_correctness(fn, args[-1], block_size, inverse)
            new_args = args + [block_size] + [inverse]
            latency = bench_model(fn, new_args)
            print(f"block_size {block_size} inverse {inverse}: {latency:.6f}")


def test_shape(fn, shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(x.shape, device='cuda', dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0]), device='cuda', dtype=torch.float32) 

    test_block_size(fn, [out, scales, x])


@torch.inference_mode()
def main():
    SHAPES = [(20736, 1280),(20736, 5120),(25920, 1280),(25920, 5120),
              (8192, 3584),(8192, 18944)]
    """
    SHAPES = [(1024, 5120),(2048, 5120),(4096, 5120),(8192, 5120),(2048*5,5120), (2048*6,5120), (2048*7,5120), (2048*8,5120), (2048*9,5120), (2048*10,5120),
              (2048*11,5120),(2048*12,5120),(2048*13,5120),(2048*14,5120),(2048*15,5120),(2048*16,5120),(2048*17,5120),(2048*18,5120),(2048*19,5120),(2048*20,5120),
              (2048*21,5120),(2048*22,5120),(2048*23,5120),(2048*24,5120),(2048*25,5120),(2048*26,5120),(2048*27,5120),(2048*28,5120),(2048*29,5120),(2048*30,5120),
              (2048*31,5120),(2048*32,5120),(2048*33,5120),(2048*34,5120),(2048*35,5120),(2048*36,5120),(2048*37,5120),(2048*38,5120),(2048*39,5120),(2048*40,5120),
              (2048*41,5120),(2048*42,5120),(2048*43,5120),(2048*44,5120),(2048*45,5120),(2048*46,5120),(2048*47,5120),(2048*48,5120),(2048*49,5120),(2048*50,5120),
              (2048*51,5120),(2048*52,5120),(2048*53,5120),(2048*54,5120),(2048*55,5120),(2048*56,5120),(2048*57,5120),(2048*58,5120),(2048*59,5120),(2048*60,5120),
              (2048*61,5120),(2048*62,5120),(2048*63,5120),(2048*64,5120),(2048*65,5120),(2048*66,5120),(2048*67,5120),(2048*68,5120),(2048*69,5120),(2048*70,5120),
              (2048*71,5120),(2048*72,5120)]"
    """
    #SHAPES = [(20736, 256*i) for i in range(4, 32)]
    #SHAPES = [(20736, 5120)]

    for shape in SHAPES:
        print(f"TEST SHAPE: {shape}. SIZE: {2*shape[0]*shape[1]/1024/1024} MB")
        print("TESTING REDUCE BY BLOCK")
        test_shape(ops.scale_ref, shape)
        print("TESTING REDUCE BY WARP")
        test_shape(ops.scale_new, shape)   


if __name__ == "__main__":
    main()