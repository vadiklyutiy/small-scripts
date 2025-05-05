import timeit
import torch
import numpy as np
import numba

def mrope_get_next_input_positions_tensor(
    mrope_position_delta: int,
    context_len: int,
    seq_len: int,
) -> torch.Tensor:
    return torch.arange(
        mrope_position_delta + context_len,
        mrope_position_delta + seq_len,
    ).expand(3, -1)


@numba.jit(nopython=True)
def mrope_assign_next_input_positions(
    out: np.ndarray,
    out_offset: int,
    mrope_position_delta: int,
    context_len: int,
    num_new_tokens: int,
):
    for dim in range(3):
        for idx in range(num_new_tokens):
            out[dim,
                out_offset + idx] = mrope_position_delta + context_len + idx


out = torch.empty(3, 1000, dtype=torch.int64)
out_np = out.numpy() # they share the underlying data

def run_torch():
    out_offset = 5
    mrope_position_delta = 100
    context_len = 20
    seq_len = 21

    positions = mrope_get_next_input_positions_tensor(mrope_position_delta, context_len, seq_len)
    out[:, out_offset:out_offset + (seq_len - context_len)] = positions

def run_np():
    out_offset = 5
    mrope_position_delta = 100
    context_len = 20
    seq_len = 21
    
    mrope_assign_next_input_positions(out_np, out_offset, mrope_position_delta, context_len, seq_len - context_len)

# warmup
run_torch()
run_np()

r1 = timeit.timeit(run_torch, number=200000)
print(f"run_torch: {r1:.3f}s")

r2 = timeit.timeit(run_np, number=200000)
print(f"run_np: {r2:.3f}s")
