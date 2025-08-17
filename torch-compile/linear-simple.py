import torch
from torch import nn
from hidet.testing.torch_utils import Backend, bench_model
import hidet

hidet.option.debug_show_verbose_flow_graph(True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 1024)

    def forward(self, x):
        y = self.linear1(x)
        y = torch.relu(y)
        y = y + 0.5
        return y
    
def run_model(model, input_size, dtype=torch.bfloat16, device='cuda'):
    input = torch.randn(input_size, dtype=dtype, device=device)
    if input_size[0] > 1:  # Only mark dynamic if batch size > 1
        torch._dynamo.mark_dynamic(input, 0)
    return model(input)


@torch.inference_mode
def main():
    # Initialize backend and model
    backend = Backend(backend='hidet', mode='max-autotune', dtype=torch.bfloat16, cache='111')
    model = Model().cuda().to(torch.bfloat16).eval()
    
    compiled_model = backend.compile(model)
    eager_model = model

    
    # Warmup phase
    input_size = (2, 4096)
    print("Running warmup...")
    run_model(compiled_model, input_size)
    run_model(eager_model, input_size)
    
    # Verify correctness with allclose comparison
    print("="*50)
    print("Verifying compiled model (inductor) against eager mode")
    print("="*50)
    
    # Generate consistent input for both models
    input = torch.randn(input_size, dtype=torch.bfloat16, device='cuda')
    
    # Get outputs from both models
    with torch.no_grad():
        compiled_output = compiled_model(input)
        eager_output = eager_model(input)
    
    # Check if outputs are numerically close
    is_close = torch.allclose(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
    
    print(f"Outputs match: {is_close}")
    max_diff = torch.max(torch.abs(compiled_output - eager_output))
    print(f"Maximum absolute difference: {max_diff:.6f}")

if __name__ == "__main__":
    main()