#include <torch/extension.h>

void scale_ref( torch::Tensor& out,        
                torch::Tensor& scales, 
                torch::Tensor const& input, 
                int block_size); 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_ref", &scale_ref, "Custom CUDA op");
}
