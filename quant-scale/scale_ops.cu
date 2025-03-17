#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>


using FP8_TYPE = c10::Float8_e4m3fn;
using BFLOAT16 = c10::BFloat16;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX =
    std::numeric_limits<FP8_TYPE>::max();

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
    static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);
    quant_type_t x;
    quant_type_t y;
    quant_type_t z;
    quant_type_t w;
};

template <bool is_scale_inverted>
__device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(FP8_TYPE* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          float const scale,
                                          int64_t const num_elems,
                                          int const tid, int const step) {
  using float8x4_t = q8x4_t<FP8_TYPE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<vec4_t<scalar_t> const*>(input);
  auto* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(input[i]), scale);
  }
}


__device__ float thread_max_vec(BFLOAT16 const* __restrict__ input,
                                int64_t const num_elems, int const tid,
                                int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<BFLOAT16> const* vectorized_in =
      reinterpret_cast<vec4_t<BFLOAT16> const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<BFLOAT16> in_vec = vectorized_in[i];
    absmax_val = max(absmax_val, fabs(in_vec.x));
    absmax_val = max(absmax_val, fabs(in_vec.y));
    absmax_val = max(absmax_val, fabs(in_vec.z));
    absmax_val = max(absmax_val, fabs(in_vec.w));
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    absmax_val = max(absmax_val, fabs(input[i]));
  }

  return absmax_val;
}


__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    FP8_TYPE* __restrict__ out, 
    float* __restrict__ scale,
    BFLOAT16 const* __restrict__ input, 
    float const* __restrict__ scale_ub,
    const int hidden_size) 
{
    float const min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);

    int const tid = threadIdx.x;
    int const token_idx = blockIdx.x;
  
    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
    BFLOAT16 const* __restrict__ token_input = &input[offset];
    FP8_TYPE* __restrict__ token_output = &out[offset];
  
    // For vectorization, token_input and token_output pointers need to be
    // aligned at 8-byte and 4-byte addresses respectively.
    bool const can_vectorize = hidden_size % 4 == 0;
  
    float absmax_val = 0.0f;
    if (can_vectorize) {
      absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
    } else {
      for (int i = tid; i < hidden_size; i += blockDim.x) {
        float const x = static_cast<float>(token_input[i]);
        absmax_val = max(absmax_val, fabs(x));
      }
    }
  
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
    __shared__ float token_scale;
    if (tid == 0) {
      if (scale_ub) {
        token_scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        token_scale = block_absmax_val_maybe;
      }
      // token scale computation
      // token_scale = max(token_scale / FP8_E4M3_MAX, min_scaling_factor);
      token_scale = max( FP8_E4M3_MAX / token_scale, 1.0f / min_scaling_factor);
      scale[token_idx] = token_scale;
    }
    __syncthreads();
  
    // Note that we don't use inverted scales so we can match FBGemm impl.
    if (can_vectorize) {
      scaled_fp8_conversion_vec<BFLOAT16, true>(
          token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
    } else {
      for (int i = tid; i < hidden_size; i += blockDim.x) {
        token_output[i] = scaled_fp8_conversion<true>(
            static_cast<float>(token_input[i]), token_scale);
      }
    }
}



void scale_ref(
    torch::Tensor& out,         
    torch::Tensor& scales,
    torch::Tensor const& input,
    int block_size) 
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  dynamic_per_token_scaled_fp8_quant_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<FP8_TYPE>(), 
        scales.data_ptr<float>(),
        input.data_ptr<BFLOAT16>(),
        nullptr,
        hidden_size);

}