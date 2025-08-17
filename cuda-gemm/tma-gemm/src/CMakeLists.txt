set(TMA_SOURCES
    gemm_tma.cu
    utils.cu
)

# Add library
add_library(cuda_tma_gemm SHARED ${TMA_SOURCES})
target_link_libraries(cuda_tma_gemm CUDA::cudart)
set_target_properties(cuda_tma_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 