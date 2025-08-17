add_executable(test_tma_gemm test_tma_gemm.cu)
target_link_libraries(test_tma_gemm cuda_tma_gemm CUDA::cudart)

# Add debug build type with additional flags
target_compile_definitions(test_tma_gemm PRIVATE 
    "$<$<CONFIG:Debug>:DEBUG_TEST>") 