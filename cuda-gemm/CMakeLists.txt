cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_gemm LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA flags for H100 architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -lcublas -lcublasLt -arch=sm_90")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add subdirectories for implementations
add_subdirectory(basic-gemm)
add_subdirectory(tensor-core-gemm)
# Uncomment when ready to build
# add_subdirectory(tma-gemm)

# Add benchmark executable
add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark 
                     basic_gemm 
                     tensor_core_gemm 
                     ${CUDA_LIBRARIES}
                     CUDA::cublas
                     CUDA::cublasLt) 