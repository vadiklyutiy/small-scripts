cmake_minimum_required(VERSION 3.18)
project(matrix_mul_benchmark CUDA CXX)

# Set CUDA architecture flags for H100 (SM 9.0)
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find packages
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add library for matrix multiplication implementations
add_library(matrix_mul
    src/matrix_mul.cu
)

# Add executable for testing
add_executable(matrix_mul_test
    src/matrix_mul_test.cu
)
target_link_libraries(matrix_mul_test matrix_mul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Add executable for benchmarking
add_executable(matrix_mul_benchmark
    src/matrix_mul_benchmark.cu
)
target_link_libraries(matrix_mul_benchmark matrix_mul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Create a build directory and specify where to place the built files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 