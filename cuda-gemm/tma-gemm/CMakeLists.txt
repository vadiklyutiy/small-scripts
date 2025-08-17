cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_tma_gemm LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA flags for H100 architecture specifically needed for TMA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -lcublas -lcublasLt -arch=sm_90")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

# Add global debug definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DDEBUG_TMA)
endif()

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add library
add_subdirectory(src)

# Build tests
add_subdirectory(tests) 