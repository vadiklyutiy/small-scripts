cmake_minimum_required(VERSION 3.10)
project(basic_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add source files
set(SOURCES
    src/gemm_basic.cu
    src/gemm_shared_mem.cu
)

# Create shared library
add_library(basic_gemm SHARED ${SOURCES})

# Create test executable
add_executable(test_basic_gemm src/test_basic_gemm.cu)
target_link_libraries(test_basic_gemm basic_gemm ${CUDA_LIBRARIES})

# Install targets
install(TARGETS basic_gemm test_basic_gemm
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Set compilation flags
set_target_properties(basic_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(basic_gemm PROPERTIES
        CUDA_FLAGS "-g -G -lineinfo"
    )
endif() 