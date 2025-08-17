#!/bin/bash

# Create build directory
mkdir -p build && cd build

# Configure with CMake - use Debug build type for debugging
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build . -j $(nproc)

# Run tests
echo "Running tests..."
./bin/test_tma_gemm

cd .. 