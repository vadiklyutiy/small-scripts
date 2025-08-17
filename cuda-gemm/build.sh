#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to build an implementation
build_implementation() {
    implementation=$1
    echo -e "${BLUE}Building ${implementation}...${NC}"
    
    # Create build directory
    mkdir -p ${implementation}/build
    cd ${implementation}/build
    
    # Configure cmake
    cmake ..
    
    # Build
    make -j
    
    # Return to root
    cd ../..
    
    echo -e "${GREEN}Done building ${implementation}${NC}"
}

# Create main build directory
mkdir -p build

# Build all implementations
echo -e "${BLUE}Building all GEMM implementations...${NC}"

if [ -d "basic-gemm" ]; then
    build_implementation "basic-gemm"
fi

if [ -d "tma-gemm" ]; then
    build_implementation "tma-gemm"
fi

if [ -d "tensor-core-gemm" ]; then
    build_implementation "tensor-core-gemm"
fi

echo -e "${GREEN}All implementations built successfully!${NC}" 