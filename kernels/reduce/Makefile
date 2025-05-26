NVCC = /usr/local/cuda-12.8/bin/nvcc
CUDA_FLAGS = -lcuda -O3 -Xcompiler -fPIC,-m64,-march=sapphirerapids,-O3,-funroll-loops,-ffast-math -std=c++11 -gencode arch=compute_90a,code=sm_90a --ptxas-options=-v -ftz=true -prec-div=false --cudart shared

# Target executable
TARGET = test_reduce

# Source files
KERNEL_SRC = reduce_kernels.cu
KERNEL_CUB_SRC = reduce_kernels_cub.cu
TEST_SRC = test_reduce.cu

# Object files
KERNEL_OBJ = reduce_kernels.o
KERNEL_CUB_OBJ = reduce_kernels_cub.o
TEST_OBJ = test_reduce.o

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(KERNEL_OBJ) $(KERNEL_CUB_OBJ) $(TEST_OBJ)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

# Compile kernel object files
$(KERNEL_OBJ): $(KERNEL_SRC)
	$(NVCC) $(CUDA_FLAGS) -dc -o $@ $<

$(KERNEL_CUB_OBJ): $(KERNEL_CUB_SRC)
	$(NVCC) $(CUDA_FLAGS) -dc -o $@ $<

$(TEST_OBJ): $(TEST_SRC)
	$(NVCC) $(CUDA_FLAGS) -dc -o $@ $<

# Clean target
clean:
	rm -f $(TARGET) *.o

# Run the test
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the test program (default)"
	@echo "  clean       - Remove build artifacts"
	@echo "  run         - Build and run test (18 kernels)"
	@echo "  help        - Show this help message"

.PHONY: all clean run help 