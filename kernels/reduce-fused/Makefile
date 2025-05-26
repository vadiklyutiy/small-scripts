NVCC = /usr/local/cuda-12.8/bin/nvcc
CUDA_FLAGS = -lcuda -O3 -Xcompiler -fPIC,-m64,-march=sapphirerapids,-O3,-funroll-loops,-ffast-math -std=c++11 -gencode arch=compute_90a,code=sm_90a --ptxas-options=-v -ftz=true -prec-div=false --cudart shared

# Target executable
TARGET = test_rms

# Source files
RMS_KERNEL_SRC = reduce_kernels_rms.cu
TEST_SRC = test_rms.cu

# Object files
RMS_KERNEL_OBJ = reduce_kernels_rms.o
TEST_OBJ = test_rms.o

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(RMS_KERNEL_OBJ) $(TEST_OBJ)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

# Compile kernel object files
$(RMS_KERNEL_OBJ): $(RMS_KERNEL_SRC)
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
	@echo "  all         - Build the RMS test program (default)"
	@echo "  clean       - Remove build artifacts"
	@echo "  run         - Build and run RMS test"
	@echo "  help        - Show this help message"

.PHONY: all clean run help 