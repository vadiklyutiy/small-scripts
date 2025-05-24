NVCC = /usr/local/cuda-12.8/bin/nvcc
CFLAGS = -lcuda -O3 -Xcompiler -fPIC,-m64,-march=sapphirerapids,-O3,-funroll-loops,-ffast-math -std=c++11 -gencode arch=compute_90a,code=sm_90a --ptxas-options=-v -lineinfo -ftz=true -prec-div=false --cudart shared

TARGET = silu_test
SOURCE = silu_kernel.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run 