.PHONY: all build debug clean profile nsight

all: build

build:
	mkdir -p build
	cd build && cmake .. && cmake --build . -j$$(nproc)

debug:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build . -j$$(nproc)
	
run:
	./build/bin/test_tma_gemm

profile:
	nsys profile --trace=cuda,osrt,nvtx ./build/bin/test_tma_gemm

nsight:
	ncu --metrics all ./build/bin/test_tma_gemm

clean:
	rm -rf build 