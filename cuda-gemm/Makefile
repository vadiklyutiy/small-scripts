.PHONY: all build debug clean benchmark

all: build

build:
	mkdir -p build
	cd build && cmake .. && cmake --build . -j$$(nproc)

debug:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build . -j$$(nproc)
	
run-basic:
	./build/bin/test_basic_gemm

benchmark:
	./build/bin/benchmark

clean:
	rm -rf build 