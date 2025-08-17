CXX = nvcc
CXXFLAGS = -std=c++14 -Xcompiler -O3 -arch=sm_90
LDFLAGS = -lcublas -lcublasLt

TARGETS = benchmark_cublaslt validate_cublaslt benchmark_cublaslt_algos

all: $(TARGETS)

benchmark_cublaslt: benchmark_cublaslt.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

validate_cublaslt: validate_cublaslt.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

benchmark_cublaslt_algos: benchmark_cublaslt_algos.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean 