#include "matrix_mul.cuh"
#include <iomanip>
#include <algorithm>

// Structure to hold benchmark results
struct BenchmarkResult {
    std::string name;
    std::vector<float> times;
    float min_time;
    float max_time;
    float avg_time;
    float gflops;
};

// Calculate FLOPS (Floating Point Operations Per Second)
float calculate_gflops(float time_us) {
    // For matrix multiply: 2 * M * N * K operations (multiply and add for each element)
    float operations = 2.0f * M * N * K;
    float time_s = time_us / 1000000.0f; // Convert microseconds to seconds
    return operations / (time_s * 1e9);
}

// Run benchmark for a specific implementation
BenchmarkResult run_benchmark(const char* name, float (*gemm_func)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*), 
                             __nv_bfloat16* d_A, __nv_bfloat16* d_B, __nv_bfloat16* d_C, int iterations) {
    BenchmarkResult result;
    result.name = name;
    
    for (int i = 0; i < iterations; i++) {
        // Run the kernel and get execution time in ms
        float time_ms = gemm_func(d_A, d_B, d_C);
        
        // Convert to microseconds
        float time_us = time_ms * 1000.0f;
        
        // Store the time
        result.times.push_back(time_us);
    }
    
    // Calculate statistics
    result.min_time = *std::min_element(result.times.begin(), result.times.end());
    result.max_time = *std::max_element(result.times.begin(), result.times.end());
    
    float sum = 0.0f;
    for (float time : result.times) {
        sum += time;
    }
    result.avg_time = sum / iterations;
    
    // Calculate GFLOPS based on microsecond time
    result.gflops = calculate_gflops(result.min_time);
    
    return result;
}

// Print benchmark results
void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n----- Benchmark Results -----\n" << std::endl;
    
    // Print table header
    std::cout << std::left << std::setw(30) << "Implementation" 
              << std::right << std::setw(15) << "Min (μs)" 
              << std::right << std::setw(15) << "Max (μs)" 
              << std::right << std::setw(15) << "Avg (μs)" 
              << std::right << std::setw(15) << "GFLOPS" 
              << std::endl;
    
    std::cout << std::string(90, '-') << std::endl;
    
    // Print results for each implementation
    for (const auto& result : results) {
        std::cout << std::left << std::setw(30) << result.name 
                  << std::right << std::fixed << std::setprecision(3) << std::setw(15) << result.min_time 
                  << std::right << std::fixed << std::setprecision(3) << std::setw(15) << result.max_time
                  << std::right << std::fixed << std::setprecision(3) << std::setw(15) << result.avg_time
                  << std::right << std::fixed << std::setprecision(2) << std::setw(15) << result.gflops
                  << std::endl;
    }
    
    // Find fastest implementation
    auto fastest = std::max_element(results.begin(), results.end(), 
                                  [](const BenchmarkResult& a, const BenchmarkResult& b) {
                                      return a.gflops < b.gflops;
                                  });
    
    std::cout << "\nFastest implementation: " << fastest->name 
              << " with " << fastest->gflops << " GFLOPS" << std::endl;
}

int main() {
    std::cout << "Matrix Multiplication Benchmark" << std::endl;
    std::cout << "A: " << M << "x" << K << " (row major)" << std::endl;
    std::cout << "B: " << K << "x" << N << " (column major)" << std::endl;
    std::cout << "C: " << M << "x" << N << std::endl;
    
    // Number of benchmark iterations
    const int iterations = 10;
    
    // Allocate host memory
    __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
    
    // Initialize matrices
    std::cout << "Initializing matrices..." << std::endl;
    __nv_bfloat16* h_C = new __nv_bfloat16[M * N]; // Only needed for initialization
    init_matrices(h_A, h_B, h_C);
    delete[] h_C;
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));
    
    // Copy input data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Vector to store benchmark results
    std::vector<BenchmarkResult> results;
    
    // Warm up GPU before benchmarking
    std::cout << "Warming up GPU..." << std::endl;
    for (int i = 0; i < 3; i++) {
        gemm_cublasGemmEx(d_A, d_B, d_C);
    }
    
    // Benchmark each implementation
    std::cout << "Running benchmarks (" << iterations << " iterations each)..." << std::endl;
    
    // Benchmark cublasGemmEx
    results.push_back(run_benchmark("cublasGemmEx", gemm_cublasGemmEx, d_A, d_B, d_C, iterations));
    
    // Benchmark cublasGemmBatchedEx
    results.push_back(run_benchmark("cublasGemmBatchedEx", gemm_cublasGemmBatchedEx, d_A, d_B, d_C, iterations));
    
    // Benchmark cublasGemmStridedBatchedEx
    results.push_back(run_benchmark("cublasGemmStridedBatchedEx", gemm_cublasGemmStridedBatchedEx, d_A, d_B, d_C, iterations));
    
    // Print benchmark results
    print_results(results);
    
    // Free memory
    delete[] h_A;
    delete[] h_B;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
} 