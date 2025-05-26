#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

// Hand-optimized kernel declarations
__global__ void reduce_mean_32_atomic(float* input, float* output, int width);
__global__ void reduce_mean_32_separate(float* input, float* output, int width);
__global__ void reduce_mean_64_atomic(float* input, float* output, int width);
__global__ void reduce_mean_64_separate(float* input, float* output, int width);
__global__ void reduce_mean_128_atomic(float* input, float* output, int width);
__global__ void reduce_mean_128_separate(float* input, float* output, int width);
__global__ void reduce_mean_256_atomic(float* input, float* output, int width);
__global__ void reduce_mean_256_separate(float* input, float* output, int width);
__global__ void reduce_mean_512_atomic(float* input, float* output, int width);
__global__ void reduce_mean_512_separate(float* input, float* output, int width);
__global__ void reduce_mean_1024_atomic(float* input, float* output, int width);
__global__ void reduce_mean_1024_separate(float* input, float* output, int width);

// CUB kernel declarations
__global__ void reduce_mean_32_cub(float* input, float* output, int width);
__global__ void reduce_mean_64_cub(float* input, float* output, int width);
__global__ void reduce_mean_128_cub(float* input, float* output, int width);
__global__ void reduce_mean_256_cub(float* input, float* output, int width);
__global__ void reduce_mean_512_cub(float* input, float* output, int width);
__global__ void reduce_mean_1024_cub(float* input, float* output, int width);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

struct KernelInfo {
    const char* name;
    void (*kernel)(float*, float*, int);
    int threads_per_block;
    const char* type;
};

// CPU reference implementation
void cpu_reduce_mean(const std::vector<float>& input, std::vector<float>& output, int height, int width) {
    for (int row = 0; row < height; row++) {
        float sum = 0.0f;
        for (int col = 0; col < width; col++) {
            sum += input[row * width + col];
        }
        output[row] = sum / width;
    }
}

// Check correctness
bool check_correctness(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, float tolerance = 1e-4f) {
    if (gpu_result.size() != cpu_result.size()) {
        std::cerr << "Size mismatch!" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < gpu_result.size(); i++) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << gpu_result[i] 
                      << ", CPU=" << cpu_result[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

// Benchmark a kernel
double benchmark_kernel(void (*kernel)(float*, float*, int), 
                       float* d_input, float* d_output, 
                       int height, int width, int threads_per_block,
                       const char* kernel_name, const char* kernel_type, int warmup_runs = 5, int bench_runs = 100) {
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; i++) {
        kernel<<<height, threads_per_block>>>(d_input, d_output, width);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_runs; i++) {
        kernel<<<height, threads_per_block>>>(d_input, d_output, width);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = duration.count() / (double)bench_runs;
    
    std::cout << std::setw(20) << kernel_name 
              << std::setw(10) << threads_per_block
              << std::setw(8) << kernel_type
              << std::setw(12) << std::fixed << std::setprecision(2) << avg_time_us << " μs";
    
    return avg_time_us;
}

int main() {
    const int HEIGHT = 2048;
    const int WIDTH = 2048;
    const int TOTAL_SIZE = HEIGHT * WIDTH;
    
    std::cout << "CUDA Reduce Mean Kernels - Comprehensive Performance Test" << std::endl;
    std::cout << "Matrix size: " << HEIGHT << "x" << WIDTH << std::endl;
    std::cout << "Total kernels: 18 (12 hand-optimized + 6 CUB)" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Initialize random data
    std::vector<float> h_input(TOTAL_SIZE);
    std::vector<float> h_output(HEIGHT);
    std::vector<float> h_cpu_output(HEIGHT);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = dis(gen);
    }
    
    // Compute CPU reference
    std::cout << "Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reduce_mean(h_input, h_cpu_output, HEIGHT, WIDTH);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    std::cout << "CPU time: " << cpu_duration.count() << " μs" << std::endl << std::endl;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, TOTAL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, HEIGHT * sizeof(float)));
    
    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define all kernels
    KernelInfo kernels[] = {
        // Hand-optimized shuffle kernels
        {"32_atomic",      (void(*)(float*, float*, int))reduce_mean_32_atomic,      32,   "Shuffle"},
        {"32_separate",    (void(*)(float*, float*, int))reduce_mean_32_separate,    32,   "Shuffle"},
        {"64_atomic",      (void(*)(float*, float*, int))reduce_mean_64_atomic,      64,   "Shuffle"},
        {"64_separate",    (void(*)(float*, float*, int))reduce_mean_64_separate,    64,   "Shuffle"},
        {"128_atomic",     (void(*)(float*, float*, int))reduce_mean_128_atomic,     128,  "Shuffle"},
        {"128_separate",   (void(*)(float*, float*, int))reduce_mean_128_separate,   128,  "Shuffle"},
        {"256_atomic",     (void(*)(float*, float*, int))reduce_mean_256_atomic,     256,  "Shuffle"},
        {"256_separate",   (void(*)(float*, float*, int))reduce_mean_256_separate,   256,  "Shuffle"},
        {"512_atomic",     (void(*)(float*, float*, int))reduce_mean_512_atomic,     512,  "Shuffle"},
        {"512_separate",   (void(*)(float*, float*, int))reduce_mean_512_separate,   512,  "Shuffle"},
        {"1024_atomic",    (void(*)(float*, float*, int))reduce_mean_1024_atomic,    1024, "Shuffle"},
        {"1024_separate",  (void(*)(float*, float*, int))reduce_mean_1024_separate,  1024, "Shuffle"},
        
        // CUB kernels
        {"32_cub",         (void(*)(float*, float*, int))reduce_mean_32_cub,         32,   "CUB"},
        {"64_cub",         (void(*)(float*, float*, int))reduce_mean_64_cub,         64,   "CUB"},
        {"128_cub",        (void(*)(float*, float*, int))reduce_mean_128_cub,        128,  "CUB"},
        {"256_cub",        (void(*)(float*, float*, int))reduce_mean_256_cub,        256,  "CUB"},
        {"512_cub",        (void(*)(float*, float*, int))reduce_mean_512_cub,        512,  "CUB"},
        {"1024_cub",       (void(*)(float*, float*, int))reduce_mean_1024_cub,       1024, "CUB"}
    };
    
    std::cout << std::setw(20) << "Kernel Name" 
              << std::setw(10) << "Threads"
              << std::setw(8) << "Type"
              << std::setw(12) << "Time"
              << std::setw(10) << "Status"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<double> times;
    double best_time = std::numeric_limits<double>::max();
    std::string best_kernel;
    
    // Test each kernel
    for (const auto& kernel_info : kernels) {
        // Clear output
        CUDA_CHECK(cudaMemset(d_output, 0, HEIGHT * sizeof(float)));
        
        try {
            // Run benchmark
            double avg_time = benchmark_kernel(kernel_info.kernel, d_input, d_output, 
                                             HEIGHT, WIDTH, kernel_info.threads_per_block,
                                             kernel_info.name, kernel_info.type);
            
            times.push_back(avg_time);
            
            // Track best performer
            if (avg_time < best_time) {
                best_time = avg_time;
                best_kernel = std::string(kernel_info.name) + " (" + kernel_info.type + ")";
            }
            
            // Copy result back for correctness check
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Check correctness and complete the line
            bool correct = check_correctness(h_output, h_cpu_output);
            std::cout << std::setw(10) << (correct ? "PASS" : "FAIL") << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << std::setw(10) << "ERROR" << std::endl;
            std::cerr << "Error testing " << kernel_info.name << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << std::string(60, '-') << std::endl;
    
    // Performance analysis
    std::cout << "\n📊 PERFORMANCE ANALYSIS:" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Find best and worst
    auto min_time = *std::min_element(times.begin(), times.end());
    auto max_time = *std::max_element(times.begin(), times.end());
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    std::cout << "🏆 Best performer: " << best_kernel << " - " << best_time << " μs" << std::endl;
    std::cout << "📈 Performance range: " << min_time << " μs to " << max_time << " μs" << std::endl;
    std::cout << "📊 Average time: " << std::fixed << std::setprecision(2) << avg_time << " μs" << std::endl;
    std::cout << "🎯 Speedup range: " << std::fixed << std::setprecision(1) 
              << (max_time/min_time) << "x difference" << std::endl;
    
    // Separate analysis by type
    std::vector<double> shuffle_times, cub_times;
    for (size_t i = 0; i < 12; i++) shuffle_times.push_back(times[i]);
    for (size_t i = 12; i < 18; i++) cub_times.push_back(times[i]);
    
    auto best_shuffle = *std::min_element(shuffle_times.begin(), shuffle_times.end());
    auto best_cub = *std::min_element(cub_times.begin(), cub_times.end());
    
    std::cout << "\n🔧 Implementation Comparison:" << std::endl;
    std::cout << "Hand-optimized shuffle: " << best_shuffle << " μs (best)" << std::endl;
    std::cout << "CUB library:           " << best_cub << " μs (best)" << std::endl;
    std::cout << "Winner: " << (best_shuffle < best_cub ? "Hand-optimized" : "CUB") 
              << " by " << std::abs(best_shuffle - best_cub) << " μs" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
} 