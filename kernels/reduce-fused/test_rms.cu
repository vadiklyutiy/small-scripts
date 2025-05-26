#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

// RMS kernel declarations
__global__ void reduce_rms_32_atomic(float* input, float* output, int width);
__global__ void reduce_rms_32_separate(float* input, float* output, int width);
__global__ void reduce_rms_64_atomic(float* input, float* output, int width);
__global__ void reduce_rms_64_separate(float* input, float* output, int width);
__global__ void reduce_rms_128_atomic(float* input, float* output, int width);
__global__ void reduce_rms_128_separate(float* input, float* output, int width);
__global__ void reduce_rms_256_atomic(float* input, float* output, int width);
__global__ void reduce_rms_256_separate(float* input, float* output, int width);
__global__ void reduce_rms_512_atomic(float* input, float* output, int width);
__global__ void reduce_rms_512_separate(float* input, float* output, int width);
__global__ void reduce_rms_1024_atomic(float* input, float* output, int width);
__global__ void reduce_rms_1024_separate(float* input, float* output, int width);

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

// CPU reference implementation for RMS
void cpu_reduce_rms(const std::vector<float>& input, std::vector<float>& output, int height, int width) {
    for (int row = 0; row < height; row++) {
        float sum_squares = 0.0f;
        for (int col = 0; col < width; col++) {
            float val = input[row * width + col];
            sum_squares += val * val;
        }
        output[row] = sqrtf((sum_squares / width) + 1e-06f);
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
    
    std::cout << "CUDA Fused RMS Kernels - Performance Test" << std::endl;
    std::cout << "Operation: RMS = sqrt((sum(x^2)/N) + epsilon)" << std::endl;
    std::cout << "Matrix size: " << HEIGHT << "x" << WIDTH << std::endl;
    std::cout << "Epsilon: 1e-06f" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    // Initialize random data
    std::vector<float> h_input(TOTAL_SIZE);
    std::vector<float> h_output_rms(HEIGHT);
    std::vector<float> h_cpu_rms(HEIGHT);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Use negative/positive for RMS
    
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = dis(gen);
    }
    
    // Compute CPU reference
    std::cout << "Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reduce_rms(h_input, h_cpu_rms, HEIGHT, WIDTH);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    std::cout << "CPU time: " << cpu_duration.count() << " μs" << std::endl << std::endl;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, TOTAL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, HEIGHT * sizeof(float)));
    
    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define RMS kernels
    KernelInfo rms_kernels[] = {
        {"32_atomic",      (void(*)(float*, float*, int))reduce_rms_32_atomic,      32,   "RMS"},
        {"32_separate",    (void(*)(float*, float*, int))reduce_rms_32_separate,    32,   "RMS"},
        {"64_atomic",      (void(*)(float*, float*, int))reduce_rms_64_atomic,      64,   "RMS"},
        {"64_separate",    (void(*)(float*, float*, int))reduce_rms_64_separate,    64,   "RMS"},
        {"128_atomic",     (void(*)(float*, float*, int))reduce_rms_128_atomic,     128,  "RMS"},
        {"128_separate",   (void(*)(float*, float*, int))reduce_rms_128_separate,   128,  "RMS"},
        {"256_atomic",     (void(*)(float*, float*, int))reduce_rms_256_atomic,     256,  "RMS"},
        {"256_separate",   (void(*)(float*, float*, int))reduce_rms_256_separate,   256,  "RMS"},
        {"512_atomic",     (void(*)(float*, float*, int))reduce_rms_512_atomic,     512,  "RMS"},
        {"512_separate",   (void(*)(float*, float*, int))reduce_rms_512_separate,   512,  "RMS"},
        {"1024_atomic",    (void(*)(float*, float*, int))reduce_rms_1024_atomic,    1024, "RMS"},
        {"1024_separate",  (void(*)(float*, float*, int))reduce_rms_1024_separate,  1024, "RMS"}
    };
    
    std::cout << std::setw(20) << "Kernel Name" 
              << std::setw(10) << "Threads"
              << std::setw(8) << "Type"
              << std::setw(12) << "Time"
              << std::setw(10) << "Status"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<double> rms_times;
    double best_rms_time = std::numeric_limits<double>::max();
    std::string best_rms_kernel;
    
    // Test RMS kernels
    for (const auto& kernel_info : rms_kernels) {
        // Clear output
        CUDA_CHECK(cudaMemset(d_output, 0, HEIGHT * sizeof(float)));
        
        try {
            // Run benchmark
            double avg_time = benchmark_kernel(kernel_info.kernel, d_input, d_output, 
                                             HEIGHT, WIDTH, kernel_info.threads_per_block,
                                             kernel_info.name, kernel_info.type);
            
            rms_times.push_back(avg_time);
            
            // Track best performer
            if (avg_time < best_rms_time) {
                best_rms_time = avg_time;
                best_rms_kernel = std::string(kernel_info.name) + " (" + kernel_info.type + ")";
            }
            
            // Copy result back for correctness check
            CUDA_CHECK(cudaMemcpy(h_output_rms.data(), d_output, HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Check correctness and complete the line
            bool correct = check_correctness(h_output_rms, h_cpu_rms);
            std::cout << std::setw(10) << (correct ? "PASS" : "FAIL") << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << std::setw(10) << "ERROR" << std::endl;
            std::cerr << "Error testing " << kernel_info.name << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << std::string(60, '-') << std::endl;
    
    // Performance analysis
    std::cout << "\n📊 RMS PERFORMANCE ANALYSIS:" << std::endl;
    std::cout << "============================" << std::endl;
    
    auto min_rms = *std::min_element(rms_times.begin(), rms_times.end());
    auto max_rms = *std::max_element(rms_times.begin(), rms_times.end());
    double avg_rms = std::accumulate(rms_times.begin(), rms_times.end(), 0.0) / rms_times.size();
    
    std::cout << "🏆 Best RMS kernel: " << best_rms_kernel << " - " << best_rms_time << " μs" << std::endl;
    std::cout << "📈 Performance range: " << min_rms << " μs to " << max_rms << " μs" << std::endl;
    std::cout << "📊 Average time: " << std::fixed << std::setprecision(2) << avg_rms << " μs" << std::endl;
    std::cout << "🎯 Speedup range: " << std::fixed << std::setprecision(1) 
              << (max_rms/min_rms) << "x difference" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
} 