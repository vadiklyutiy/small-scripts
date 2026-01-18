#!/bin/bash

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
TP=1
SERVER_PID=""

# Function to start vLLM server
start_server() {
    echo "Starting vLLM server..."
    echo "Command: CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' CUDA_COREDUMP_FILE=\"/tmp/vgimpelson/cuda_coredump_%h.%p.%t\" CUDA_LAUNCH_BLOCKING=1 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve $MODEL --tokenizer-mode auto --gpu-memory-utilization 0.8 --tensor-parallel-size $TP --no-enable-prefix-caching --speculative-config '{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 2}'"
    echo ""
    
    PYTHONFAULTHANDLER=1 \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    TORCH_USE_CUDA_DSA=1 \
    CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
    CUDA_COREDUMP_SHOW_PROGRESS=1 \
    CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
    CUDA_COREDUMP_FILE="/tmp/vgimpelson/cuda_coredump_%h.%p.%t" \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    vllm serve $MODEL \
        --tokenizer-mode auto \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8 \
        --tensor-parallel-size $TP \
        --no-enable-prefix-caching \
        --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
        > vllm_server.log 2>&1 &
        
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # Wait for server to be ready
    printf "Waiting for server to start..."
    until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
        # Check if server process is still running
        if ! ps -p $SERVER_PID > /dev/null 2>&1; then
            echo ""
            echo "ERROR: Server process died. Check vllm_server.log for details."
            exit 1
        fi
        printf "."
        sleep 1
    done
    printf " Server ready!\n"
    echo ""
}

# Function to run evaluation
run_benchmarks() {
    local iteration=$1
    echo "Running lm_eval evaluation (iteration $iteration/5)..."
    
    # Activate lm_eval virtual environment and run evaluation
    source /home/scratch.vgimpelson_ent/venv_lm_eval/bin/activate
    
    lm_eval \
        --model local-completions \
        --tasks gsm8k \
        --model_args base_url=http://localhost:8000/v1/completions,model=$MODEL,tokenized_requests=False,tokenizer_backend=None,num_concurrent=512,timeout=120,max_retries=5
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ERROR: Evaluation failed with exit code $exit_code"
        return $exit_code
    fi
    
    echo ""
    echo "Evaluation iteration $iteration completed successfully!"
    return 0
}

# Function to stop vLLM server
stop_server() {
    echo "Terminating vLLM server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    sleep 2
    
    # Force kill if still running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Force killing server..."
        kill -9 $SERVER_PID
    fi
    
    echo "Server terminated."
    echo "Server logs saved to: vllm_server.log"
}

# Main execution
echo "==================== GPU Information ===================="
nvidia-smi --query-gpu=index,persistence_mode,clocks.current.sm --format=csv
echo "========================================================="
echo ""

start_server

# Run benchmarks 5 times, stop if any iteration fails
BENCHMARK_SUCCESS=true
for i in {1..5}; do
    run_benchmarks $i
    if [ $? -ne 0 ]; then
        echo "Stopping benchmark iterations due to failure."
        BENCHMARK_SUCCESS=false
        break
    fi
    echo ""
    echo "-----------------------------------------------------------"
    echo ""
done

stop_server

