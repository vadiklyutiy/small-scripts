#!/bin/bash

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
TP=2
SERVER_PID=""

# Function to start vLLM server
start_server() {
    echo "Starting vLLM server..."
    echo "Command: VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve $MODEL -tp $TP --enable-expert-parallel --async-scheduling --no-enable-prefix-caching --compilation_config.max_cudagraph_capture_size 2048"
    echo ""
    
    VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve $MODEL \
        -tp $TP \
        --enable-expert-parallel \
        --async-scheduling \
        --no-enable-prefix-caching \
        --compilation_config.max_cudagraph_capture_size 2048 \
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

# Function to run benchmarks
run_benchmarks() {
    echo "Running benchmarks..."
    for i in {1..3}; do
        echo "Running iteration $i"
        vllm bench serve \
            --backend vllm \
            --model $MODEL \
            --endpoint /v1/completions \
            --dataset-name random \
            --random-input 32 \
            --random-output 1024 \
            --max-concurrency 1024 \
            --num-prompt 1024 \
            --ignore-eos
    done
    
    echo ""
    echo "Benchmarks completed!"
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
run_benchmarks
stop_server

