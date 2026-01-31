#!/bin/bash
set -e  # Exit immediately if any command fails

MODEL="nvidia/DeepSeek-R1-0528-FP4"
TP=1
PP=1
DP=8
LOG_FILE="vllm_server-dsr1-fp4.log"
SERVER_PID=""

# Trap to ensure server cleanup on exit
# trap 'if [ -n "$SERVER_PID" ]; then stop_server; fi' EXIT INT TERM

# Function to start vLLM server
start_server() {
    echo "Starting vLLM server..."
    echo "Command: VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_USE_NCCL_SYMM_MEM=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 vllm serve $MODEL --port $PORT -tp $TP -pp $PP -dp $DP --enable-expert-parallel --attention-backend FLASHINFER_MLA --max-cudagraph-capture-size 2048 --compilation_config.cudagraph_mode FULL_DECODE_ONLY --no-enable-prefix-caching --async-scheduling"
    echo ""
    
    VLLM_USE_FLASHINFER_MOE_FP4=1 \
    vllm serve $MODEL \
        -tp $TP -pp $PP -dp $DP --enable-expert-parallel \
        --attention-backend FLASHINFER_MLA \
        --max-cudagraph-capture-size 2048 \
        --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
        --no-enable-prefix-caching \
        --async-scheduling \
        > $LOG_FILE 2>&1 &
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # Wait for server to be ready
    printf "Waiting for server to start..."
    until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
        # Check if server process is still running
        if ! ps -p $SERVER_PID > /dev/null 2>&1; then
            echo ""
            echo "ERROR: Server process died. Check $LOG_FILE for details."
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
            --random-output 1000 \
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
    echo "Server logs saved to: $LOG_FILE"
}

# Main execution
echo "==================== GPU Information ===================="
nvidia-smi --query-gpu=index,persistence_mode,clocks.current.sm --format=csv
echo "========================================================="
echo ""

start_server
run_benchmarks
stop_server
