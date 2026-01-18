#!/bin/bash

K1=1024
K8=8192

TP=4
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
CONCURRENCY_LIST=(1 4 8 16 32 64 128 256 512 1024)
#IOSL=("${K1} ${K1}" "${K8} ${K1}" "${K1} ${K8}")
IOSL=("${K1} ${K8}")


# Display GPU Information
echo "==================== GPU Information ===================="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo "========================================================="
echo ""


printf "Waiting for Server to start..."

until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
  printf "."
  sleep 1
done

printf "Server started\n"

for iosl in "${IOSL[@]}"; do
    ISL=$(echo $iosl | awk '{print $1}')
    OSL=$(echo $iosl | awk '{print $2}')
    
    echo "Running benchmarks with ISL=$ISL, OSL=$OSL"
    
    for concurrency in "${CONCURRENCY_LIST[@]}"; do 
        echo "Running benchmarks with concurrency $concurrency"
        for i in {1..3}; do 
            echo "Running iteration $i"
            vllm bench serve --backend vllm --model $MODEL --endpoint /v1/completions \
            --dataset-name random --random-input $ISL --random-output $OSL \
            --max-concurrency $concurrency --num-prompt $((5 * concurrency)) --ignore-eos; 
        done
        echo "Finished concurrency $concurrency"
    done
done



