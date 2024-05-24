#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <cuda_device> <seed1> [seed2] ... [seedN]"
    exit 1
fi

cuda_device=$1
shift  
seeds=("$@")  

json_path="your.jsonl"

trap 'echo "Script interrupted."; exit' SIGINT

for seed in "${seeds[@]}"; do
    record_dir="your-path/outputs/record_${seed}.jsonl"
    
    echo "Running with seed $seed on $cuda_device..."
    python ./benchmark/t2iGen/eval_t2igen.py --json_path $json_path --start_line 0 --end_line 3000 --seed $seed --device $cuda_device --record_dir $record_dir
    
    # 检查上一个命令的退出状态，如果失败则停止执行
    if [ $? -ne 0 ]; then
        echo "Error encountered. Exiting."
        exit 1
    fi
done

echo "All runs completed."


