#!/bin/bash


if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <model_path> <lora_path> <question_path> <base_answer_path> <image_folder> <N> <temperature> <start_gpu>"
    exit 1
fi

# Assign the command line arguments to variables
model_path=$1
lora_path=$2
question_path=$3
base_answer_path=$4
image_folder=$5
N=$6
temperature=$7
GS=$8



# Define the GPUs to be used
gpus=(0 1 2 3 4 5 6 7)

for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
    
    # Calculate the GPU index by cycling through the gpus array
    gpu_index=$((chunk_id % ${#gpus[@]}))
    gpu=${gpus[$gpu_index]}
    
    # Run the Python program in the background
    CUDA_VISIBLE_DEVICES="$gpu" python ./benchmark/POPE/eval_pope.py --model-base "$model_path" --model-path "$lora_path" --question-file "$question_path" --answers-file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --image-folder "$image_folder" --temperature "$temperature" &

done


wait

merged_file="${base_answer_path}_merged.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi

for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}_${i}.jsonl"
  cat "$input_file" >> "${base_answer_path}_merged.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done
