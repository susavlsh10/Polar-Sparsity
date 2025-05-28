#!/bin/bash

# Define parameters
# model_index=7
# stats_dir="results/mlp_results/batch_activations/opt-30b/"
# mlp_ckpt_dir="/pscratch/sd/s/<name>/HybridTensor/checkpoint/opt-30b-routers/mlp/"
# act_data_dir="/pscratch/sd/s/<name>/HybridTensor/opt-30b_act_data/"

model_index=8
stats_dir="results/mlp_results/batch_activations/opt-66b/"
mlp_ckpt_dir="/pscratch/sd/s/<name>/HybridTensor/checkpoint/opt-66b-routers/mlp/"
act_data_dir="/pscratch/sd/s/<name>/HybridTensor/opt-66b_act_data/"

# Array of batch sizes to test (adjust as needed)
batch_sizes=(48 64)

# Concurrency control
max_concurrent=4
current_jobs=0
counter=2

for batch_size in "${batch_sizes[@]}"; do
  # Choose GPU in a round-robin manner (assuming 4 GPUs)
  gpu=$(( counter % 4 ))
  
  # Run the command in the background
  python -m HybridTensor.routers.mlp.mlp_router_optim_fast \
    --model_index "$model_index" \
    --stats_dir "$stats_dir" \
    --mlp_ckpt_dir "$mlp_ckpt_dir" \
    --act_data_dir "$act_data_dir" \
    --batch_size "$batch_size" \
    --delta 200 \
    --device "$gpu" &
    
  current_jobs=$(( current_jobs + 1 ))
  counter=$(( counter + 1 ))
  
  # Check if we've reached the maximum number of concurrent jobs
  if [ "$current_jobs" -ge "$max_concurrent" ]; then
    wait
    current_jobs=0
  fi
done

# Final wait for any remaining background jobs
wait

echo "All tests completed."