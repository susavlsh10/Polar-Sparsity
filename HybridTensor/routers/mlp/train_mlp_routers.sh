#!/usr/bin/env bash

# Number of GPUs available
NUM_GPUS=2

# Each GPU can handle JOBS_PER_GPU concurrent trainings
JOBS_PER_GPU=1

# Total layers you want to process
START_LAYER=16
END_LAYER=31


MODEL_INDEX=5
D=1024
DATA_DIR="/home/grads/s/<name>/nvme/HybridTensor/opt-6.7b_act_data/"
CKPT_DIR="/home/grads/s/<name>/nvme/HybridTensor/checkpoint"

BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

layer=$START_LAYER
while [ $layer -le $END_LAYER ]; do
    # Launch a batch of jobs
    for (( i=0; i<$BATCH_SIZE && layer<=$END_LAYER; i++ )); do
        GPU_ID=$(( i / JOBS_PER_GPU ))
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m HybridTensor.routers.mlp.main_mlp\
            --model_index $MODEL_INDEX \
            --L $layer \
            --D $D \
            --data_dir $DATA_DIR \
            --ckpt_dir $CKPT_DIR &
        layer=$((layer+1))
    done

    # Wait for this batch to finish
    wait
done

echo "All training jobs finished!"