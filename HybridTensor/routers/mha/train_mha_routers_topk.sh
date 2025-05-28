#!/usr/bin/env bash

# Number of GPUs available
NUM_GPUS=2

# Each GPU can handle JOBS_PER_GPU concurrent trainings
JOBS_PER_GPU=8

# Total layers you want to process
START_LAYER=0
END_LAYER=31

MODEL_INDEX=14
D=128
DATA_DIR="<path_to_your_data_dir>"
CKPT_DIR="<path_to_your_checkpoint_dir>"

BATCH_SIZE=$((NUM_GPUS * JOBS_PER_GPU))

# print all the configs 
echo "NUM_GPUS: $NUM_GPUS, JOBS_PER_GPU: $JOBS_PER_GPU, START_LAYER: $START_LAYER, END_LAYER: $END_LAYER, MODEL_INDEX: $MODEL_INDEX, DATA_DIR: $DATA_DIR, CKPT_DIR: $CKPT_DIR, BATCH_SIZE: $BATCH_SIZE"

layer=$START_LAYER
while [ $layer -le $END_LAYER ]; do
    # Launch a batch of jobs
    for (( i=0; i<$BATCH_SIZE && layer<=$END_LAYER; i++ )); do
        GPU_ID=$(( i / JOBS_PER_GPU ))
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m HybridTensor.routers.mha.main_att\
            --model_index $MODEL_INDEX \
            --L $layer \
            --D $D \
            --k 0.5 \
            --data_dir $DATA_DIR \
            --ckpt_dir $CKPT_DIR &
        layer=$((layer+1))
    done

    # Wait for this batch to finish
    wait
done

echo "All training jobs finished!"