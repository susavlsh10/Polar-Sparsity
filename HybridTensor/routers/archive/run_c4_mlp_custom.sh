#!/bin/bash

# update this based on the model and the number of layers
total_num_layers=32

for l in $(seq 0 2 $total_num_layers)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_mlp.py --dataset c4 --lr 0.001 --L ${l} > logs/c4_mlp_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+1)) > logs/c4_mlp_out_$((l+1)).txt & \
    wait)
done