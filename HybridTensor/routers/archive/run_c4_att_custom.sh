#!/bin/bash

# update this based on the model and the number of layers
total_num_layers=32

for l in $(seq 0 4 32)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L ${l} > logs/c4_att_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L $((l+1)) > logs/c4_att_out_$((l+1)).txt & \
    CUDA_VISIBLE_DEVICES=0 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L $((l+2)) > logs/c4_att_out_$((l+2)).txt & \
    CUDA_VISIBLE_DEVICES=1 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L $((l+3)) > logs/c4_att_out_$((l+3)).txt & \ 

    wait)
done