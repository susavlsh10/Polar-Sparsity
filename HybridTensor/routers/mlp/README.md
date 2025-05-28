# To run the mlp router training use the following scripts

for a single layer

python -m HybridTensor.routers.mlp.main_mlp --model_index 5 --L 4 --data_dir <PATH_TO_ACTIVATION_DATA> --ckpt_dir <PATH_TO_CHECKPOINT> --gpu 1


for all the layers 

./HybridTensor/routers/mlp/train_mlp_routers.sh