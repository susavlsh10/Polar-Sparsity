# Train activation routers for MLP and MHA layers 

## MLP 

Update the bash file in HybridTensor/routers/mlp/train_mlp_routers.sh to include details about the job

```bash
./HybridTensor/routers/mlp/train_mlp_routers.sh
```


## MLP Router Optimization


A5000
```bash
python -m HybridTensor.routers.mlp.mlp_router_optim_fast --model_index 5 \
--batch_size 32 --delta 200 
```

Single batch optimization

```bash
python -m HybridTensor.routers.mlp.mlp_router_optim --model_index 8 \
--stats_dir results/mlp_results/batch_activations/opt-66b/ \
--mlp_ckpt_dir /pscratch/sd/s/<name>/HybridTensor/checkpoint/opt-66b-routers/mlp/ \
--act_data_dir /pscratch/sd/s/<name>/HybridTensor/opt-66b_act_data/
```

Batched optimization 

```bash
python -m HybridTensor.routers.mlp.mlp_router_optim_fast --model_index 8 \
--stats_dir results/mlp_results/batch_activations/opt-66b/ \
--mlp_ckpt_dir /pscratch/sd/s/<name>/HybridTensor/checkpoint/opt-66b-routers/mlp/ \
--act_data_dir /pscratch/sd/s/<name>/HybridTensor/opt-66b_act_data/ \
--batch_size 2 --delta 200 --device 2
```

## MHA 

Update the bash file in HybridTensor/routers/mha/train_mha_routers.sh 

```bash
./HybridTensor/routers/mha/train_mha_routers_topk.sh 
```
