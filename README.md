# Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity

Polar Sparsity is a framework for efficient sparse inferencing in large language models (LLMs), leveraging custom Triton kernels and learned routers for selective activation of MLP neurons and attention heads. This repository provides tools for data collection, router training, benchmarking, and end-to-end sparse generation.

---

## 📦 Repository Structure

- **Router Data Collection & Training**
  - Data Collection: [`HybridTensor/routers/datacollection/data_collection.py`](HybridTensor/routers/datacollection/data_collection.py)
  - MLP Router Training: [`HybridTensor/routers/mlp/main_mlp.py`](HybridTensor/routers/mlp/main_mlp.py)
  - MHA Router Training: [`HybridTensor/routers/mha/main_att.py`](HybridTensor/routers/mha/main_att.py)
- **Benchmarks**
  - Evaluation: [`HybridTensor/benchmarks/model_eval.py`](HybridTensor/benchmarks/model_eval.py)
- **Kernel Implementations**
  - Triton Kernels: [`HybridTensor/triton/`](HybridTensor/triton/)
  - Example Runners: [`run_sparse_mlp.py`](run_sparse_mlp.py), [`run_sparse_attn.py`](run_sparse_attn.py), [`run_sparse_transformer_block.py`](run_sparse_transformer_block.py)
- **Sparse Generation**
  - End-to-End Sparse Generation: [`model_sparse_generation.py`](model_sparse_generation.py)

---

## 🚀 Getting Started

### 1. Environment Setup

- Install dependencies (see [`environment.yml`](environment.yml) for details).

```bash
conda env create -f environment.yml
```

- For Triton kernels, install the latest nightly build:
  ```bash
  pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
  ```

---

### 2. Router Data Collection

To collect router data for a specific model, you can use:

```bash
python -m HybridTensor.routers.datacollection.data_collection \
    --model_index 5 \
    --batch_size 8 \ 
    --device_map auto \
    --data_dir <PATH_TO_ACTIVATION_DATA> \
    --max_samples 400000 \
    --model_family <opt/llama> \
    --mlp_activation True \
    --attn_norm True
```

**Argument explanations:**

- `--model_index`: Index of the model to use (see `HybridTensor/utils/activations.py` for available indices).
- `--batch_size`: Number of samples per batch during data collection, adjust to configure GPU memory usage.
- `--data_dir`: Directory to save the collected activation data.
- `--model_family`: Model family (e.g., `opt`, `llama`).
- `--mlp_activation`: Set to `True` to collect MLP activation data. Only for sparse MLP models.
- `--attn_norm`: Set to `True` to collect attention norm data.

---

### 3. Router Training and Optimizations

**MLP Router:**

To run the MLP router training use the following scripts

For a single layer:

```bash
python -m HybridTensor.routers.mlp.main_mlp \
    --model_index <MODEL_INDEX> \
    --L <LAYER_NUMBER> \
    --data_dir <PATH_TO_ACTIVATION_DATA> \
    --ckpt_dir <PATH_TO_SAVE_CHECKPOINTS> \
    --gpu <GPU_ID>
```

For all layers, edit the [`HybridTensor/routers/mlp/train_mlp_routers.sh'](HybridTensor/routers/mlp/train_mlp_routers.sh) file with the number of GPUs available, model index, total number of layers, data_dir and ckpt_dir.  

```bash
./HybridTensor/routers/mlp/train_mlp_routers.sh
```

**MHA Router:**

To run the attention router training use the following scripts

For a single layer:

```bash
python -m HybridTensor.routers.mha.main_att \
    --model_index <MODEL_INDEX> \
    --L <LAYER_NUMBER> \
    --k <TOPK_VALUE> \
    --data_dir <PATH_TO_ACTIVATION_DATA> \
    --ckpt_dir <PATH_TO_SAVE_CHECKPOINTS>
```

For all layers, edit the [`/HybridTensor/routers/mha/train_mha_routers_topk.sh'](/HybridTensor/routers/mha/train_mha_routers_topk.sh) file with the number of GPUs available, model index, total number of layers, data_dir and ckpt_dir.  

```bash
./HybridTensor/routers/mha/train_mha_routers_topk.sh
```
To optimize the MLP layers for ReLU model with our dynamic layer wise top-k algorithm, you can use:



```bash
python -m HybridTensor.routers.mlp.mlp_router_optim_fast --model_index <MODEL_INDEX> --batch_size <BATCH_SIZE_INFERENCE> --mlp_ckpt_dir <PATH_TO_MLP_ROUTER_CHECKPOINTS> --act_data_dir <PATH_TO_ACTIVATION_DATA>

```
- `--batch_size`: batch size to optimize for inference

---

### 4. Model Evaluation

You can evaluate your models on various benchmarks using the [`HybridTensor/benchmarks/model_eval.py`](/HybridTensor/benchmarks/model_eval.py) script. Below are example commands and explanations for the main arguments. These scripts use huggingface implementations with masking for easy benchmarking. These do not use the optimized kernels for efficient inference.  

**Example usage:**

```bash
python -m HybridTensor.benchmarks.model_eval \
    --model_index <MODEL_INDEX> \
    --batch_size <BATCH_SIZE> \
    --mode <dense|sparse|sparse_attn> \
    --benchmark <all|BENCHMARK_NAME> \
    --attn_topk <TOPK_VALUE> \
    --attn_ckpt_dir <PATH_TO_ATTENTION_ROUTER_CHECKPOINTS> \
    --mlp_ckpt_dir <PATH_TO_MLP_ROUTER_CHECKPOINTS> \
    --data_collection <True|False> \
    --device auto \
    --note <NOTE>
```

**Additional argument explanations:**

- `--batch_size`: Batch size to use for evaluation.
- `--mode`: Evaluation mode. Options are `dense` (standard), `sparse` (sparse MLP and/or attention using trained routers), or `sparse_attn` (sparse attention only using ground truth activations ,doesn't require routers).
- `--benchmark`: Which benchmark(s) to run. Use `all` for the full suite or specify a single benchmark (e.g., `mmlu`).
- `--attn_topk`: Top-k value for attention sparsity (e.g., 0.5 for 50% sparsity).
- `--attn_ckpt_dir`: Directory containing attention router checkpoints.
- `--mlp_ckpt_dir`: Directory containing MLP router checkpoints.
- `--data_collection`: Set to `True` to enable data collection mode for threshold sweeps.
- `--device`: Device ID to use (e.g., `0` for `cuda:0`). 
- `--note`: Optional note to append to the results filename.


Adjust the arguments as needed for your experiment or hardware setup.

---

### 5. Kernel Implementations

**Triton Kernels:** Custom kernels for selective MLP and attention are in [`HybridTensor/triton/`](HybridTensor/triton/). 

Benchmark the speedup of the selective GEMM kernel (used for sparse MLPs):

```bash
python -m HybridTensor.triton.gather_gemm_col \
    --batch_size <BATCH_SIZE> \
    --in_features <EMBEDDING_DIMENSION> \
    --index_size <TOTAL_ACTIVE_NEURONS>
```

- `--in_features`: Model embedding dimension (e.g., 8192).
- `--index_size`: Total number of active neurons selected by the router. Needs to be less than or equal to total neurons.

---

Benchmark the speedup for a sparse MLP layer:

```bash
python run_sparse_mlp.py \
    --in_features <EMBEDDING_DIMENSION> \
    --batch_size <BATCH_SIZE> \
    --index_size <ACTIVE_NEURONS>
```

Benchmark the speedup for a sparse Multi-Head Attention (MHA) layer:

---


```bash
python run_sparse_attn.py \
    --in_features <EMBEDDING_DIMENSION> \
    --batch_size <BATCH_SIZE> \
    --seq_len <SEQUENCE_LENGTH> \
    --attn_topk <TOPK_VALUE>
```

- `--attn_topk`: Fraction of attention heads to keep active (e.g., 0.5 for 50%).

---

Use the following script before running autotune_configs.py

``` bash 
export TRITON_PRINT_AUTOTUNING="1" 
```

For models with sparse MLP, use the [`HybridTensor/triton/heuristics/autotune_configs.py`](HybridTensor/triton/heuristics/autotune_configs.py) script to compile the kernels for different batch sizes and activation to speedup inference. 

Benchmark the speedup for a full sparse transformer block with different batch sizes and sequence lengths:

```bash
python run_sparse_transformer_block.py \
    --in_features <EMBEDDING_DIMENSION> \
    --batch_size <BATCH_SIZE> \
    --seq_len <SEQUENCE_LENGTH> \
    --index_size <ACTIVE_NEURONS> \
    --attn_topk <TOPK_VALUE>
```

> **Note:**  
> The `run_sparse_transformer_block.py` script can also be used to simulate large-scale inferencing setups with large batch sizes and sequence lengths on a single GPU if multi-GPU system is not available, since only a single transformer layer is executed in this script. 


### 6. Sparse Generation

Run end-to-end sparse generation using trained routers. This example uses a simple example how to build the spase model for end-to-end generation using the optimized kernels and batched inference. 

```bash
python -m HybridTensor.benchmarks.generation.model_sparse_generation \
    --model_index <MODEL_INDEX> \
    --mlp_ckpt_dir <PATH_TO_MLP_ROUTER_CHECKPOINTS> \
    --attn_ckpt_dir <PATH_TO_ATTENTION_ROUTER_CHECKPOINTS> \
    --batch_stats_dir <PATH_TO_BATCH_STATS> \
    --attn_topk <TOPK_VALUE>
```
- `--batch_stats_dir`: used for sparse MLP models, path to the output from dynamic top-k optimization. Saved in configs/<model_name>



---

We have dedicated significant effort to developing Polar Sparsity as a practical and impactful tool for efficient large language model inference. We hope our approach offers both technical value and inspiration, and we sincerely appreciate your time and consideration in reviewing our work.