from transformers.models.opt import OPTConfig
from transformers import AutoTokenizer
from flash_attn.models.opt import opt_config_to_gpt2_config

import os
import torch
import argparse
from apex.transformer import parallel_state

from HybridTensor.utils.utils import arg_parser, _get_device
from HybridTensor.utils.activations import OPT_MODELS
from HybridTensor.models.opt import SparseConfig, build_sparse_opt 

def update_router_config(model, num_layers, mlp_act_th, attn_topk, layer_config = None):
    for i in range(num_layers):
        model.transformer.layers[i].mlp_router.act_th = mlp_act_th
        model.transformer.layers[i].mha_router.topk = attn_topk

def initialize_distributed_environment():
    # Set environment variables for NCCL
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

    # Initialize the distributed process group
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Set the device based on the rank of the current process
    device = f"cuda:{torch.distributed.get_rank()}"
    world_size = torch.distributed.get_world_size()

    # Set the current CUDA device to avoid operations being executed on the wrong GPU
    torch.cuda.set_device(device)

    # You can return device, world_size, and any other relevant information
    return device, world_size


def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=28)
    parser.add_argument('--index_size', type=int, default=8192)
    parser.add_argument('--head_density', type=float, default=0.25)
    parser.add_argument('--print_results', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--check_results', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_ROUTER_CHECKPOINTS>')
    parser.add_argument('--attn_topk', type=float, default=0.5, help='Attention topk for sparse model')
    parser.add_argument('--attn_ckpt_dir', type=str, default='<PATH_TO_ATTENTION_CHECKPOINTS>')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = arg_parser()
    model_name = OPT_MODELS[args.model_index-1]

    device, world_size = initialize_distributed_environment()
    dtype = torch.float16
    
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_sparse_opt(model_name, args.mlp_ckpt_dir, args.attn_ckpt_dir, device = device, dtype=dtype, process_group = process_group, world_size = world_size, rank = rank)
    model.eval()
    print("Model loaded with sparse routers")
    
    mlp_act_th = 0.5
    attn_topk = 0.5
    
    update_router_config(model, model.config.n_layer, mlp_act_th, attn_topk)
    print("Router config updated")
    
    # print router configs from all layers
    # for i in range(model.config.n_layer):
    #     print(f"Layer {i}: mlp_act_th = {model.transformer.layers[i].mlp_router.act_th}, attn_topk = {model.transformer.layers[i].mha_router.topk}")
    
    input_texts = ["Hello, my dog is cute and", "The future of AI is", "In a distant galaxy, a spaceship", "The cat is sleeping on the "]
    # input_texts = ["Hello, my dog is cute and", "Hello, my rat is cute and"]
    
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids=tokenized_inputs["input_ids"]
    
    # input_ids = tokenizer("Hello, my dog is cute and", return_tensors="pt").input_ids.to(device=device)
    max_length = args.seq_len
    position_ids = None
    eos_token_id = tokenizer.eos_token_id
    num_layers = model.config.n_layer
    
    # print all the model weights and check the accuracy
    # if rank == 0:
    #     print(model.state_dict())

    # out = model(input_ids)
    # print(out)

    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=True,
        )
    if rank == 0:
        print(tokenizer.batch_decode(out.sequences.tolist()))
        
