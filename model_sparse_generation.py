# python model_sparse_generation.py --model_index 11 --attn_ckpt_dir  <PATH_TO_ATTENTION_ROUTER_CHECKPOINTS> --attn_topk 0.5 
# python model_sparse_generation.py --model_index 5 --mlp_ckpt_dir <PATH_TO_MLP_ROUTER_CHECKPOINTS> --attn_ckpt_dir <PATH_TO_ATTENTION_ROUTER_CHECKPOINTS> --batch_stats_dir configs/mlp_router/opt-6.7b --attn_topk 0.5 

import torch
import argparse

from HybridTensor.utils.utils import _get_device
from HybridTensor.utils.activations import MODELS
from HybridTensor.models.opt import SparseConfig, build_sparse_opt 

from HybridTensor.models.llama import build_sparse_llama
from HybridTensor.benchmarks.generation.gen_util import tokenize_dataset, get_random_batch
from HybridTensor.utils.activations import build_mlp_topk_lookup
from HybridTensor.routers.mlp.mlp_router_optim import load_router_dict_from_csv

from datasets import load_dataset

from transformers.models.opt import OPTConfig
from transformers import AutoTokenizer
from flash_attn.models.opt import opt_config_to_gpt2_config
from flash_attn.utils.generation import update_graph_cache

def update_router_config(model, num_layers, mlp_topk_lookup, attn_topk):
    for i in range(num_layers):
        if mlp_topk_lookup is not None:
            model.transformer.layers[i].mlp_topk = mlp_topk_lookup[i]
        # model.transformer.layers[i].mlp_topk = 512
        model.transformer.layers[i].mha_router.topk = attn_topk
    
    # dense attention in layer 0
    model.transformer.layers[0].mha_router.topk = 1.0

def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--print_results', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--attn_topk', type=float, default=0.5, help='Attention topk for sparse model')
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_ROUTER_CHECKPOINTS>')
    parser.add_argument('--attn_ckpt_dir', type=str, default='<PATH_TO_ATTENTION_ROUTER_CHECKPOINTS>')
    parser.add_argument('--batch_stats_dir', type=str, default='configs/mlp_router/opt-6.7b')
    parser.add_argument('--delta', type=int, default=256, help='Delta value for MLP topk calculation')
    parser.add_argument('--use_cuda_graph', type=bool, default=False, help='Use CUDA graph for inference')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = arg_parser()
    model_name = MODELS[args.model_index-1]
    print(f"Model name: {model_name}")
    dtype = torch.float16
    device= _get_device(args.gpu)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if "llama" in model_name.lower():
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.pad_token_id
        
    if "llama" in model_name:
        model = build_sparse_llama(args, model_name,
                               args.attn_ckpt_dir,
                               device = device, dtype=dtype)
        update_router_config(model, model.config.n_layer, None, args.attn_topk)  # this sets the router config for all layers using a single config
        
    else:
        mlp_topk_lookup = load_router_dict_from_csv(args.batch_stats_dir, args.batch_size)
        model = build_sparse_opt(args, model_name,
                               args.mlp_ckpt_dir,
                               args.attn_ckpt_dir,
                               device = device, dtype=dtype)
        update_router_config(model, model.config.n_layer, mlp_topk_lookup, args.attn_topk)  # this sets the router config for all layers using a single config
        

    model.eval()
    print(model)
    
    # test input
    input_texts = ["The future of AI is", "In a distant galaxy,"] 
    # input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=False).input_ids.to(device)
    encoding = tokenizer(
        input_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=False # Or True with a specified max_length if inputs can be long
    )
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device) # Crucial: get the attention mask
    position_ids = None
    eos_token_id = tokenizer.eos_token_id
    return_dict_in_generate=True,
    max_length = 50
    
    # Generate output
    with torch.no_grad():
        out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        cg=False,
        )
    
    print(tokenizer.batch_decode(out.sequences.tolist()))
