import torch
import argparse

from HybridTensor.utils.utils import _get_device
from HybridTensor.utils.activations import MODELS
from HybridTensor.models.opt import build_sparse_opt 
from HybridTensor.models.llama import build_sparse_llama
from HybridTensor.routers.mlp.mlp_router_optim import load_router_dict_from_csv
from transformers import AutoTokenizer


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
    parser.add_argument('--mlp_ckpt_dir', type=str, default='/home/grads/s/<name>/nvme/HybridTensor/checkpoint/opt-6.7b-routers/mlp')
    parser.add_argument('--attn_ckpt_dir', type=str, default='/home/grads/s/<name>/nvme/HybridTensor/checkpoint/opt-6.7b-routers/mha_linear')
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
    input_text = "Once upon a time in a land far, far away, there lived a"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate output
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
