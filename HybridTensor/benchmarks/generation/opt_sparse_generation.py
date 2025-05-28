import torch
import argparse

from HybridTensor.utils.utils import _get_device
from HybridTensor.utils.activations import OPT_MODELS
from HybridTensor.models.opt import SparseConfig, build_sparse_opt 
from HybridTensor.benchmarks.generation.gen_util import tokenize_dataset, get_random_batch
from HybridTensor.utils.activations import build_mlp_topk_lookup
from HybridTensor.routers.mlp.mlp_router_optim import load_router_dict_from_csv

from datasets import load_dataset

from transformers.models.opt import OPTConfig
from transformers import AutoTokenizer
from flash_attn.models.opt import opt_config_to_gpt2_config
from flash_attn.utils.generation import update_graph_cache

def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--index_size', type=int, default=8192)
    parser.add_argument('--head_density', type=float, default=0.5)
    parser.add_argument('--print_results', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--check_results', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--attn_topk', type=float, default=0.5, help='Attention topk for sparse model')
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_ROUTER_CHECKPOINTS>')
    parser.add_argument('--attn_ckpt_dir', type=str, default='<PATH_TO_ATTENTION_CHECKPOINTS>')
    parser.add_argument('--batch_stats_dir', type=str, default='configs/mlp_router/opt-6.7b')
    parser.add_argument('--delta', type=int, default=256, help='Delta value for MLP topk calculation')
    parser.add_argument('--use_cuda_graph', type=bool, default=False, help='Use CUDA graph for inference')

    return parser.parse_args()

def update_router_config(model, num_layers, mlp_topk_lookup, attn_topk):
    for i in range(num_layers):
        model.transformer.layers[i].mlp_topk = mlp_topk_lookup[i]
        # model.transformer.layers[i].mlp_topk = 512
        model.transformer.layers[i].mha_router.topk = attn_topk
        
        # model.transformer.layers[i].skip_mlp_router = True
    model.transformer.layers[0].mha_router.topk = 1.0  # dense attention in layer 0

if __name__ == "__main__":
    args = arg_parser()
    model_name = OPT_MODELS[args.model_index-1]
    dtype = torch.float16
    device= _get_device(args.gpu)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # args.mlp_ckpt_dir = None
    # args.attn_ckpt_dir = None
    
    model = build_sparse_opt(args, model_name, args.mlp_ckpt_dir, args.attn_ckpt_dir, device = device, dtype=dtype)
    model.eval()
    print(model)
    print("Model loaded with sparse routers")
    
    # mlp_topk_lookup = build_mlp_topk_lookup("results/mlp_results/batch_activations/opt-6.7b", args.batch_size, args.delta)
    mlp_topk_lookup = load_router_dict_from_csv(args.batch_stats_dir, args.batch_size)
    print("MLP topk values updated: ", mlp_topk_lookup)
    update_router_config(model, model.config.n_layer, mlp_topk_lookup, args.attn_topk)  # this sets the router config for all layers using a single config
    # update_router_config(model, model.config.n_layer, 2048, args.attn_topk)
    print("Router config updated \n")
    
    
    max_length = args.seq_len + 20
    batch_size = args.batch_size
    
    # input_texts = ["Hello, my dog is cute and", "The future of AI is", "In a distant galaxy, a spaceship", "The cat is sleeping on the "]
    # input_texts = ["In a distant galaxy, a spaceship"]
    # tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=False).to(device)
    # input_ids=tokenized_inputs["input_ids"]
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenize_dataset(dataset, tokenizer)
    input_ids = get_random_batch(tokens, args.batch_size, args.seq_len).to(device)
    
    print("Input ids generated, starting inference")
    
    # input_ids = tokenizer("Hello, my dog is cute and he", return_tensors="pt").input_ids.to(device)
    position_ids = None
    eos_token_id = tokenizer.eos_token_id
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        # warm up
        _ = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            cg=False,
            )
        
        print("Warm up done")
        
        start_event.record()
        for i in range(args.iterations):
            out = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                cg=False,
                )
        
        end_event.record()
        
        torch.cuda.synchronize()
        print("Without CUDA graph")
        elapsed_time = start_event.elapsed_time(end_event) / args.iterations
        print(f"Average time per genearation : {elapsed_time:.1f} ms")
        
        # Compute throughput and latency per token
        num_tokens_generated = out.sequences.shape[1] - input_ids.shape[1]
        throughput = batch_size * num_tokens_generated / (elapsed_time / 1000)  # tokens per second
        latency_per_token = elapsed_time / num_tokens_generated  # ms per token

        print(f"Number of tokens generated: {num_tokens_generated}")
        print(f"Throughput: {throughput:.1f} tokens/second")
        print(f"Latency per token: {latency_per_token:.1f} ms")

        # print(tokenizer.batch_decode(out.sequences.tolist()))
        print("\n")
        
        # print only the new tokens generated 
        print("New tokens generated:")
        print(tokenizer.batch_decode(out.sequences[:, input_ids.shape[1]:].tolist()))
        
        # ====================== With CUDA graph ======================
        if args.use_cuda_graph:
            batch_size, seqlen_og = input_ids.shape
            model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
            print("With CUDA graph")
            torch.cuda.synchronize()
            
            start_event.record()
            
            for i in range(args.iterations):
                out = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    cg=True,
                    eos_token_id=eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    enable_timing=False,
                    )
            
            end_event.record()
            
            torch.cuda.synchronize()
            
            
            elapsed_time = start_event.elapsed_time(end_event) / args.iterations
            print(f"Average time per genearation : {elapsed_time:.1f} ms")
            
            # Compute throughput and latency per token
            num_tokens_generated = out.sequences.shape[1] - input_ids.shape[1]
            throughput = batch_size * num_tokens_generated / (elapsed_time / 1000)  # tokens per second
            latency_per_token = elapsed_time / num_tokens_generated  # ms per token

            print(f"Number of tokens generated: {num_tokens_generated}")
            print(f"Throughput: {throughput:.1f} tokens/second")
            print(f"Latency per token: {latency_per_token:.1f} ms")

            # print(tokenizer.batch_decode(out.sequences.tolist()))
            print("New tokens generated:")
            print(tokenizer.batch_decode(out.sequences[:, input_ids.shape[1]:].tolist()))
    