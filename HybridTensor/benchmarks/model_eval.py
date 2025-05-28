import torch
import argparse
import os
import json
import logging
import numpy as np
import csv

# from hf_models.opt.modeling_opt_routers import (
#     SparseOPTForCausalLM,
#     create_hf_mha_router_state_dict,
#     create_hf_mlp_router_state_dict
# )

from hf_models.opt.modeling_opt_routers_topk import (
    SparseOPTForCausalLM,
    create_hf_mha_router_state_dict,
    create_hf_mlp_router_state_dict
)

from hf_models.llama.modeling_sparse_llama_routers import (
    SparseLlamaForCausalLM,
    create_hf_attn_router_state_dict
)

from hf_models.opt.modeling_sparse_opt_topk import SparseOPTForCausalLM as SparseOPTTopKAttn
from hf_models.llama.modeling_sparse_llama_mha_topk import SparseLlamaForCausalLM as SparseLlamaTopKAttn
from HybridTensor.benchmarks.opt_attn_sparse_topk_perplexity import _update_model_attn_thresholds
from HybridTensor.benchmarks.model_perplexity import compute_attn_layer_sparsity, compute_average_activation
from HybridTensor.utils.activations import ActivationThresholds, build_mlp_topk_lookup, _update_hf_mlp_topk, CONFIGS, MODELS
from HybridTensor.routers.mlp.mlp_router_optim import load_router_dict_from_csv
from HybridTensor.utils.utils import extract_model_name

from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import lm_eval

import pandas as pd
from tabulate import tabulate


import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from huggingface_hub import login

def read_and_print_results(filepath='results.csv'):
    """
    Reads the CSV file containing evaluation results and prints them in a formatted table.
    """
    if not os.path.exists(filepath):
        print(f"File '{filepath}' not found.")
        return
    
    df = pd.read_csv(filepath)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def save_results_to_csv(results, attn_topk, filepath='eval_results.csv'):
    """
    Extracts benchmark accuracies from results and saves them along with the attn_topk config.
    
    Parameters:
      results: dict, evaluation results with structure results['results'][<benchmark>]['acc,none']
      attn_topk: float, the attention top-k value used for this run
      filepath: str, CSV file to write to (appends if it exists)
    """
    # Build a dictionary row with attn_topk and each benchmark's accuracy
    row = {'attn_topk': attn_topk}
    for benchmark, data in results['results'].items():
        # Default to None if the key is missing
        row[benchmark] = data.get('acc,none', None)
    
    # Check if file exists to decide on writing header
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _update_model_attn_sparsity(model, attn_th):
    num_layers = model.config.num_hidden_layers

    # Use the 'decoder' attribute if it exists; otherwise use model.model.layers
    layers = model.model.decoder.layers if hasattr(model.model, 'decoder') else model.model.layers
    attn_sparsity_map = compute_attn_layer_sparsity(model_name=model_name, min_th=0.2, critical_th=0.3, attn_sparsity=attn_th)
    
    for i in range(num_layers):
        layers[i].self_attn.sp_threshold = attn_sparsity_map[i]
    
    average_act = compute_average_activation(attn_sparsity_map)
    print(f"Attention sparsity {attn_th}: {attn_sparsity_map}")
    print(f"Average activation: {average_act:.2f}")
    
    return model

def _evaluate_model(model, tokenizer, benchmarks: list, device: str, batch_size: int = 8):
    logging.info("Evaluating on benchmarks: %s", benchmarks)
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size
    )
    task_manager = TaskManager()
    num_fewshot = 5
    print(f"Number of fewshot examples: {num_fewshot}")
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=benchmarks,
        num_fewshot=num_fewshot,  # change this 
        task_manager=task_manager
    )
    logging.info("Evaluation complete.")
    for benchmark, benchmark_results in results['results'].items():
        logging.info("Results for %s: %s", benchmark.upper(), benchmark_results)
    return results

def _load_model(model_name, num_layers, device, args):
    if args.mode == 'sparse':
        logging.info("Loading sparse model...")
        sp_thresholds = ActivationThresholds(num_layers=num_layers, attn_th= args.attn_topk, mlp_th=args.mlp_topk)
        
        if args.model_index <=8:
            # OPT models
            model = SparseOPTForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16,
                sp_thresholds=sp_thresholds.activation_threshold,
                mlp_thresholds=sp_thresholds.mlp_threshold,
                attn_implementation="flash_attention_2"
            )
            logging.info("Loading router states...")
            mlp_router_state = create_hf_mlp_router_state_dict(args.mlp_ckpt_dir)
            mha_router_state = create_hf_mha_router_state_dict(args.attn_ckpt_dir)
            model_state = model.state_dict()
            model_state.update(mlp_router_state)
            model_state.update(mha_router_state)
            model.load_state_dict(model_state)
            logging.info("Sparse model loaded with routers!")
            
            # load topk values for mlp and attn here
            # mlp_topk_lookup = build_mlp_topk_lookup(args.batch_stats_dir, args.batch_size, args.delta)
            # mlp_topk_lookup = build_mlp_topk_lookup(args.batch_stats_dir, 1, args.delta)
            mlp_topk_lookup = load_router_dict_from_csv(args.batch_stats_dir, 1)
            
            _update_hf_mlp_topk(model, mlp_topk_lookup)
            # print("MLP topk values updated.")
            # print("MLP topk values: ", mlp_topk_lookup)
            logging.info("Using MLP topk values: %s", mlp_topk_lookup)
            
            # print("Using delta value: ", args.delta)
            
            # the first layer should use dense attention
            model.model.decoder.layers[0].self_attn.sp_threshold = 1.0
        else:
            # Llama models
            
            if not args.static_thresholds:
                attn_sparsity_map = compute_attn_layer_sparsity(model_name=model_name, min_th=0.2, critical_th=0.3, attn_sparsity=args.attn_topk)
                sp_thresholds.load_thresholds(attn_sparsity_map)
                average_act = compute_average_activation(attn_sparsity_map)
                print(f"Layer imporatance weights attention activations {sp_thresholds.activation_threshold}")
                print(f"Average activation: {average_act:.2f}")
            
            model = SparseLlamaForCausalLM.from_pretrained(model_name,
                                                           device_map = device,
                                                           torch_dtype=torch.float16,
                                                            sp_thresholds = sp_thresholds.activation_threshold, 
                                                            attn_implementation="flash_attention_2")
            logging.info("Loading router states...")
            model_state = model.state_dict()
            attn_router_states = create_hf_attn_router_state_dict(args.attn_ckpt_dir)
            model_state.update(attn_router_states)
            model.load_state_dict(model_state)
            logging.info("Sparse model loaded with routers!")
            
            # the first layer should use dense attetnion
            _update_model_attn_thresholds(model, args.attn_topk)

        # load topk values for mha here
        # TODO: create a function to update the topk values for mha
    
    elif args.mode == 'sparse_attn':
        logging.info("Loading model with sparse attention")
        sp_thresholds = ActivationThresholds(num_layers=num_layers, attn_th=args.attn_topk)
        
        if not args.static_thresholds:
            attn_sparsity_map = compute_attn_layer_sparsity(model_name=model_name, min_th=0.2, critical_th=0.3, attn_sparsity=args.attn_topk)
            sp_thresholds.load_thresholds(attn_sparsity_map)
            average_act = compute_average_activation(attn_sparsity_map)
            print(f"Layer imporatance weights attention activations {sp_thresholds.activation_threshold}")
            print(f"Average activation: {average_act:.2f}")
        
        if args.model_index <= 8:
            # opt models
            model = SparseOPTTopKAttn.from_pretrained(model_name, device_map = device, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
        else:
            # llama models
            model = SparseLlamaTopKAttn.from_pretrained(model_name, device_map = device, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
    else:
        logging.info("Loading dense model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.float16)
    return model

def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--print_results', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='results/eval')
    parser.add_argument('--device', type=int, default=100)
    parser.add_argument('--mode', type=str, default='sparse', choices=['sparse', 'dense', 'sparse_attn'])
    parser.add_argument('--attn_topk', type=float, default=0.5, help='Attention topk for sparse model')
    parser.add_argument('--mlp_topk', type=int, default=2048, help='MLP topk for sparse model')
    parser.add_argument('--delta', type=int, default=128, help='Delta value for MLP topk calculation')
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_ROUTER_CHECKPOINTS>')
    parser.add_argument('--attn_ckpt_dir', type=str, default='<PATH_TO_ATTENTION_CHECKPOINTS>')
    parser.add_argument('--batch_stats_dir', type=str, default='configs/mlp_router')
    parser.add_argument('--data_collection', type=bool, default=False, help='Collect data for different activation thresholds')
    parser.add_argument('--benchmark', type=str, default='all', help='Options: all, or a single benchmark name')
    parser.add_argument('--note', type=str, default='', help='Note to add to the results filename')
    parser.add_argument('--static_thresholds', type=bool, default=True, help='Use static thresholds for attention layers')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    
    login_token = None # insert your token here
    assert login_token is not None, "Please provide a valid Hugging Face token."
    login(token=login_token)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model_name = MODELS[args.model_index - 1]
    # print(f"Evaluating Model: {model_name}")
    logging.info("Evaluating Model: %s", model_name)
    logging.info("Mode: %s", args.mode)
    
    num_layers = CONFIGS[model_name]['num_layer']
    device = 'auto' if args.device == 100 else f'cuda:{args.device}'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = _load_model(model_name, num_layers, device, args)
    model.eval()
    
    # Determine benchmarks to evaluate
    if args.benchmark == 'all':
        benchmarks = ["piqa", "winogrande", "copa", "rte", "openbookqa", "arc_easy", "arc_challenge", "mmlu", "hellaswag"]
    else:   
        benchmarks = [args.benchmark]
    
    model_name_clean = extract_model_name(model_name)
    
    if args.data_collection:
        # make sure the model is not dense
        assert args.mode != 'dense', "Data collection is only available for sparse models"
        logging.info("Data collection mode enabled.")
        if args.mode == 'sparse':
            filepath = f"{args.results_dir}/eval_results_{model_name_clean}_sparse_sweep_dpsd.csv"
        else:   # sparse_attn
            filepath = f"{args.results_dir}/eval_results_{model_name_clean}_attn_sweep_dpsd.csv"
        
        if args.note != '':
            filepath = filepath.replace('.csv', f"_{args.note}.csv")
        # attn_topk_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # MHA
        attn_topk_values = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1] 
        # attn_topk_values = [7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8] # GQA
        for attn_topk in attn_topk_values:
            logging.info("Evaluating with attention top-k value: %s", attn_topk)
            if args.static_thresholds:
                _update_model_attn_thresholds(model, attn_topk, mode=args.mode)
            else:
                _update_model_attn_sparsity(model, attn_topk)
            
            results = _evaluate_model(
                model=model,
                tokenizer=tokenizer,
                benchmarks=benchmarks,
                device=device,
                batch_size=args.batch_size
            )
            save_results_to_csv(results, attn_topk, filepath = filepath)
    else:
        logging.info("Evaluating with attention top-k value: %s", args.attn_topk)
        if args.mode == 'dense':
            filepath = f"{args.results_dir}/eval_results_{model_name_clean}_dense.csv"
        elif args.mode == 'sparse_attn':
            filepath = f"{args.results_dir}/eval_results_{model_name_clean}_sparse_attn_{args.attn_topk}_dpsd.csv"
        else:
            filepath = f"{args.results_dir}/eval_results_{model_name_clean}_test_attn_{args.attn_topk}_dpsd.csv"
        if args.note != '':
            filepath = filepath.replace('.csv', f"_{args.note}.csv")
        results = _evaluate_model(
            model=model,
            tokenizer=tokenizer,
            benchmarks=benchmarks,
            device=device,
            batch_size=args.batch_size
        )
        save_results_to_csv(results, args.attn_topk, filepath = filepath)
        
    if args.print_results:
        read_and_print_results(filepath=filepath)