# python -m HybridTensor.benchmarks.model_perplexity --model_index 14 --batch_size 4 --max_length 512 --attn_th 1 --static_thresholds True

import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from hf_models.opt.modeling_sparse_opt_topk import SparseOPTForCausalLM as SparseOPTTopkAttn
from hf_models.llama.modeling_sparse_llama_mha_topk import SparseLlamaForCausalLM as SparseLlamaTopKAttn
from HybridTensor.utils.activations import ActivationThresholds, identify_model_type, MODELS, CONFIGS
from HybridTensor.utils.utils import extract_model_name, compute_perplexity
import argparse
from datasets import load_dataset
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


from HybridTensor.benchmarks.opt_attn_sparse_topk_perplexity import (_update_model_attn_thresholds,
                                                                    build_data_loader,
                                                                    compute_sparse_perplexity,
                                                                    compute_perplexity_data_collection,
                                                                    display_model_menu,
                                                                    _interactive_mode,
                                                                    arg_parser,
                                                                     )


results_dir = "results/activations"

def compute_attn_layer_sparsity(model_name, min_th, critical_th, attn_sparsity):
    # Get model configuration
    # model_name = MODELS[model_index - 1]
    model_config = CONFIGS[model_name]
    num_layers = model_config['num_layer']
    
    # Load the importance scores from the file specified in the configuration
    file_path = model_config['layer_imp']
    with open(file_path, 'r') as f:
        attn_layer_imp = json.load(f)
    layer_importance = attn_layer_imp['importance_scores']

    # Classify layers as critical or sparse
    critical_layers = [i for i, imp in enumerate(layer_importance) if imp >= critical_th]
    sparse_layers   = [i for i, imp in enumerate(layer_importance) if imp < critical_th]

    # Calculate total sparse importance and the attention value
    sum_sparse_importance = sum(layer_importance[i] for i in sparse_layers)
    attn_val = attn_sparsity * len(sparse_layers)

    # Compute the sparsity map per layer
    layer_sparsity_map = {}
    for layer_idx in range(num_layers):
        if layer_idx in critical_layers:
            layer_sparsity_map[layer_idx] = 1.0  # Fully dense for critical layers
        else:
            if sum_sparse_importance > 0:
                raw_fraction = (layer_importance[layer_idx] / sum_sparse_importance) * attn_val
            else:
                raw_fraction = attn_sparsity
            # Clamp the fraction between min_th and 1.0
            fraction = max(raw_fraction, min_th)
            fraction = min(fraction, 1.0)
            layer_sparsity_map[layer_idx] = fraction

    return layer_sparsity_map

def compute_average_activation(layer_sparsity_map):
    """
    Computes the average activation for each layer based on the sparsity map.
    """
    total_activation = 0.0
    for layer_idx, fraction in layer_sparsity_map.items():
        total_activation += fraction

    average_activation = total_activation / len(layer_sparsity_map)
    return average_activation

def compute_sparse_perplexity(model_name='facebook/opt-125m',
                            dataset_name='wikitext',
                            dataset_config='wikitext-2-raw-v1',
                            batch_size=8,
                            max_length=512,
                            attn_th=0.0,
                            static_thresholds=True,
                            device_map="cuda:0"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load the activation thresholds
    num_layers = CONFIGS[model_name]['num_layer']
    sp_thresholds = ActivationThresholds(num_layers=num_layers, attn_th=attn_th)
    
    print(f"Static attention activations: {sp_thresholds.activation_threshold}")
    if not static_thresholds:  
        # act_threshold_filepath = CONFIGS[model_name]['sp_config']
        attn_sparsity_map = compute_attn_layer_sparsity(model_name=model_name, min_th=0.2, critical_th=0.3, attn_sparsity=attn_th)
        sp_thresholds.load_thresholds(attn_sparsity_map)
        average_act = compute_average_activation(attn_sparsity_map)
        print(f"Layer imporatance weights attention activations {sp_thresholds.activation_threshold}")
        print(f"Average activation: {average_act:.2f}")
    
    # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_type = identify_model_type(model_name)
    if model_type == 'OPT':
        print(f"Loading OPT model: {model_name}")
        model = SparseOPTTopkAttn.from_pretrained(model_name, device_map = device_map, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
    elif model_type == 'Llama':
        print(f"Loading Llama model: {model_name}")
        model = SparseLlamaTopKAttn.from_pretrained(model_name, device_map = device_map, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
    model.eval()

    # # Load dataset
    dataloader = build_data_loader(model_name, dataset_name, dataset_config, batch_size, max_length)
    perplexity = compute_perplexity(model, dataloader, device)
    return perplexity


def arg_parser():
    parser = argparse.ArgumentParser(description='Sparse Perplexity Evaluation')
    parser.add_argument('--model_index', type=int, default=5, help='Index of the model to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--attn_th', type=float, default=0.0, help='Activation threshold for attention layers')
    parser.add_argument('--data_collection', type=bool, default=False, help='Collect data for different activation thresholds')
    parser.add_argument('--device_map', type=str, default='auto', help='Device to use for evaluation')
    parser.add_argument('--interactive', type=bool, default=False, help='Interactive mode for model selection')
    parser.add_argument('--static_thresholds', type=bool, default=False, help='Use static thresholds for attention layers')

    return parser.parse_args()

def main():
    """
    Main function to execute the perplexity computation with user-selected OPT model.
    """
    print("=== OPT Models Perplexity Evaluation ===\n")
    args = arg_parser()
    
    if args.interactive:
        selected_model, batch_size, max_length, data_collection, device_map, attn_th = _interactive_mode()
    
    else:
        selected_model, batch_size, max_length, data_collection, device_map, attn_th = MODELS[args.model_index-1], args.batch_size, args.max_length, args.data_collection, args.device_map, args.attn_th
        print(f"Selected model: {selected_model}, batch size: {batch_size}, max length: {max_length}, attn_th: {attn_th}, data_collection: {data_collection}, device: {device_map}")
        
    if data_collection:
        print("\nStarting data collection...\n")
        compute_perplexity_data_collection(model_name=selected_model, batch_size=batch_size, max_length=max_length, device_map=device_map)
        print("\nData collection complete.\n")
    
    else:
        print("\nStarting perplexity computation...\n")
        perplexity = compute_sparse_perplexity(model_name=selected_model, batch_size=batch_size, max_length=max_length,
                                               attn_th=attn_th,
                                               device_map=device_map,
                                               static_thresholds=args.static_thresholds)
        print(f"\n=== Perplexity Results ===")
        print(f"Model: {selected_model}")
        print(f"Perplexity: {perplexity:.2f}\n")

if __name__ == "__main__":
    main()