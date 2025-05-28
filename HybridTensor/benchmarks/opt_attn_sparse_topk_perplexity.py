import sys

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from hf_models.opt.modeling_sparse_opt import SparseOPTForCausalLM
from hf_models.opt.modeling_sparse_opt_topk import SparseOPTForCausalLM
from HybridTensor.utils.activations import ActivationThresholds, MODELS, CONFIGS
from HybridTensor.utils.utils import extract_model_name, compute_perplexity

import argparse
from datasets import load_dataset

from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

results_dir = "results/activations"

def _update_model_attn_thresholds(model, attn_th, mode='sparse'):
    num_layers = model.config.num_hidden_layers

    # Use the 'decoder' attribute if it exists; otherwise use model.model.layers
    layers = model.model.decoder.layers if hasattr(model.model, 'decoder') else model.model.layers

    for i in range(num_layers):
        layers[i].self_attn.sp_threshold = attn_th

    # For non-sparse attention, layer 0 should use a threshold of 1.0
    # if mode != 'sparse_attn':
    #     layers[0].self_attn.sp_threshold = 1.0
    layers[0].self_attn.sp_threshold = 1.0
    
    return model


def build_data_loader(model_name, dataset_name, dataset_config, batch_size, max_length, split='test'):
    """
    Build a DataLoader for the specified dataset.

    Args:
    - model_name (str): The Hugging Face identifier of the model.
    - dataset_name (str): The name of the dataset.
    - dataset_config (str): The configuration of the dataset.
    - batch_size (int): The batch size for the DataLoader.
    - max_length (int): The maximum sequence length.
    - split (str): The split of the dataset to use (default='test'). options: 'train', 'validation', 'test'

    Returns:
    - dataloader (DataLoader): The DataLoader for the specified dataset.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.filter(lambda x: len(x["text"]) >= max_length)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Create DataLoader
    def collate_fn(batch):
        input_ids = [torch.tensor(example['input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['attention_mask']) for example in batch]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader

def compute_sparse_perplexity(model_name='facebook/opt-125m', dataset_name='wikitext', dataset_config='wikitext-2-raw-v1', batch_size=8, max_length=512, attn_th=0.0, static_thresholds=True, device_map="cuda:0"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load the activation thresholds
    num_layers = CONFIGS[model_name]['num_layer']
    
    sp_thresholds = ActivationThresholds(num_layers=num_layers, attn_th=attn_th)
    
    if not static_thresholds:  
        act_threshold_filepath = CONFIGS[model_name]['sp_config']
        sp_thresholds.load_thresholds(act_threshold_filepath)
        print(f'Activation thresholds loaded from {act_threshold_filepath}')
        
    print(sp_thresholds.activation_threshold)
    
    # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = SparseOPTForCausalLM.from_pretrained(model_name, device_map = device_map, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
    model.eval()

    # # Load dataset
    dataloader = build_data_loader(model_name, dataset_name, dataset_config, batch_size, max_length)

    perplexity = compute_perplexity(model, dataloader, device)
    
    return perplexity

def compute_perplexity_data_collection(model_name='facebook/opt-125m', dataset_name='wikitext', dataset_config='wikitext-2-raw-v1', batch_size=8, max_length=512, device_map="cuda:0"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split='test')
    dataset = dataset.filter(lambda x: len(x["text"]) >= 512)
    dataloader = build_data_loader(model_name, dataset_name, dataset_config, batch_size, max_length)
    
    attn_thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # attn_thresholds = [1, 0.5, 0.2]
    
    print(f"Computing perplexity for the following attention thresholds: {attn_thresholds}")

    # load the model
    num_layers = CONFIGS[model_name]['num_layer']
    
    sp_thresholds = ActivationThresholds(num_layers=num_layers, attn_th=0.1)
    model = SparseOPTForCausalLM.from_pretrained(model_name, device_map = device_map, torch_dtype=torch.float16, sp_thresholds = sp_thresholds.activation_threshold, attn_implementation="flash_attention_2")
    model.eval()
    
    results = []
    for attn_th in attn_thresholds:
        
        print(f'Computing perplexity for attn top k: {attn_th}')
        
        # update the model with new threshold
        model = _update_model_attn_thresholds(model, attn_th)
        
        # compute and store the perplexity
        perplexity = compute_perplexity(model, dataloader, device)
        print(f'Perplexity: {perplexity:.2f}\n')
        results.append({
            'model': model_name,
            'top_k': attn_th,
            'perplexity': perplexity
        })
        
    
    # save the results to a csv file
    results_df = pd.DataFrame(results)
    model_name_str = extract_model_name(model_name)
    
    # save the results to a csv file in the results directory
    results_df.to_csv(f'{results_dir}/sparse_perplexity_results_{model_name_str}_topk.csv', index=False)
    

def display_model_menu():
    """
    Displays a numbered menu of available OPT models and prompts the user to make a selection.

    Returns:
    - selected_model (str): The Hugging Face identifier of the selected model.
    """
    print("Available OPT Models:")
    for idx, model in enumerate(MODELS, 1):
        print(f"{idx}. {model}")

    while True:
        try:
            choice = input("\nEnter the number corresponding to the model you want to evaluate (e.g., 1): ")
            if choice.lower() in ['q', 'quit', 'exit']:
                print("Exiting the program.")
                sys.exit(0)
            choice = int(choice)
            if 1 <= choice <= len(MODELS):
                selected_model = MODELS[choice - 1]
                print(f"\nYou have selected: {selected_model}\n")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(MODELS)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def _interactive_mode():
    selected_model = display_model_menu()

    # Optional: Allow user to adjust batch size and max sequence length
    try:
        batch_size_input = input("Enter batch size (default=8): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 8
    except ValueError:
        print("Invalid input for batch size. Using default value of 8.")
        batch_size = 8

    max_length = 512
        
    try:
        data_collection = input("Do you want to collect data for different activation thresholds? (y/n): ").strip()
        data_collection = True if data_collection.lower() == 'y' else False
    except ValueError:
        print("Invalid input for data collection. Using default value of False.")
        data_collection = False
        
    
    # select device
    device_map = input("Enter device (cuda:0/auto) [default=cuda:0]: ").strip()
    if not device_map:
        device_map = "cuda:0"
        
    # select attention threshold
    attn_th = 0.0
    if not data_collection:
        try:
            attn_th = input("Enter activation threshold for attention layers: ").strip()
            attn_th = float(attn_th) if attn_th else 0.0
        except ValueError:
            print("Invalid input for attention threshold. Using default value of 0.")
            attn_th = 0.0
    
    return selected_model, batch_size, max_length, data_collection, device_map, attn_th
    

def arg_parser():
    parser = argparse.ArgumentParser(description='Sparse Perplexity Evaluation')
    parser.add_argument('--model_index', type=int, default=5, help='Index of the model to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--attn_th', type=float, default=0.0, help='Activation threshold for attention layers')
    parser.add_argument('--data_collection', type=bool, default=False, help='Collect data for different activation thresholds')
    parser.add_argument('--device_map', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--interactive', type=bool, default=False, help='Interactive mode for model selection')

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
        perplexity = compute_sparse_perplexity(model_name=selected_model, batch_size=batch_size, max_length=max_length, attn_th=attn_th, device_map=device_map)
        print(f"\n=== Perplexity Results ===")
        print(f"Model: {selected_model}")
        print(f"Perplexity: {perplexity:.2f}\n")

if __name__ == "__main__":
    main()