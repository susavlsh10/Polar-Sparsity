import torch
import numpy as np
import os

from hf_models.opt.modeling_opt import OPTForCausalLM
from hf_models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

# from tests.activation_tests.eval_sparse_perplexity import OPT_MODELS, build_data_loader
from HybridTensor.benchmarks.opt_attn_sparse_topk_perplexity import build_data_loader
from HybridTensor.utils.utils import extract_model_name

from datasets import load_dataset
import json

from tqdm import tqdm
import argparse
from HybridTensor.utils.activations import MODELS


def load_layer_data(data_dir, layer_idx, data_type):
    """
    Load data for a specific layer and data type.

    Args:
        data_dir (str): Directory where data is stored.
        layer_idx (int): Layer index.
        data_type (str): One of 'hidden_states', 'mlp_activations', 'attn_norms'.

    Returns:
        np.ndarray: The data array of shape (num_samples, feature_size).
    """
    # Load metadata
    metadata_filename = os.path.join(data_dir, 'metadata.json')
    with open(metadata_filename, 'r') as f:
        metadata = json.load(f)

    num_layers = metadata['num_layers']
    hidden_size = metadata['hidden_size']
    num_heads = metadata['num_heads']
    max_samples = metadata['max_samples']

    # Validate layer index
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be between 0 and {num_layers - 1}.")

    # Get the sample count and feature size
    if data_type == 'hidden_states':
        sample_counts = metadata['hidden_states_counters']
        sample_count = sample_counts[layer_idx]
        feature_size = hidden_size
    elif data_type == 'mlp_activations':
        sample_counts = metadata['mlp_activations_counters']
        sample_count = sample_counts[layer_idx]
        feature_size = hidden_size * 4
    elif data_type == 'attn_norms':
        sample_counts = metadata['attn_norms_counters']
        sample_count = sample_counts[layer_idx]
        feature_size = num_heads
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'hidden_states', 'mlp_activations', or 'attn_norms'.")

    # Load the data file
    filename = os.path.join(data_dir, f'{data_type}_layer_{layer_idx}.dat')
    data_mmap = np.memmap(filename, dtype='float16', mode='r', shape=(max_samples, feature_size))

    # Return only the valid data
    data = np.array(data_mmap[:sample_count])
    del data_mmap  # Close the memmap
    return data

def initialize_data_structures(data_dir, num_layers, hidden_size, num_heads, num_neurons, max_samples,
                               mlp_activation=True, attn_norm=True):
    """
    Initialize mmap files and counters for hidden_states, mlp_activations, and attn_norms.

    Args:
        data_dir (str): Directory where data is stored.
        num_layers (int): Number of transformer layers.
        hidden_size (int): Hidden size of the model.
        num_heads (int): Number of attention heads.
        max_samples (int): Maximum number of samples to collect.

    Returns:
        tuple: Contains lists of mmap files and counters for each data type.
    """
    # Initialize lists to hold mmap files and counters
    hidden_states_files = []
    mlp_activations_files = []
    attn_norms_files = []

    hidden_states_counters = []
    mlp_activations_counters = []
    attn_norms_counters = []

    for layer_idx in range(num_layers):
        # Hidden states
        hs_filename = os.path.join(data_dir, f'hidden_states_layer_{layer_idx}.dat')
        hs_file = np.memmap(hs_filename, dtype='float16', mode='w+', shape=(max_samples, hidden_size))
        hidden_states_files.append(hs_file)
        hidden_states_counters.append(0)

        # MLP activations
        if mlp_activation:
            mlp_filename = os.path.join(data_dir, f'mlp_activations_layer_{layer_idx}.dat')
            mlp_file = np.memmap(mlp_filename, dtype='float16', mode='w+', shape=(max_samples, num_neurons))
            mlp_activations_files.append(mlp_file)
            mlp_activations_counters.append(0)

        # Attention norms
        if attn_norm:
            attn_filename = os.path.join(data_dir, f'attn_norms_layer_{layer_idx}.dat')
            attn_file = np.memmap(attn_filename, dtype='float16', mode='w+', shape=(max_samples, num_heads))
            attn_norms_files.append(attn_file)
            attn_norms_counters.append(0)

    return (
        hidden_states_files,
        hidden_states_counters,
        mlp_activations_files,
        mlp_activations_counters,
        attn_norms_files,
        attn_norms_counters
    )

def process_hidden_states(layer_idx, hidden_states_layer, valid_token_indices, hidden_size, hidden_states_files, hidden_states_counters):
    """
    Process and store hidden states for a specific layer.
    """
    hs = hidden_states_layer.view(-1, hidden_size)  # Shape: (batch_size * seq_len, hidden_size)
    hs_valid = hs[valid_token_indices.cpu()]  # Select valid tokens
    hs_counter = hidden_states_counters[layer_idx]
    hs_file = hidden_states_files[layer_idx]
    hs_file[hs_counter:hs_counter+hs_valid.shape[0], :] = hs_valid.cpu().numpy().astype('float16')
    hidden_states_counters[layer_idx] += hs_valid.shape[0]

def process_mlp_activations(layer_idx, mlp_activations_layer, valid_token_indices, hidden_size, mlp_activations_files, mlp_activations_counters):
    """
    Process and store MLP activations for a specific layer.
    """
    neuron_shape = mlp_activations_layer.shape[-1]
    mlp_activations_layer = mlp_activations_layer.view(-1, neuron_shape)  # Shape: (batch_size * seq_len, hidden_size)
    
    mlp_valid = mlp_activations_layer[valid_token_indices.cpu()]  # Select valid tokens
    mlp_counter = mlp_activations_counters[layer_idx]
    mlp_file = mlp_activations_files[layer_idx]
    mlp_file[mlp_counter:mlp_counter+mlp_valid.shape[0], :] = mlp_valid.cpu().numpy().astype('float16')
    mlp_activations_counters[layer_idx] += mlp_valid.shape[0]

def process_attn_norms(layer_idx, attn_outputs_layer, valid_token_indices, num_heads, attn_norms_files, attn_norms_counters):
    """
    Process and store attention norms for a specific layer.
    """
    # attn_outputs_layer has shape (batch_size, seq_len, num_heads, head_dim)
    attn = attn_outputs_layer  # Shape: (batch_size, seq_len, num_heads, head_dim)
    attn_norms = torch.norm(attn, dim=-1)  # Shape: (batch_size, seq_len, num_heads)
    attn_norms = attn_norms.view(-1, num_heads)  # Shape: (batch_size * seq_len, num_heads)
    attn_valid = attn_norms[valid_token_indices.cpu()]  # Select valid tokens
    attn_counter = attn_norms_counters[layer_idx]
    attn_file = attn_norms_files[layer_idx]
    attn_file[attn_counter:attn_counter+attn_valid.shape[0], :] = attn_valid.cpu().numpy().astype('float16')
    attn_norms_counters[layer_idx] += attn_valid.shape[0]

def process_batch(
    outputs,
    input_ids,
    attention_mask,
    total_samples,
    max_samples,
    num_layers,
    hidden_size,
    num_heads,
    hidden_states_files,
    hidden_states_counters,
    mlp_activations_files,
    mlp_activations_counters,
    attn_norms_files,
    attn_norms_counters,
    args
):
    """
    Process a batch of model outputs and update the data files.

    Returns:
        total_samples (int): Updated total number of samples processed.
        reached_max_samples (bool): Indicates if the maximum number of samples has been reached.
    """
    # Extract the outputs
    # hidden_states = outputs['hidden_states'][:-1]    # Ignore the last hidden state
    
    # use router_inputs instead of hidden_states
    hidden_states = outputs['router_inputs']
    
    if args.mlp_activation:
        mlp_activations = outputs['mlp_activations']    # Tuple of MLP activations
    else:
        mlp_activations = None
    
    if args.attn_norm:
        attn_outputs = outputs['attn_outputs']   # Tuple of attention outputs
    else:
        attn_outputs = None

    batch_size, seq_len = input_ids.shape

    # Flatten attention_mask to (batch_size * seq_len)
    attention_mask_flat = attention_mask.view(-1).bool()
    num_valid_tokens = attention_mask_flat.sum().item()

    # Determine tokens to process
    if total_samples + num_valid_tokens >= max_samples:
        tokens_to_process = max_samples - total_samples
        total_samples = max_samples
        reached_max_samples = True
    else:
        tokens_to_process = num_valid_tokens
        total_samples += num_valid_tokens
        reached_max_samples = False

    # Find indices of valid tokens
    valid_token_indices = attention_mask_flat.nonzero(as_tuple=False).view(-1)
    # If tokens_to_process < num_valid_tokens, limit the indices
    valid_token_indices = valid_token_indices[:tokens_to_process]

    for layer_idx in range(num_layers):
        # Process hidden states
        process_hidden_states(
            layer_idx,
            hidden_states[layer_idx],
            valid_token_indices,
            hidden_size,
            hidden_states_files,
            hidden_states_counters
        )
        if args.mlp_activation:
            # Process MLP activations
            process_mlp_activations(
                layer_idx,
                mlp_activations[layer_idx],
                valid_token_indices,
                hidden_size,
                mlp_activations_files,
                mlp_activations_counters
            )

        if args.attn_norm:
            # Process attention norms
            process_attn_norms(
                layer_idx,
                attn_outputs[layer_idx],
                valid_token_indices,
                num_heads,
                attn_norms_files,
                attn_norms_counters
            )

    return total_samples, reached_max_samples, num_valid_tokens

def finalize_data_collection(
    data_dir,
    num_layers,
    hidden_size,
    num_heads,
    max_samples,
    hidden_states_files,
    mlp_activations_files,
    attn_norms_files,
    hidden_states_counters,
    mlp_activations_counters,
    attn_norms_counters,
    args
):
    """
    Finalize the data collection by flushing and closing mmap files and saving metadata.

    Args:
        data_dir (str): Directory where data is stored.
        num_layers (int): Number of transformer layers.
        hidden_size (int): Hidden size of the model.
        num_heads (int): Number of attention heads.
        max_samples (int): Maximum number of samples to collect.
        hidden_states_files (list): List of mmap files for hidden states.
        mlp_activations_files (list): List of mmap files for MLP activations.
        attn_norms_files (list): List of mmap files for attention norms.
        hidden_states_counters (list): List of counters for hidden states.
        mlp_activations_counters (list): List of counters for MLP activations.
        attn_norms_counters (list): List of counters for attention norms.
    """

    for layer_idx in range(num_layers):
        # Hidden states
        hs_file = hidden_states_files[layer_idx]
        hs_file.flush()
        del hs_file  # Close the memmap

        if args.mlp_activation:
            # MLP activations
            mlp_file = mlp_activations_files[layer_idx]
            mlp_file.flush()
            del mlp_file  # Close the memmap

        if args.attn_norm:
            # Attention norms
            attn_file = attn_norms_files[layer_idx]
            attn_file.flush()
            del attn_file  # Close the memmap

    # Prepare metadata
    metadata = {
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'max_samples': max_samples,
        'hidden_states_counters': hidden_states_counters,
        'mlp_activations_counters': mlp_activations_counters,
        'attn_norms_counters': attn_norms_counters
    }

    # Save metadata to a JSON file
    metadata_filename = os.path.join(data_dir, 'metadata.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f)

    print("Finalization complete. Metadata saved.")

def arg_parser():
    parser = argparse.ArgumentParser(description='Sparse Perplexity Evaluation')
    parser.add_argument('--model_index', type=int, default=5, help='Index of the model to evaluate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--data_collection', type=bool, default=False, help='Collect data for different activation thresholds')
    parser.add_argument('--device_map', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--interactive', type=bool, default=False, help='Interactive mode for model selection')
    parser.add_argument('--data_dir', type=str, default='<PATH_TO_DATA_DIR>', help='Directory to store generated data')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum number of samples to collect')
    parser.add_argument('--model_family', type=str, default='opt', choices= ["opt", "llama"], help='Model family to evaluate')
    parser.add_argument('--mlp_activation', type=bool, default=False, help='Collect MLP activations')
    parser.add_argument('--attn_norm', type=bool, default=True, help='Collect attention norms')

    return parser.parse_args()

if __name__ =="__main__":
    args = arg_parser()
    model_name = MODELS[args.model_index-1]
    batch_size = args.batch_size
    max_length = args.max_length
    data_collection = args.data_collection
    device_map = args.device_map

    # Load the model
    if args.model_family == 'opt':
        model = OPTForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2", output_hidden_states=True, output_attentions=True,
            return_dict=True
        )
        num_neurons = model.config.ffn_dim
        
    elif args.model_family == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2", output_hidden_states=True, output_attentions=True,
            return_dict=True
        )
        num_neurons = model.config.intermediate_size

    data_loader = build_data_loader(
        model_name, "wikitext", "wikitext-2-raw-v1", batch_size, max_length, split='train'
    )
    if args.device_map == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device(device_map if torch.cuda.is_available() else 'cpu')

    # Ensure data directory exists
    model_name_clean = extract_model_name(model_name)
    folder_name = f"{model_name_clean}_act_data"
    data_dir = os.path.join(args.data_dir, folder_name)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    max_samples = args.max_samples
    
    # print if we are using mlp activations and attn norms
    print(f"Collecting data for model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Number of layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Number of neurons: {num_neurons}")
    print(f"Max samples: {max_samples}")
    print(f"Collecting MLP activations: {args.mlp_activation}")
    print(f"Collecting attention norms: {args.attn_norm}")
        
    # Initialize data structures using the function
    (hidden_states_files, hidden_states_counters, mlp_activations_files,
    mlp_activations_counters, attn_norms_files, attn_norms_counters) = initialize_data_structures(data_dir, num_layers,
                                                                                                  hidden_size, num_heads, num_neurons, max_samples,
                                                                                                  mlp_activation=args.mlp_activation, attn_norm=args.attn_norm)

    total_samples = 0
    # Data collection loop
    with torch.no_grad():
        with tqdm(total=max_samples, desc="Router training data collection") as pbar:
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                output_attentions=False, return_dict=True, output_mlp_activation=args.mlp_activation,  
                                output_attn_output=args.attn_norm, output_router_inputs=True)

                # Process the batch outputs
                total_samples, reached_max_samples, num_valid_tokens = process_batch(
                    outputs, input_ids, attention_mask,
                    total_samples, max_samples, num_layers,
                    hidden_size, num_heads, hidden_states_files,
                    hidden_states_counters, mlp_activations_files, mlp_activations_counters,
                    attn_norms_files, attn_norms_counters,
                    args=args)
                
                pbar.update(num_valid_tokens)

                if reached_max_samples:
                    break

    # Finalization: Flush and clean up using the new function
    finalize_data_collection(
        data_dir, num_layers, hidden_size,
        num_heads, max_samples, hidden_states_files,
        mlp_activations_files, attn_norms_files, hidden_states_counters,
        mlp_activations_counters, attn_norms_counters,
        args
    )
    
    if reached_max_samples:
        print(f"Reached maximum number of samples. total_samples = {total_samples}")
        print(f"Data collection complete. Data saved to {data_dir}")