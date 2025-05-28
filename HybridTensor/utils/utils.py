import math
import numpy as np
import torch
import argparse
import os
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # For progress bars

def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_features', type=int, default=32768)
    parser.add_argument('--in_features', type=int, default=8192)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--index_size', type=int, default=8192)
    parser.add_argument('--head_density', type=float, default=0.25)
    parser.add_argument('--attn_topk', type=float, default=0.5)
    parser.add_argument('--print_results', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--check_results', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--max_batch_size', type=int, default=32)
    parser.add_argument('--max_seqlen', type=int, default=2048)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--mode', type=str, default='row', choices=['row', 'col', 'auto'])
    
    return parser.parse_args()

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

def get_gpu_name():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        # Clean the GPU name to make it filename-friendly
        gpu_name_clean = gpu_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return gpu_name_clean
    else:
        return "CPU"
    
def _get_device(device_id):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    return device

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]

def create_results_directory(results_dir):
    """
    Creates the results directory if it does not exist.

    Parameters:
    - results_dir (str or Path): The path to the results directory.

    Returns:
    - Path: The Path object representing the results directory.
    """
    path = Path(results_dir).resolve()
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created results directory at: {path}")
    else:
        # print(f"Results directory already exists at: {path}")
        pass
    
    return path

def ZeroIndex(index_vec, max_value):
    # Create a set of all integers from 0 to max_value - 1
    all_integers = set(range(max_value))

    # Convert index_vec to a set
    index_set = set(index_vec.cpu().numpy())

    # Subtract index_set from all_integers
    remaining_integers = all_integers - index_set

    # Convert the result back to a tensor
    zero_index = torch.tensor(list(remaining_integers), dtype=torch.int32, device='cuda')

    return zero_index

def sparse_index(index_size, max_value, return_zero_index = False):
    index_vec = torch.randperm(max_value, dtype=torch.int32, device='cuda')[:index_size]
    index_vec, _ = torch.sort(index_vec)
    if return_zero_index:
        zero_index = ZeroIndex(index_vec, max_value)
    else:
        zero_index = None
    return index_vec, zero_index

# utility function to study activations

def create_random_batches(labels, batch_size=32):
    """
    Shuffles the labels and splits them into random batches.

    Parameters:
    - labels (np.ndarray): The labels matrix of shape (212646, 16384).
    - batch_size (int): The number of samples per batch.

    Returns:
    - List[np.ndarray]: A list of batches, each containing `batch_size` rows.
    """
    num_samples = labels.shape[0]
    
    # Generate a permutation of indices
    shuffled_indices = np.random.permutation(num_samples)
    
    # Shuffle the labels matrix
    shuffled_labels = labels[shuffled_indices]
    
    # Calculate the number of complete batches
    num_batches = num_samples // batch_size
    
    # Split the shuffled labels into batches
    batches = np.split(shuffled_labels[:num_batches * batch_size], num_batches)
    
    return batches

def generate_BH_index(batch_size: int, heads: int, selected_heads: int, device = 'cuda'):
    '''
    Generates a random list of selected heads for each batch.
    
    Args:
    - batch_size (int): Number of batches.
    - heads (int): Total number of heads.
    - selected_heads (int): Number of heads to select for each batch.
    
    Returns:
    - bh_index (torch.Tensor): Tensor of shape (batch_size * selected_heads, 2) where each row is (batch_idx, head_idx).
    '''
    N_selected = batch_size * selected_heads
    bh_index = torch.zeros((N_selected, 2), dtype=torch.int32, device=device)
    
    for batch_idx in range(batch_size):
        selected_head_indices = torch.randperm(heads)[:selected_heads]
        sorted_head_indices = torch.sort(selected_head_indices).values
        for i, head_idx in enumerate(sorted_head_indices):
            bh_index[batch_idx * selected_heads + i] = torch.tensor([batch_idx, head_idx], dtype=torch.int32)
    
    return bh_index

def generate_random_BH_index(batch_size: int, heads: int, selected_heads: int, device = 'cuda'):
    '''
    Generates a random list of selected heads for each batch.
    
    Args:
    - batch_size (int): Number of batches.
    - heads (int): Total number of heads.
    - selected_heads (int): Number of heads to select for each batch.
    
    Returns:
    - bh_index (torch.Tensor): Tensor of shape (batch_size, selected_heads)
    '''
    bh_index = torch.zeros((batch_size, selected_heads), dtype=torch.int32, device=device)
    
    for batch_idx in range(batch_size):
        selected_head_indices = torch.randperm(heads)[:selected_heads]
        # sort the selected head indices
        sorted_head_indices = torch.sort(selected_head_indices).values
        bh_index[batch_idx] =  sorted_head_indices
    
    return bh_index

def generate_random_BG_index(batch_size: int, groups: int, selected_groups: int, device = 'cuda'):
    '''
    Generates a random list of selected heads for each batch.
    
    Args:
    - batch_size (int): Number of batches.
    - heads (int): Total number of heads.
    - selected_heads (int): Number of heads to select for each batch.
    
    Returns:
    - bh_index (torch.Tensor): Tensor of shape (batch_size, selected_heads)
    '''
    bg_index = torch.zeros((batch_size, selected_groups), dtype=torch.int32, device=device)
    
    for batch_idx in range(batch_size):
        selected_group_indices = torch.randperm(groups)[:selected_groups]
        # sort the selected head indices
        sorted_group_indices = torch.sort(selected_group_indices).values
        bg_index[batch_idx] =  sorted_group_indices
    
    return bg_index



def activation_stats_layer(test_batches, total_neurons, device):
    """
    Calculates the average and standard deviation of activations across batches.

    Parameters:
    - test_batches (List[np.ndarray] or List[torch.Tensor]): List of batches containing label data.
    - total_neurons (int): Total number of neurons.
    - device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
    - avg_act (float): The average number of activations per batch.
    - std_dev (float): The standard deviation of activations across batches.
    """
    sum_activation = 0.0       # To accumulate the total activations
    sum_activation_sq = 0.0    # To accumulate the squared activations
    num_batches = len(test_batches)

    for i, batch in enumerate(test_batches):
        # Convert batch to a PyTorch tensor if it's not already
        if not isinstance(batch, torch.Tensor):
            torch_labels = torch.tensor(batch, dtype=torch.float32, device=device)
        else:
            torch_labels = batch.to(device=device, dtype=torch.float32)

        # Binarize the labels: 1 if activation > 0, else 0
        binary_labels = (torch_labels > 0).int()

        # Sum activations per neuron across the batch
        activation_counts = binary_labels.sum(dim=0)

        # Convert counts to binary: 1 if neuron is activated in the batch, else 0
        activated_neurons = (activation_counts > 0).int()

        # Total number of activated neurons in this batch
        total_activations = activated_neurons.sum().item()

        # Accumulate sum and sum of squares
        sum_activation += total_activations
        sum_activation_sq += total_activations ** 2

        # Optional: Print progress every 1000 batches
        # if (i + 1) % 1000 == 0 or (i + 1) == num_batches:
        #     print(f"Processed {i + 1}/{num_batches} batches")

    # Calculate average activation
    avg_act = sum_activation / num_batches

    # Calculate variance and standard deviation
    variance = (sum_activation_sq / num_batches) - (avg_act ** 2)
    std_dev = variance ** 0.5

    # Display results
    print(f"\nAverage activation: {avg_act:.2f} "
          f"({(avg_act / total_neurons) * 100:.2f}% of total neurons)")
    print(f"Standard deviation of activation: {std_dev:.2f}")

    return avg_act, std_dev

def calculate_index_sizes(in_features):
    """
    Calculate index sizes based on the given in_features.
    The sizes are rounded up to the nearest multiple of 1024 
    and generated in 5% increments up to 100% of total neurons.
    
    Args:
        in_features (int): The number of input features.
    
    Returns:
        List[int]: A list of index sizes rounded up to the nearest multiple of 1024.
    """
    index_sizes = []
    total_neurons = in_features * 4

    # Generate 20 percentages from 5% to 100% in increments of 5%
    percentages = [i for i in range(5, 105, 5)]

    # Calculate index sizes and round up to the nearest multiple of 1024
    for p in percentages:
        index_size = int((p / 100) * total_neurons)
        index_size = math.ceil(index_size / 1024) * 1024
        index_sizes.append(index_size)
    
    return index_sizes


def compute_perplexity(model, dataloader, device):
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Shift input_ids and labels for causal language modeling
            labels = input_ids.clone()
            # Replace padding tokens in labels by -100 so they are ignored in loss computation
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Multiply loss by number of tokens in the batch
            # The loss is averaged over the number of non-masked tokens
            # To get the total loss, multiply by the number of non-masked tokens
            total_loss += loss.item() * torch.sum(labels != -100).item()
            total_tokens += torch.sum(labels != -100).item()

    # Calculate perplexity
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity