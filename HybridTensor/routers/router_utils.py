import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import numpy as np
from pathlib import Path
from datetime import datetime

DATA = {
    "175b": {
        "c4": "../data/175b_c4",
    },
    "66b": {
        "c4": "../data/66b_c4",
    },
    "30b": {
        "c4": "../data/30b_c4",
    },
    "6_7b": {
        "c4": "/home/grads/s/<name>/nvme/ssd_backup/dejavu/data/6_7b_c4"
    }
}

MODEL_CHOICES = ['175b', '66b', '30b', '6_7b']
DATA_CHOICES = ['c4']
CONFIG = {
    '175b':{
        'num_layer': 95,
        'ckt_storage': "bylayer",
        'd':12288,
        'h': 96,
        'N':400000,
    },
    '66b':{
        'num_layer': 64,
        'ckt_storage': "bylayer",
        'd':9216,
        'h': 72,
        'N':400000,
    },
    '30b':{
        'num_layer': 24,
        'ckt_storage': "bylayer",
        'd':2048,
        'h': 32,
        'N':400000,
    },
    '6_7b':{
        'num_layer': 32,
        'ckt_storage': "bylayer",
        'd':4096,
        'h': 32,
        'N':400000,
    },
}

class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train ):
        self.X = X
        self.Y = Y 
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        if y.sum()== 0:
            #print("all zero y")
            # exit()
            pass
        return x, y


def get_data(args, l):
    if CONFIG[args.model]['ckt_storage'] == "bylayer":
        #path = f"{DATA[args.model][args.dataset]}/mlp_x_{l}.mmap"
        path = f"{DATA[args.model][args.dataset]}/mlp_sp_x_{l}.mmap"
        logging.info(f"Reading query from {path}")
        query = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000,CONFIG[args.model]['d']))[: CONFIG[args.model]['N']])
        path = f"{DATA[args.model][args.dataset]}/mlp_label_{l}.mmap"
        logging.info(f"Reading MLP label from {path}")
        label = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000,CONFIG[args.model]['d'] * 4))[: CONFIG[args.model]['N']])
        
        num_valid = (label.sum(-1) > 0).sum()
        return  query[:num_valid], label[:num_valid]
        #return  query, label

import numpy as np
import os 
import json 

# HybridTensor get data 
def get_data_(data_dir, layer_idx, data_type, total_neurons = None, total_samples = None):
    """
    Load query and label data for a specific layer based on the configuration.

    Args:
        args (argparse.Namespace): Contains configuration parameters such as model, dataset, and data_type.
        layer_idx (int): The index of the layer to load data for.

    Returns:
        tuple: A tuple containing:
            - query (np.ndarray): The query data array of shape (num_valid, feature_size_query).
            - label (np.ndarray): The label data array of shape (num_valid, feature_size_label).
    """

    # Ensure data_type is valid
    if data_type not in ['mlp_activations', 'attn_norms']:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'mlp_activations' or 'attn_norms'.")

    # logging.info(f"Loading data from directory: {data_dir}")

    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_layers = metadata['num_layers']
    hidden_size = metadata['hidden_size']
    num_heads = metadata['num_heads']
    max_samples = metadata['max_samples']
    if total_neurons == None:
        total_neurons = hidden_size * 4

    # Validate layer index
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be between 0 and {num_layers - 1}.")

    # Determine feature sizes and corresponding file names
    if data_type == 'mlp_activations':
        label_feature_size = total_neurons
        label_filename = f"mlp_activations_layer_{layer_idx}.dat"
    elif data_type == 'attn_norms':
        label_feature_size = num_heads
        label_filename = f"attn_norms_layer_{layer_idx}.dat"

    # Query corresponds to hidden_states
    query_feature_size = hidden_size
    query_filename = f"hidden_states_layer_{layer_idx}.dat"

    # Paths to the mmap files
    query_path = os.path.join(data_dir, query_filename)
    label_path = os.path.join(data_dir, label_filename)

    # Log the paths
    logging.info(f"Reading query from {query_path}")
    logging.info(f"Reading label from {label_path}")

    # Determine number of samples to load
    if total_samples is not None:
        if total_samples <= 0:
            raise ValueError(f"total_samples must be a positive integer, got {total_samples}")
        num_samples = min(total_samples, max_samples)
    else:
        num_samples = max_samples

    # Load query data
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found at {query_path}")
    
    query_mmap = np.memmap(query_path, dtype='float16', mode='r', shape=(max_samples, query_feature_size))
    query = np.array(query_mmap[:num_samples])
    del query_mmap  # Close the memmap

    # Load label data
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    
    label_mmap = np.memmap(label_path, dtype='float16', mode='r', shape=(max_samples, label_feature_size))
    label = np.array(label_mmap[:num_samples])
    del label_mmap  # Close the memmap

    logging.info(f"Loaded {label.shape[0]} valid samples for layer {layer_idx}.")

    return query, label


def get_expert_config(expert_dir, layer_idx):
    expert_file = f"{expert_dir}/layer_{layer_idx}_experts.json"
    
    # if expert_file does not exist, then we need to create the file before starting the training
    if os.path.exists(expert_file):
        # load the expert file
        with open(expert_file, 'r') as f:
            experts = json.load(f)
            print(f"Expert file {expert_file} read successfully.")
    else:
        raise FileNotFoundError(f"Expert file {expert_file} not found.")
    return experts


def augment_data(labels, device='cpu'):
    """
    Data augmentation function that processes the input query and labels,
    identifies hot and cold neurons based on activation counts, and returns
    the cold labels (corresponding to cold neurons).
    
    Args:
        query (torch.Tensor): Input data or query tensor.
        labels (torch.Tensor): Corresponding labels tensor.
        device (str): The device ('cpu' or 'cuda') to perform operations on.
        
    Returns:
        torch.Tensor: Cold labels corresponding to the cold neurons.
    """

    # Convert labels into binary labels (labels > 0)
    # print("Convert labels into binary labels (labels > 0):")
    binary_labels = (labels > 0).astype(int)

    # print("Binary Labels:", binary_labels)
    # print("Binary Labels shape ", binary_labels.shape)

    # Count activations for each neuron
    activation_counts = binary_labels.sum(axis=0)
    activation_counts = torch.tensor(activation_counts, device=device)

    # Sort neurons by activation counts in descending order
    sorted_counts, sorted_indices = torch.sort(activation_counts, descending=True)

    # Compute cumulative sum of sorted activation counts
    cumulative_counts = torch.cumsum(sorted_counts, dim=0)

    # Compute total activations
    total_activations = cumulative_counts[-1]

    # Find the index where cumulative sum reaches 80% of total activations
    threshold = 0.8 * total_activations
    hot_neuron_threshold_idx = torch.nonzero(cumulative_counts >= threshold)[0].item()

    # Indices of hot and cold neurons
    hot_neuron_indices = sorted_indices[:hot_neuron_threshold_idx + 1]
    cold_neuron_indices = sorted_indices[hot_neuron_threshold_idx + 1:]

    # Move indices back to CPU
    cold_neuron_indices = cold_neuron_indices.cpu()
    hot_neuron_indices = hot_neuron_indices.cpu()

    # Filter cold and pruned (zero-activation) neurons
    pruned_cold_neurons = torch.nonzero(activation_counts == 0).cpu().flatten()
    filtered_cold_neurons = cold_neuron_indices[~cold_neuron_indices.unsqueeze(1).eq(pruned_cold_neurons).any(dim=1)]

    logging.info(f"Hot neurons count: {len(hot_neuron_indices)}")
    logging.info(f"Cold neurons count: {len(cold_neuron_indices)}")
    logging.info(f"Non-zero neurons count: {len(filtered_cold_neurons)}")
    logging.info(f"Zero neurons count: {len(pruned_cold_neurons)}")

    # Extract the cold neurons from the binary labels
    cold_labels = labels[:, filtered_cold_neurons]
    
    return (hot_neuron_indices, filtered_cold_neurons, pruned_cold_neurons), cold_labels


def create_dataset(query, labels, args):
    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)

    logging.info(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    logging.info(f"# training data: {num_train}, # test data: {num_test}")

    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)

    train_dataloader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader

def create_log_path(model: str, layer_idx: int) -> str:
    """
    Creates the necessary directories and generates a log file path.

    Parameters:
    - model (str): The name of the model.
    - layer_idx (int): The index of the MLP layer.

    Returns:
    - str: The full path to the log file.
    """
    # Define the base log directory
    base_log_dir = 'log'
    
    # Get today's date in YYYY-MM-DD format
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Path to today's log directory
    today_log_dir = os.path.join(base_log_dir, today_str)
    
    # Create the directories if they don't exist
    os.makedirs(today_log_dir, exist_ok=True)
    
    # Get the current time in HH-MM-SS format for the filename
    current_time_str = datetime.now().strftime('%H-%M-%S')
    
    # Create the log filename
    log_name = f'mlp_{model}_{layer_idx}_{current_time_str}.log'
    
    # Combine the directory and filename to get the full path
    log_path = os.path.join(today_log_dir, log_name)
    
    return log_path

def generate_label(y):
    # positive
    one_hot = (y > 0).to(y.dtype)
    return one_hot

def evaluate(model, device, loader, args, smalltest=False):
    model.eval()

    eval = {
        "Loss": [],
        "Loss Weight": [],
        "Recall": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate((loader)):
            x, y = batch
            y = y.float().to(device)
            
            y = generate_label(y)
            
            logits = model(x.to(device))
            probs = logits.sigmoid()
            preds = probs >= 0.5

            dif = y.int() - preds.int()
            miss = dif > 0.0  # classifier didn't activated target neuron

            weight = (y.sum() / y.numel()) + 0.005
            loss_weight = y * (1 - weight) + weight
            eval["Loss Weight"] += [weight.item()]
            eval["Loss"] += [
                torch.nn.functional.binary_cross_entropy(
                    probs, y, weight=loss_weight
                ).item()
            ]

            eval["Recall"] += [
                ((y.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item())
            ]
            eval["True Sparsity"] += [y.sum(dim=1).float().mean().item()]
            eval["Classifier Sparsity"] += [preds.sum(dim=1).float().mean().item()]

            if batch_idx >= 100 and smalltest:
                break

    for k, v in eval.items():
        eval[k] = np.array(v).mean()

    eval["Recall"] = eval["Recall"] / eval["True Sparsity"]
    return eval

def generate_experiment_id():
    """
    Generates a unique experiment identifier based on the current time in HHMMSS format.
    
    Returns:
    - str: A string representing the current time as HHMMSS.
    """
    return datetime.now().strftime('%H%M%S')

def create_date_directory(ckpt_dir: str, model_name: str) -> Path:    
    # Step 2: Define the base directory and create date-based directory
    base_dir = f"{ckpt_dir}/opt-{model_name}-sparse-predictor"
    base_dir_path = Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure base directory exists
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    date_dir = Path(base_dir) / today_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir

def generate_log_filename(experiment_id: str, model_name: str, layer_idx: int, date_dir: str) -> str:
    log_filename = f"{experiment_id}_{model_name}_mlp_layer{layer_idx}.log"
    log_file_path = date_dir / log_filename
    return log_file_path

def generate_model_filename(experiment_id: str, model_name: str, layer_idx: int, eval_result: dict) -> str:
    recall = eval_result.get('Recall', 0)
    precision = eval_result.get('Precision', 0)
    sparsity = eval_result.get('Classifier Sparsity', 0)
    return f"{experiment_id}_{model_name}_mlp_layer{layer_idx}_-{recall:.4f}-{precision:.4f}-{sparsity:.0f}.pt"

def setup_logging(log_file= Path):
    """
    Sets up the logging configuration.
    
    Parameters:
    - log_file (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Write logs to a file
            logging.StreamHandler()  # Also output logs to the console
        ]
    )



def plot_neuron_activation_frequency(
    labels: torch.Tensor,
    threshold: float = 0.0,
    max_xticks: int = 20,
    figsize: tuple = (12, 6),
    title: str = 'Neuron Activation Frequency in MLP Layer',
    color: str = 'skyblue'
):
    """
    Plots the activation frequency of neurons in a Multi-Layer Perceptron (MLP) layer.

    Args:
        labels (torch.Tensor): A 2D tensor of shape (num_samples, num_neurons) containing label data.
        threshold (float, optional): Threshold to convert labels into binary. Labels > threshold are set to 1. Defaults to 0.0.
        max_xticks (int, optional): Maximum number of x-axis ticks to display. Defaults to 20.
        figsize (tuple, optional): Size of the matplotlib figure. Defaults to (12, 6).
        title (str, optional): Title of the plot. Defaults to 'Neuron Activation Frequency in MLP Layer'.
        color (str or list, optional): Color of the bars in the plot. Defaults to 'skyblue'.

    Raises:
        ValueError: If `labels` is not a 2D tensor.
    """
    
    binary_labels = (labels > threshold).astype(int)
    activation_counts = binary_labels.sum(axis=0)

    # Get sorted indices in descending order
    sorted_indices = np.argsort(-activation_counts)

    # Sort the activation counts accordingly
    sorted_counts = activation_counts[sorted_indices]


    plt.figure(figsize=figsize)
    plt.bar(range(len(sorted_counts)), sorted_counts, color=color)
    plt.xlabel('Neurons (sorted by activation frequency)')
    plt.ylabel('Activation Frequency')
    plt.title(title)
    plt.xticks(ticks=range(0, len(sorted_counts), max(1, len(sorted_counts)//20)),
            labels=sorted_indices[::max(1, len(sorted_counts)//20)],
            rotation=90)  # Adjust ticks for better readability
    plt.tight_layout()
    plt.show()
    
# Argument parser for jupyter notebook
from unittest.mock import patch
import argparse
import sys

def arg_parser_notebook(test_args=[None]):
    parser = argparse.ArgumentParser(description="PyTorch OPT Full Model")
    parser.add_argument("--model", type=str, default="6_7b", choices=MODEL_CHOICES)
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--dataset", type=str, default="c4", choices=DATA_CHOICES)
    parser.add_argument("--L", type=int, default=0, help="which layer")
    parser.add_argument("--D", type=int, default=1024, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")  # a smaller batch size results in better precision and recall of the model
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="/home/grads/s/<name>/nvme/HybridTensor/checkpoint", help="checkpoint directory")  # add a argument for checkpoint dir
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")  # add a argument for which gpu to use
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss_fn", type=str, default="focal")
    parser.add_argument("--data_augmentation", type=bool, default=False)    # use cold neurons for routing 
    
    with patch.object(sys, 'argv', test_args):
        parser = arg_parser_notebook()
        args = parser.parse_args()
        args_dict = vars(args)  # Convert args to a dictionary for display

    return args_dict