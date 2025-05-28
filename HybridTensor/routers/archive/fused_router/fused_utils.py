import numpy as np
import os
import json
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# router training functions 

def get_layer_data(data_dir, layer_idx, total_samples=None):
    """
    Load hidden_states, attn_norms, and mlp_activations data for a specific layer.

    Args:
        data_dir (str): Directory where the data is stored.
        layer_idx (int): The index of the layer to load data for.
        total_samples (int, optional): The number of samples to load. If None, load all samples.

    Returns:
        tuple: A tuple containing:
            - hidden_states (np.ndarray): Array of shape (num_samples, hidden_size)
            - attn_norms (np.ndarray): Array of shape (num_samples, num_heads)
            - mlp_activations (np.ndarray): Array of shape (num_samples, hidden_size * 4)
    """
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

    # Validate layer index
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be between 0 and {num_layers - 1}.")

    # Determine number of samples to load
    if total_samples is not None:
        if total_samples <= 0:
            raise ValueError(f"total_samples must be a positive integer, got {total_samples}")
        num_samples = min(total_samples, max_samples)
    else:
        num_samples = max_samples

    # File names and feature sizes
    hidden_states_feature_size = hidden_size
    hidden_states_filename = f"hidden_states_layer_{layer_idx}.dat"

    attn_norms_feature_size = num_heads
    attn_norms_filename = f"attn_norms_layer_{layer_idx}.dat"

    mlp_activations_feature_size = hidden_size * 4
    mlp_activations_filename = f"mlp_activations_layer_{layer_idx}.dat"

    # Paths to the mmap files
    hidden_states_path = os.path.join(data_dir, hidden_states_filename)
    attn_norms_path = os.path.join(data_dir, attn_norms_filename)
    mlp_activations_path = os.path.join(data_dir, mlp_activations_filename)

    # Load hidden_states data
    if not os.path.exists(hidden_states_path):
        raise FileNotFoundError(f"Hidden states file not found at {hidden_states_path}")

    logging.info(f"Reading hidden_states from {hidden_states_path}")

    hidden_states_mmap = np.memmap(hidden_states_path, dtype='float16', mode='r',
                                   shape=(max_samples, hidden_states_feature_size))
    hidden_states = np.array(hidden_states_mmap[:num_samples])
    del hidden_states_mmap  # Close the memmap

    # Load attn_norms data
    if not os.path.exists(attn_norms_path):
        raise FileNotFoundError(f"Attn norms file not found at {attn_norms_path}")

    logging.info(f"Reading attn_norms from {attn_norms_path}")

    attn_norms_mmap = np.memmap(attn_norms_path, dtype='float16', mode='r',
                                shape=(max_samples, attn_norms_feature_size))
    attn_norms = np.array(attn_norms_mmap[:num_samples])
    del attn_norms_mmap  # Close the memmap

    # Load mlp_activations data
    if not os.path.exists(mlp_activations_path):
        raise FileNotFoundError(f"MLP activations file not found at {mlp_activations_path}")

    logging.info(f"Reading mlp_activations from {mlp_activations_path}")

    mlp_activations_mmap = np.memmap(mlp_activations_path, dtype='float16', mode='r',
                                     shape=(max_samples, mlp_activations_feature_size))
    mlp_activations = np.array(mlp_activations_mmap[:num_samples])
    del mlp_activations_mmap  # Close the memmap

    logging.info(f"Loaded {num_samples} samples for layer {layer_idx}.")

    return hidden_states, attn_norms, mlp_activations

def create_labels(mlp_activations, attn_norms, mlp_threshold=0, attn_top_k_ratio=0.3):
    # Create neuron labels
    neuron_labels = (mlp_activations > mlp_threshold).astype(np.float32)

    # Create head labels
    num_heads = attn_norms.shape[1]
    top_k = int(num_heads * attn_top_k_ratio)
    head_labels = np.zeros_like(attn_norms, dtype=np.float32)
    
    for i in range(attn_norms.shape[0]):
        indices = np.argpartition(-attn_norms[i], top_k)[:top_k]
        head_labels[i, indices] = 1.0

    return neuron_labels, head_labels

class FusedRouterDataset(Dataset):
    def __init__(self, hidden_states, neuron_labels, head_labels):
        self.hidden_states = hidden_states
        self.neuron_labels = neuron_labels
        self.head_labels = head_labels

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        x = self.hidden_states[idx]
        neuron_label = self.neuron_labels[idx]
        head_label = self.head_labels[idx]
        return x, neuron_label, head_label
    
def create_dataloaders(hidden_states, neuron_labels, head_labels, batch_size=64):
    # Define your split ratio and random seed
    validation_split = 0.2  # 20% for validation
    random_seed = 11

    # Convert numpy arrays to tensors
    hidden_states_tensor = torch.from_numpy(hidden_states).float()
    neuron_labels_tensor = torch.from_numpy(neuron_labels).float()
    head_labels_tensor = torch.from_numpy(head_labels).float()

    # Split the data
    hidden_states_train, hidden_states_val, neuron_labels_train, neuron_labels_val, head_labels_train, head_labels_val = train_test_split(
        hidden_states_tensor,
        neuron_labels_tensor,
        head_labels_tensor,
        test_size=validation_split,
        random_state=random_seed,
        shuffle=True
    )
    
    # Create dataset instances
    train_dataset = FusedRouterDataset(hidden_states_train, neuron_labels_train, head_labels_train)
    val_dataset = FusedRouterDataset(hidden_states_val, neuron_labels_val, head_labels_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader