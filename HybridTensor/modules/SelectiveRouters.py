import os
import re
import torch
from collections import OrderedDict

def create_mlp_router_state_dict(router_files_dir):
    """
    Loads all mlp_router weight files from the specified directory and creates a router_state_dict
    with keys formatted as 'transformer.layers.{layer_num}.mlp_router.{param_name}'.

    Args:
        router_files_dir (str): Path to the directory containing mlp_router_*.pt files.

    Returns:
        OrderedDict: A state dictionary suitable for loading into a transformer model.
    """
    # Regular expression to extract layer number from filename
    router_file_pattern = re.compile(r'mlp_router_(\d+)-[\d.]+-[\d.]+-[\d.]+\.pt$')

    router_state_dict = OrderedDict()

    # List all files in the directory
    try:
        all_files = os.listdir(router_files_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{router_files_dir}' does not exist.")
        return None

    # Filter files matching the pattern
    router_files = [f for f in all_files if router_file_pattern.match(f)]

    if not router_files:
        print(f"No router files found in directory '{router_files_dir}'.")
        return None

    for file_name in sorted(router_files, key=lambda x: int(router_file_pattern.match(x).group(1))):
        match = router_file_pattern.match(file_name)
        if not match:
            print(f"Skipping file '{file_name}' as it does not match the pattern.")
            continue

        layer_num = int(match.group(1))
        file_path = os.path.join(router_files_dir, file_name)

        try:
            # Load the router's state dict
            router_weights = torch.load(file_path, map_location='cpu')
            if not isinstance(router_weights, dict):
                print(f"Warning: The file '{file_path}' does not contain a state dictionary. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            continue

        # Iterate through each parameter in the router's state dict
        for param_name, param_tensor in router_weights.items():
            # Construct the new key
            new_key = f"transformer.layers.{layer_num}.mlp_router.{param_name}"
            router_state_dict[new_key] = param_tensor

        # print(f"Loaded router for layer {layer_num} from '{file_name}'.")

    print(f"Total routers loaded: {len(router_state_dict) // 2}")  # Assuming 4 params per router (weight & bias for 2 layers)
    return router_state_dict


def create_attn_router_state_dict(router_files_dir):
    """
    Loads all attn_router weight files from the specified directory and creates a router_state_dict
    with keys formatted as 'transformer.layers.{layer_num}.mha_router.{param_name}'.

    Args:
        router_files_dir (str): Path to the directory containing attn_router_*.pt files.

    Returns:
        OrderedDict: A state dictionary suitable for loading into a transformer model.
    """
    # Regular expression to extract layer number from filename
    # Pattern: attn_router_{layer_num}-{value1}-{value2}.pt
    router_file_pattern = re.compile(r'attn_router_(\d+)-[\d.]+-[\d.]+\.pt$')

    router_state_dict = OrderedDict()

    # List all files in the directory
    try:
        all_files = os.listdir(router_files_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{router_files_dir}' does not exist.")
        return None

    # Filter files matching the pattern
    router_files = [f for f in all_files if router_file_pattern.match(f)]

    if not router_files:
        print(f"No attn_router files found in directory '{router_files_dir}'.")
        return None

    # To handle potential duplicates, keep track of loaded layer numbers
    loaded_layers = set()

    for file_name in sorted(router_files, key=lambda x: int(router_file_pattern.match(x).group(1))):
        match = router_file_pattern.match(file_name)
        if not match:
            print(f"Skipping file '{file_name}' as it does not match the pattern.")
            continue

        layer_num = int(match.group(1))
        if layer_num in loaded_layers:
            print(f"Warning: Multiple router files found for layer {layer_num}. Skipping '{file_name}'.")
            continue  # Skip duplicate layers

        file_path = os.path.join(router_files_dir, file_name)

        try:
            # Load the router's state dict
            router_weights = torch.load(file_path, map_location='cpu')
            if not isinstance(router_weights, dict):
                print(f"Warning: The file '{file_path}' does not contain a state dictionary. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            continue

        # Iterate through each parameter in the router's state dict
        for param_name, param_tensor in router_weights.items():
            # Construct the new key
            new_key = f"transformer.layers.{layer_num}.mha_router.{param_name}"
            router_state_dict[new_key] = param_tensor

        loaded_layers.add(layer_num)
        # print(f"Loaded MHA router for layer {layer_num} from '{file_name}'.")

    print(f"Total MHA routers loaded: {len(loaded_layers)}")
    return router_state_dict


