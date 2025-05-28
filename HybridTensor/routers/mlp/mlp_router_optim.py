# python -m HybridTensor.routers.mlp.mlp_router_optim --model_index 5 --batch_size <BATCH_SIZE_INFERENCE> --mlp_ckpt_dir <PATH_TO_MLP_ROUTER_CHECKPOINTS> --act_data_dir <PATH_TO_ACTIVATION_DATA>

from HybridTensor.routers.router_utils import get_data_
from HybridTensor.modules.SelectiveMLP import MLPRouter
from HybridTensor.utils.activations import MODELS, CONFIGS
from HybridTensor.utils.activations import build_mlp_topk_lookup
from HybridTensor.utils.utils import extract_model_name
import os
import torch
import argparse
import re
import numpy as np
import csv
from tqdm import tqdm

def save_dict_to_csv(data, batch_size, model_name, results_dir):
    """
    Saves a dictionary to a CSV file with columns 'layer', 'batch_size', and 'optimal_k'.
    A new directory named {model_name} is created under results_dir, and the file is saved there as 
    mlp_router_configs_bsize_{batch_size}.csv.
    
    Args:
        data (dict): Dictionary with layer index (int) as key and optimal k (int) as value.
        batch_size (int): Fixed batch size for all layers.
        model_name (str): Model name used for directory creation.
        results_dir (str): Directory where the model-specific folder will be created.
    """
    # Create new directory for the model inside results_dir
    model_name = extract_model_name(model_name)
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define the filename and complete path
    filename = f"mlp_router_configs_bsize_{batch_size}.csv"
    file_path = os.path.join(model_dir, filename)
    
    # Write CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer', 'batch_size', 'optimal_k'])
        for layer, optimal_k in data.items():
            writer.writerow([layer, batch_size, optimal_k])
    
    print(f"File saved as {file_path}")

def load_router_dict_from_csv(file_path: str, batch_size: int) -> dict:
    """
    Loads a dictionary from a CSV file with columns 'layer', 'batch_size', and 'optimal_k'.
    The batch_size column is ignored; only the mapping of layer -> optimal_k is returned.
    
    Args:
        file_path (str): Full path to the input CSV file.
    
    Returns:
        dict: Dictionary mapping layer index (int) to optimal k (int).
    """
    file_name = f"mlp_router_configs_bsize_{batch_size}.csv"
    file_path = os.path.join(file_path, file_name)
    
    data = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            layer = int(row['layer'])
            optimal_k = int(row['optimal_k'])
            data[layer] = optimal_k
    return data

def build_default_mlp_topk_lookup(num_layers, default_topk = 512):
    """
    Creates a default lookup table for MLP top-k values.
    
    The lookup maps each layer to a default top-k value.
    
    Parameters:
        num_layers (int): The number of layers in the model.
        default_topk (int): The default top-k value to assign to each layer.
    Returns:
        dict: A mapping from layer id to the default top-k value.
    """
    
    mlp_lookup = {i: default_topk for i in range(num_layers)}
    return mlp_lookup

def load_mlp_router(layer_id, directory, embed_dim, low_rank_dim, out_dim, act_th, device=None, dtype=None):
    """
    Load the MLP router for a specific layer.

    Args:
        layer_id (int): The ID of the layer.
        directory (str): Path to the directory containing the router checkpoint files.
        embed_dim (int): Input embedding dimension.
        low_rank_dim (int): Low-rank intermediate dimension.
        out_dim (int): Number of neurons.
        act_th (float): Activation threshold.
        device: Device to load the model onto.
        dtype: Data type.

    Returns:
        MLPRouter: Loaded MLP router model.
    """
    pattern = re.compile(rf"mlp_router_{layer_id}-.*\.pt")
    files = os.listdir(directory)
    router_file = next((f for f in files if pattern.match(f)), None)

    if router_file is None:
        raise FileNotFoundError(f"No router file found for layer {layer_id} in {directory}")

    router_path = os.path.join(directory, router_file)

    router = MLPRouter(embed_dim, low_rank_dim, out_dim, act_th, device=device, dtype=dtype)
    state_dict = torch.load(router_path, map_location=device, weights_only=True)
    router.load_state_dict(state_dict)

    return router

def compute_recall(router_output, ground_truth, k, eps=1e-8):
    """
    Computes recall for router predictions.

    Args:
        router_output (torch.Tensor): Router's output of shape (B, N).
        ground_truth (torch.Tensor): True activations of shape (B, N), where values > 0 indicate active neurons.
        k (int): Number of top neurons to select per sample.
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple: (precision, recall) averaged over the batch.
    """
    # Create a binary mask for ground truth activations
    gt_mask = (ground_truth > 0).float()

    # Get top-k indices per sample from router output
    topk_indices = torch.topk(router_output, k, dim=1).indices

    # Create prediction mask: mark top-k as active (1)
    pred_mask = torch.zeros_like(router_output)
    pred_mask.scatter_(1, topk_indices, 1)

    # Compute true positives per sample
    true_positives = (pred_mask * gt_mask).sum(dim=1)

    # Precision: fraction of top-k that are correct
    # precision = true_positives / k

    # Recall: fraction of ground truth active neurons that are predicted
    recall = true_positives / (gt_mask.sum(dim=1) + eps)

    # return precision.mean().item(), recall.mean().item()
    return recall.mean().item()


def _optimize_layer(args, layer_idx, starting_k = 128, target_recall = 0.99, delta = 100, min_k = 1024, max_k = 4096):
    '''
    This function optimizes the router for a single layer.
    Given the hidden states and true activations, it uses the model to predict the activations.
    The top-k values of the router outputs are used as the predictions for the activations.
    It then computes the precision and recall of the predictions.
    While the recall is less than the target recall, the model is optimized.
    
    '''
    
    # load the router for the layer
    mlp_router_layer = load_mlp_router(
        layer_id=layer_idx,
        directory= args.mlp_ckpt_dir,
        embed_dim=args.embed_dim,
        low_rank_dim=1024,
        out_dim=args.total_neurons,
        act_th=0.5,
        device="cuda",
        dtype=torch.float16,
    )
    
    hidden_states, true_activations = get_data_(args.act_data_dir,
                                                layer_idx=layer_idx,
                                                data_type='mlp_activations',
                                                total_neurons=args.total_neurons,
                                                total_samples=args.total_samples)
    
    hidden_states = torch.from_numpy(hidden_states).to("cuda")
    hidden_states.requires_grad = False
    true_activations = torch.from_numpy(true_activations).to("cuda")
    true_activations.requires_grad = False

    current_recall = 0
    k = starting_k
    
    while current_recall < target_recall:

        with torch.no_grad():
            predictions = mlp_router_layer(hidden_states)
        recall = compute_recall(predictions, true_activations, k=k)
        if recall < target_recall:
            k = k + delta
        current_recall = recall        
        # print(f"Layer {layer_idx}: Recall = {recall:.4f} at k = {k}")
        
    k = int(np.ceil(k /128) * 128)
    k = min(max_k, max(min_k, k)) 
    return k

def _optimize_router(args):
    starting_topk_layer = build_mlp_topk_lookup(data_path=args.stats_dir, batch_size=args.batch_size, delta=0)
    total_layers = args.num_layers
    optim_k = {}
    
    for layer_idx in tqdm(range(total_layers), desc="Optimizing MLP Router"):
        optim_k[layer_idx] = _optimize_layer(args, layer_idx, starting_k = starting_topk_layer[layer_idx],
                                            target_recall=args.target_recall, delta=args.delta,
                                            min_k=args.min_k, max_k=args.max_k
                                            )

    # save the optimized topk values in a csv file
    save_dict_to_csv(optim_k, args.batch_size, args.model_name, args.results_dir)


def compute_recall_batch(router_output, ground_truth, k, eps=1e-8):
    # Ground truth: if any sample in the batch is active, mark neuron as active.
    gt_sum = (ground_truth > 0).float().sum(dim=0)
    gt_mask = (gt_sum >= 1).float()
    # Prediction: zero-out negatives, sum over batch, then select top-k neurons.
    neurons_nonzero = torch.nn.ReLU()(router_output)
    pred_sum = neurons_nonzero.sum(dim=0)
    _, topk_indices = torch.topk(pred_sum, k, sorted=False)
    pred_mask = torch.zeros_like(pred_sum)
    pred_mask[topk_indices] = 1.0
    true_positives = (pred_mask * gt_mask).sum()
    recall = true_positives / (gt_mask.sum() + eps)
    return recall.item()

def _optimize_layer_batched(args, layer_idx, starting_k=128, target_recall=0.99, delta=100, min_k=512, max_k=4096):
    mlp_router_layer = load_mlp_router(
        layer_id=layer_idx,
        directory=args.mlp_ckpt_dir,
        embed_dim=args.embed_dim,
        low_rank_dim=1024,
        out_dim=args.total_neurons,
        act_th=0.5,
        device="cuda",
        dtype=torch.float16,
    )
    
    hidden_states, true_activations = get_data_(args.act_data_dir,
                                                layer_idx=layer_idx,
                                                data_type='mlp_activations',
                                                total_neurons=args.total_neurons,
                                                total_samples=args.total_samples)
    hidden_states = torch.from_numpy(hidden_states).to("cuda")
    true_activations = torch.from_numpy(true_activations).to("cuda")
    
    total_samples = hidden_states.shape[0]
    batch_size = args.batch_size
    num_batches = total_samples // batch_size
    k = starting_k
    current_recall = 0
    
    while current_recall < target_recall:
        batch_recalls = []
        indices = torch.randperm(total_samples)
        for i in range(num_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            hs_batch = hidden_states[batch_idx]
            gt_batch = true_activations[batch_idx]
            with torch.no_grad():
                predictions = mlp_router_layer(hs_batch)
            batch_recall = compute_recall_batch(predictions, gt_batch, k)
            batch_recalls.append(batch_recall)
        current_recall = sum(batch_recalls) / len(batch_recalls)
        if current_recall < target_recall:
            k += delta
    k = int(np.ceil(k / 128) * 128)
    # k = min(max_k, max(min_k, k))
    return k

def _optimize_router_batched(args):
    if args.stats_dir is not None:
        starting_topk_layer = build_mlp_topk_lookup(data_path=args.stats_dir, batch_size=args.batch_size, delta=0)
    else:
        starting_topk_layer = build_default_mlp_topk_lookup(args.num_layers, default_topk=args.min_k)
    
    total_layers = args.num_layers
    optim_k = {}
    for layer_idx in tqdm(range(total_layers), desc="Optimizing MLP Router (Batched)"):
        optim_k[layer_idx] = _optimize_layer_batched(args, layer_idx, starting_k=starting_topk_layer[layer_idx],
                                                     target_recall=args.target_recall, delta=args.delta,
                                                     min_k=args.min_k, max_k=args.max_k)
    save_dict_to_csv(optim_k, args.batch_size, args.model_name, args.results_dir)


def arg_parser():
    parser = argparse.ArgumentParser(description=' MLP Router Optimizations')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='configs/mlp_router')
    parser.add_argument('--stats_dir', type=str, default=None)
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_ROUTER_CHECKPOINTS>')
    parser.add_argument('--act_data_dir', type=str, default='<PATH_TO_ACTIVATION_DATA>')
    parser.add_argument('--max_batch_size', type=int, default=32)
    parser.add_argument('--total_samples', type=int, default=50000)
    parser.add_argument('--target_recall', type=float, default=0.99)
    parser.add_argument('--delta', type=int, default=100)
    parser.add_argument('--min_k', type=int, default=512)
    parser.add_argument('--max_k', type=int, default=8192)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--mode', type=str, default='row', choices=['row', 'col', 'auto'])
    
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    
    # model config
    model_name = MODELS[args.model_index-1]
    config = CONFIGS[model_name]
    args.model_name = model_name
    args.embed_dim = config['d']
    args.total_neurons = config['neurons']
    args.num_layers = config['num_layer']
    
    if args.batch_size == 1:
        _optimize_router(args)
    else:
        _optimize_router_batched(args)
    