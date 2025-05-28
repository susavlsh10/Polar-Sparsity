# python -m HybridTensor.routers.mlp.mlp_router_optim_fast --model_index 5 --batch_size 256 --delta 200 --device 0
from HybridTensor.routers.router_utils import get_data_
from HybridTensor.modules.SelectiveMLP import MLPRouter
from HybridTensor.utils.activations import MODELS, CONFIGS
from HybridTensor.utils.activations import build_mlp_topk_lookup
from HybridTensor.utils.utils import extract_model_name
from HybridTensor.routers.mlp.mlp_router_optim import save_dict_to_csv, load_router_dict_from_csv, load_mlp_router, _optimize_layer, _optimize_router
from HybridTensor.utils.utils import _get_device

import os
import torch
import argparse
import re
import numpy as np
import csv
from tqdm import tqdm

# --- New vectorized batch recall function ---
def vectorized_compute_recall_batch(predictions_batches, gt_batches, k, eps=1e-8):
    # Aggregate ground truth: if any sample in the batch is active, mark neuron as active.
    gt_mask = ((gt_batches > 0).float().sum(dim=1) >= 1).float()  # shape: (num_batches, num_neurons)
    # Sum predictions over batch dimension (predictions are already ReLU-ed)
    pred_sum = predictions_batches.sum(dim=1)  # shape: (num_batches, num_neurons)
    topk_indices = pred_sum.topk(k, dim=1).indices  # shape: (num_batches, k)
    pred_mask = torch.zeros_like(pred_sum)
    pred_mask.scatter_(1, topk_indices, 1)
    true_positives = (pred_mask * gt_mask).sum(dim=1)
    batch_recall = true_positives / (gt_mask.sum(dim=1) + eps)
    return batch_recall.mean().item()

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

def _optimize_layer_batched(args, layer_idx, starting_k=128, target_recall=0.99, delta=100, min_k=512, max_k=4096):
    device = _get_device(args.device)
    
    mlp_router_layer = load_mlp_router(
        layer_id=layer_idx,
        directory=args.mlp_ckpt_dir,
        embed_dim=args.embed_dim,
        low_rank_dim=1024,
        out_dim=args.total_neurons,
        act_th=0.5,
        device=device,
        dtype=torch.float16,
    )
    
    
    
    hidden_states, true_activations = get_data_(args.act_data_dir,
                                                layer_idx=layer_idx,
                                                data_type='mlp_activations',
                                                total_neurons=args.total_neurons,
                                                total_samples=args.total_samples)
    hidden_states = torch.from_numpy(hidden_states).to(device)
    true_activations = torch.from_numpy(true_activations).to(device)
    
    with torch.no_grad():
        predictions_all = mlp_router_layer(hidden_states)
        predictions_all = torch.nn.functional.relu(predictions_all)
    
    total_samples = hidden_states.shape[0]
    batch_size = args.batch_size
    num_batches = total_samples // batch_size
    k = starting_k
    max_topk = starting_k + args.lambda_param  # New maximum topk constraint
    current_recall = 0.0
    
    while current_recall < target_recall and k < max_topk:
        indices = torch.randperm(total_samples)
        predictions_shuffled = predictions_all[indices]
        true_activations_shuffled = true_activations[indices]
        trimmed = num_batches * batch_size
        predictions_batches = predictions_shuffled[:trimmed].view(num_batches, batch_size, -1)
        gt_batches = true_activations_shuffled[:trimmed].view(num_batches, batch_size, -1)
        current_recall = vectorized_compute_recall_batch(predictions_batches, gt_batches, k)
        if current_recall < target_recall:
            k += delta
            if k >= max_topk:
                k = max_topk
                break
    k = int(np.ceil(k / 128) * 128)
    print(f"Layer {layer_idx}: k = {k}, recall = {current_recall}")
    # k = min(max_k, max(min_k, k))
    return k

def _optimize_router_batched(args):
    # starting_topk_layer = build_mlp_topk_lookup(data_path=args.stats_dir, batch_size=args.batch_size, delta=0)
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
    parser = argparse.ArgumentParser(description='MLP Router Optimizations')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='configs/mlp_router')
    parser.add_argument('--stats_dir', type=str, default=None)
    parser.add_argument('--mlp_ckpt_dir', type=str, default='<PATH_TO_MLP_CHECKPOINT_DIR>')
    parser.add_argument('--act_data_dir', type=str, default='<PATH_TO_ACT_DATA_DIR>')
    parser.add_argument('--max_batch_size', type=int, default=32)
    parser.add_argument('--total_samples', type=int, default=50000)
    parser.add_argument('--target_recall', type=float, default=0.99)
    parser.add_argument('--delta', type=int, default=100)
    parser.add_argument('--lambda_param', type=int, default=2048,
                        help='Maximum extra topk offset allowed (max_topk = starting_k + lambda_param)')
    parser.add_argument('--min_k', type=int, default=512)
    parser.add_argument('--max_k', type=int, default=4096)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--mode', type=str, default='row', choices=['row', 'col', 'auto'])
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    
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
