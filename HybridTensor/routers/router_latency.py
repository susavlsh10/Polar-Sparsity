import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import os

'''
python -m HybridTensor.routers.router_latency --block mlp --D 1024 --gpu 0 --model_dim 9216 --out_dim $((9216*4))
python -m HybridTensor.routers.router_latency --block attn --D 1024 --gpu 0 --model_dim 9216 --heads 72

'''

from HybridTensor.routers.router_utils import DATA, MODEL_CHOICES, DATA_CHOICES, CONFIG
from HybridTensor.utils.profiling import cuda_profiler
from HybridTensor.modules.SelectiveMLP import MLPRouter
from HybridTensor.modules.SelectiveMHA import MHARouter
from HybridTensor.routers.archive.fused_router.router import FusedRouter, HybridFusedRouter

# ---------------------------
# Latency Measurement Function
# ---------------------------

class Router(torch.nn.Module):
    def __init__(self, args):
        super(Router, self).__init__()
        self.fc1 = torch.nn.Linear(args.model_dim, args.D, bias=None)
        # self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(args.D)
        # output dimension
        if args.block == "mlp":
            out_dim = args.out_dim 
        else:   # args.block == "attn"
            out_dim = args.heads
        self.fc2 = torch.nn.Linear(args.D, out_dim, bias=None)
        
    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn(x)
        x = self.fc2(x)
        return x


def measure_inference_latency(model, input_data, device, num_runs=100, warm_up=10):
    """
    Measures the average inference latency of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        input_data (torch.Tensor): Input tensor for the model.
        device (torch.device): The device to perform inference on.
        num_runs (int, optional): Number of forward passes to measure. Defaults to 100.
        warm_up (int, optional): Number of warm-up runs before measurement. Defaults to 10.

    Returns:
        float: Average inference latency in milliseconds.
    """
    model.eval()  # Set model to evaluation mode
    input_data = input_data.to(device)
    model.to(device)
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    # Warm-up runs (not timed)
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(input_data)
    
    # Start timing
    start_time.record()
    with torch.no_grad():
        for _ in range(num_runs):
            out = model(input_data)
            
    end_time.record()
    torch.cuda.synchronize()
    
    avg_latency = start_time.elapsed_time(end_time) / num_runs
    
    return avg_latency


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch OPT Full Model")
    parser.add_argument("--model", type=str, default="6_7b", choices=MODEL_CHOICES)
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--block", type=str, default="mlp", choices=["mlp", "attn"])
    parser.add_argument("--L", type=int, default=0, help="which layer")
    parser.add_argument("--D", type=int, default=1024, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")  # a smaller batch size results in better precision and recall of the model
    parser.add_argument("--ckpt_dir", type=str, default="/home/grads/s/<name>/nvme/HybridTensor/checkpoint", help="checkpoint directory")  # add a argument for checkpoint dir
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")  # add a argument for which gpu to use
    parser.add_argument("--model_dim", type=int, default=8192, help="model dimension")
    parser.add_argument("--heads", type=int, default=64, help="number of heads")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--out_dim", type=int, default=32768, help="output dimension")
    args = parser.parse_args()
    return args


# ---------------------------
# 4. Main Execution
# ---------------------------

if __name__ == "__main__":
    # 4.1. Define Arguments
    args = arg_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(args)
    # 4.2. Instantiate the Model
    # router_model = Router(args)
    mlp_router = MLPRouter(embed_dim=args.model_dim, low_rank_dim=args.D, out_dim=args.out_dim, act_th=0.5, device="cuda:0", dtype=torch.float16)
    mha_router = MHARouter(embed_dim=args.model_dim, low_rank_dim=128, out_dim=args.heads, top_k=0.3, device="cuda:0", dtype=torch.float16)
    fused_router = FusedRouter(embed_dim=args.model_dim, num_heads=args.heads, dim=args.D, mlp_th=0.5, attn_top_k=0.3, device="cuda:0", dtype=torch.float16)
    hybrid_fused_router = HybridFusedRouter(embed_dim=args.model_dim, num_heads=args.heads, mlp_dim=1024, mha_dim=128, mlp_th=0.5, attn_top_k=0.3, device="cuda:0", dtype=torch.float16)
    
    # 4.3. Select Device
    print(f"Using device: {device}")
    print(f"Model dimension: {args.model_dim}")
    print(f"Block: {args.block}")
    print(f"Low rank dimension: {args.D}")
        
    
    # 4.4. Prepare Dummy Input Data
    # batch_size = args.batch_size  # You can adjust the batch size as needed
    # input_dim = CONFIG[args.model]['d']
    input_dim = args.model_dim
    
    # batch_range= [1, 8, 16, 32, 64]
    batch_range = [64]
    for batch_size in batch_range:
        test_input = torch.randn(batch_size, input_dim, device = device, dtype = torch.float16)  # Random input tensor
        
        selected_neurons, mlp_router_time = cuda_profiler(mlp_router._select_neurons, test_input)
        selected_heads, mha_router_time = cuda_profiler(mha_router._select_heads, test_input)
        fused_activations, fused_router_time = cuda_profiler(fused_router.select_neurons_heads, test_input)
        hybrid_fused_activations, hybrid_fused_router_time = cuda_profiler(hybrid_fused_router.select_neurons_heads, test_input)
        hybrid_mha_activations, hybrid_mha_router_time = cuda_profiler(hybrid_fused_router.select_heads_, test_input)
        hybrid_mlp_activations, hybrid_mlp_router_time = cuda_profiler(hybrid_fused_router.select_neurons_, test_input)
        
        # 4.5. Measure Inference Latency
        # avg_latency = measure_inference_latency(
        #     model=router_model,
        #     input_data=test_input,
        #     device=device,
        #     num_runs=100,   # Number of measurements
        #     warm_up=10      # Number of warm-up runs
        # )
        
        total_router_time = mlp_router_time + mha_router_time
        total_hybrid_router_time = hybrid_mlp_router_time + hybrid_mha_router_time
        difference = total_router_time - total_hybrid_router_time
        
        print(f"Batch_size: {batch_size}, MLP Router latency: {mlp_router_time:.4f} ms")
        print(f"Batch_size: {batch_size}, MHA Router latency: {mha_router_time:.4f} ms")
        print(f"Batch_size: {batch_size}, Total Router latency: {total_router_time:.4f} ms")
        # print(f"Batch_size: {batch_size}, Fused Router latency: {fused_router_time:.4f} ms")
        
        # print(f"Batch_size: {batch_size}, Hybrid Fused Router latency: {hybrid_fused_router_time:.4f} ms")
        # print(f"Batch_size: {batch_size}, Hybrid MHA Router latency: {hybrid_mha_router_time:.4f} ms")
        # print(f"Batch_size: {batch_size}, Hybrid MLP Router latency: {hybrid_mlp_router_time:.4f} ms")
        # print(f"Batch_size: {batch_size}, Total Hybrid Router latency: {total_hybrid_router_time:.4f} ms")
        # print(f"Batch_size: {batch_size}, Difference: {difference:.4f} ms")
        print()
