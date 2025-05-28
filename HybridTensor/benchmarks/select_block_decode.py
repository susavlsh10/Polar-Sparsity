import torch

from tests.test_select_block import create_block, Config, SparseConfig
import csv
import time
import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams
from HybridTensor.utils.utils import arg_parser, _get_device, sparse_index, generate_random_BH_index, get_gpu_name
from HybridTensor.utils.profiling import cuda_profiler
import math
from tqdm import tqdm

def run_simulation(args, batch_size, seq_len, index_size, attn_topk, device, dtype):
    config = Config()
    sp_config = SparseConfig()
    sp_config.attn_topk = attn_topk
    
    config.hidden_size = args.in_features
    config.num_attention_heads = args.in_features // 128
    config.use_heuristic = False    # use pre-compiled heuristic or complie new one during runtime

    # Test create_block
    sparse_block = create_block(config, sp_config, layer_idx=0, process_group=None, device=device, dtype=dtype)
    sparse_block.eval()
    sparse_block.mlp_topk = index_size
    
    regular_config = config
    regular_config.att_sparse = False
    regular_config.mlp_sparse = False
    regular_block = create_block(regular_config, None, layer_idx=0, process_group=None, device=device, dtype=dtype)
    regular_block.eval()
    
    # inference simulation with select block
    max_seqlen = seq_len + 16
    max_batch_size = batch_size
    in_features = args.in_features
    head_dim = 128
    
    inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=max_batch_size)
    process_group = None
    sequence_parallel = False
    
    # for testing and debugging
    heads = config.num_attention_heads
    selected_heads = heads // 2
    
    # Create a static index vector (length equals total columns in B).
    total_neurons = args.in_features * 4
    test_index_vec = torch.empty((total_neurons,), device='cuda', dtype=torch.int32)
    active_indices = sparse_index(args.index_size, total_neurons)[0]
    test_index_vec[:args.index_size] = active_indices
    if args.index_size < total_neurons:
        test_index_vec[args.index_size:] = 0  # Fill the rest with dummy values.
    
    # test_index_vec = sparse_index(args.in_features, args.in_features*4)[0].cuda()
    test_bh_idx = generate_random_BH_index(args.batch_size, heads, selected_heads)
    test_index_size = args.index_size
    
    mixer_kwargs = (
        {"seqlen": seq_len}
        if process_group is not None and sequence_parallel
        else {}
    )
    if inference_params is not None:
        mixer_kwargs["inference_params"] = inference_params
    
    with torch.no_grad():
        # prefill stage 
        original_seq = torch.randn(batch_size, seq_len, in_features, device='cuda', dtype=torch.float16)
            
        # Test prefill
        output_sparse = sparse_block(original_seq, mixer_kwargs=mixer_kwargs)
        output_regular = regular_block(original_seq, mixer_kwargs=mixer_kwargs)
        
        # need to update inference_params to reflect the new sequence length
        mixer_kwargs["inference_params"].seqlen_offset = seq_len
        
        # Decode stage  
        input_x = torch.randn(batch_size, 1, in_features, device='cuda', dtype=torch.float16)
        
        out_decode_sparse = sparse_block(input_x, mixer_kwargs=mixer_kwargs)
        
        mixer_kwargs["inference_params"].seqlen_offset = seq_len
        
        out_decode_regular = regular_block(input_x, mixer_kwargs=mixer_kwargs)
        
        # mesure decode stage time in ms
        # print("Without CUDA Graphs")
        # out_decode_regular, regular_time = cuda_profiler(regular_block, input_x, mixer_kwargs=mixer_kwargs, warmup_runs=1, timed_runs=2)
        # print(f"Regular time: {regular_time} ms")
        
        # out_decode_sparse, sparse_time = cuda_profiler(sparse_block, input_x, mixer_kwargs=mixer_kwargs, warmup_runs=1, timed_runs=2)
        # print(f"Sparse time: {sparse_time} ms")
        
        # speedup = regular_time / sparse_time
        # print(f"Speedup: {speedup}")
        
        # --- CUDA Graph Capture for Decode Stage ---
        # Allocate static buffer for regular block (shape assumed fixed)
        input_x_static = input_x.clone()
        output_regular_static = torch.empty((batch_size, 1, in_features), device=device, dtype=dtype)

        # Capture regular block graph
        _ = regular_block(input_x_static, mixer_kwargs=mixer_kwargs)
        torch.cuda.synchronize()
        graph_regular = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_regular):
            res = regular_block(input_x_static, mixer_kwargs=mixer_kwargs)
            if isinstance(res, tuple):
                res = res[0]
            output_regular_static.copy_(res)

        # For the sparse block, run a dummy call to determine its output shape.
        # Also, reset the inference parameter to ensure consistent behavior.
        mixer_kwargs["inference_params"].seqlen_offset = seq_len
        temp = sparse_block(input_x_static, mixer_kwargs=mixer_kwargs)
        if isinstance(temp, tuple):
            temp = temp[0]
        # print("Captured sparse block output shape:", temp.shape)
        # Allocate static buffer matching the dummy run's shape.
        output_sparse_static = torch.empty_like(temp)
        # print("output_sparse_static shape:", output_sparse_static.shape)
        torch.cuda.synchronize()
        
        mixer_kwargs["inference_params"].seqlen_offset = seq_len
        graph_sparse = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_sparse):
            res = sparse_block(input_x_static, mixer_kwargs=mixer_kwargs)
            if isinstance(res, tuple):
                res = res[0]
            output_sparse_static.copy_(res)

        # Warmup CUDA Graph replays
        for _ in range(5):
            graph_regular.replay()
            graph_sparse.replay()
        torch.cuda.synchronize()

        # --- Measure CUDA Graph Replay Latency ---
        num_replays = 10

        start = time.time()
        for _ in range(num_replays):
            graph_regular.replay()
        torch.cuda.synchronize()
        regular_graph_time = (time.time() - start) * 1000 / num_replays

        start = time.time()
        for _ in range(num_replays):
            graph_sparse.replay()
        torch.cuda.synchronize()
        sparse_graph_time = (time.time() - start) * 1000 / num_replays
        speedup = regular_graph_time / sparse_graph_time
        # print()
        # print("With CUDA Graphs")
        # print(f"Regular block time (CUDA Graphs): {regular_graph_time} ms")
        # print(f"Sparse block time (CUDA Graphs): {sparse_graph_time} ms")
        # print(f"Speedup (CUDA Graphs): {speedup}")
        
    return regular_graph_time, sparse_graph_time, speedup

if __name__ == "__main__":
    
    args = arg_parser()
    device = _get_device(0)
    dtype = torch.float16
    gpu_name = get_gpu_name()

    # Parameter grids.
    # batch_sizes = [1, 4, 8, 16]
    # seq_lengths = [128, 512]
    # index_sizes = [512, 1024, 2048, 4096]
    # attn_topks = [0.3, 0.4, 0.5]

    batch_sizes = [1, 8, 16, 32]
    seq_lengths = [1024, 2048]
    # index_sizes = [512, 1024, 2048, 4096, 8192]
    index_size_p = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    total_neurons = args.in_features * 4
    
    # Calculate initial index_size values
    index_sizes = [int(total_neurons * i) for i in index_size_p]

    # Round up to the nearest multiple of 128 if necessary
    index_sizes = [math.ceil(size / 128) * 128 if size % 128 != 0 else size for size in index_sizes]
    
    attn_topks = [0.3, 0.4, 0.5]

    # Calculate total number of simulations.
    total_runs = len(batch_sizes) * len(seq_lengths) * len(index_sizes) * len(attn_topks)
    output_file = f"results/simulations/{gpu_name}_select_block_{args.in_features}_inference_sim.csv"

    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ["in_features", "batch_size", "seq_len", "index_size", "neuron_activation", "attn_topk",
                      "regular_graph_time_ms", "sparse_graph_time_ms", "speedup"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over all combinations with tqdm progress bar.
        for batch_size in tqdm(batch_sizes, desc="Batch Sizes"):
            for seq_len in seq_lengths:
                for index_size in index_sizes:
                    for attn_topk in attn_topks:
                        reg_time, spa_time, speedup = run_simulation(args, batch_size, seq_len, index_size, attn_topk, device, dtype)
                        writer.writerow({
                            "in_features": args.in_features,
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "index_size": index_size,
                            "neuron_activation": index_size / total_neurons,
                            "attn_topk": attn_topk,
                            "regular_graph_time_ms": reg_time,
                            "sparse_graph_time_ms": spa_time,
                            "speedup": speedup
                        })
                        csv_file.flush()
    print(f"Simulation complete. Results saved to {output_file}")