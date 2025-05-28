import math
import torch
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

from HybridTensor.utils.utils import get_gpu_name, create_results_directory

from tqdm import tqdm  # For progress bars

# from HybridTensor.modules.MLP import StandardMLPBlock, SelectiveMLP, SelectiveMLPTriton
# from HybridTensor.utils.utils import sparse_index

def benchmark_mlp_fwd(x, model, index_vec = None, iterations = 100, print_result = False):
    
    class_name_ = model.__class__.__name__
    if index_vec is not None:
        model = partial(model, index_vec=index_vec)
    # warm up, this also compiles the triton kernel before measuring the execution time
    for _ in range(10):
        out = model(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        out = model(x)
    torch.cuda.synchronize()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time = start.elapsed_time(end)/iterations

    if print_result:
        print(f"{class_name_} Execution time: {elapsed_time} ms")
    
    return out, elapsed_time

def generate_index_sizes(hidden_features):
    index_sizes = []
    idx = 0
    while idx < hidden_features:
        idx += 512
        index_sizes.append(min(idx, hidden_features))
    return index_sizes


def save_results_to_csv(df, filename_prefix='mlp_profiling_results', results_dir = create_results_directory("results")):
    """
    Saves the profiling results DataFrame to a CSV file within the specified results directory.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing profiling results.
    - filename_prefix (str): The prefix for the CSV filename.
    - results_dir (Path): The Path object for the results directory.
    """
    
    # Retrieve the GPU name
    gpu_name = get_gpu_name()
    
    # Define the filename with GPU name
    filename = f"{filename_prefix}_{gpu_name}.csv"
    
    # Define the full path for the CSV file
    csv_path = results_dir / filename
    
    # Save the DataFrame to the CSV file
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
def plot_results(df, output_prefix='mlp_profiling', results_dir=create_results_directory("results")):
    """
    Plots the profiling results and saves the plot image within the specified results directory.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing profiling results.
    - output_prefix (str): The prefix for the plot filename.
    - results_dir (Path): The Path object for the results directory.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Execution Time
    plt.subplot(1, 2, 1)
    plt.plot(df['index_size'], df['standard_time'], label='Standard MLP', marker='o')
    plt.plot(df['index_size'], df['selective_cutlass_time'], label='Selective MLP Cutlass', marker='o')
    plt.plot(df['index_size'], df['selective_triton_time'], label='Selective MLP Triton', marker='o')
    plt.xlabel('Index Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs. Index Size')
    plt.legend()
    plt.grid(True)
    
    # Plot Speedup
    plt.subplot(1, 2, 2)
    plt.plot(df['index_size'], df['cutlass_speedup'], label='Cutlass Speedup', marker='o')
    plt.plot(df['index_size'], df['triton_speedup'], label='Triton Speedup', marker='o')
    plt.xlabel('Index Size')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Index Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Retrieve the GPU name
    gpu_name = get_gpu_name()
    
    # Define the filename with GPU name
    plot_filename = f"{output_prefix}_{gpu_name}.png"
    
    # Define the full path for the plot image
    plot_path = results_dir / plot_filename
    
    # Save the plot
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved as {plot_path}")

def cuda_profiler(func, *args, warmup_runs=10, timed_runs=1000, **kwargs):
    """
    Generic profiler function to measure execution time of a given function.

    Parameters:
    - func: The function to be profiled.
    - *args: Positional arguments to be passed to the function.
    - warmup_runs: Number of warm-up runs (default: 2).
    - timed_runs: Number of timed iterations (default: 10).
    - **kwargs: Keyword arguments to be passed to the function.

    Returns:
    - float: The average execution time of the function in milliseconds.
    """
    # Warm-up phase
    for _ in range(warmup_runs):
        out = func(*args, **kwargs)

    # Synchronize before starting the timer to ensure accurate measurements
    torch.cuda.synchronize()

    # Create CUDA events for measuring execution time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record execution times for the given number of runs
    start_event.record()
    for _ in range(timed_runs):
        # Execute the function
        out = func(*args, **kwargs)

    # Wait for the events to be completed
    # torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    # Calculate elapsed time for this iteration
    elapsed_time = start_event.elapsed_time(end_event)

    # Calculate average time per run
    avg_time = elapsed_time / timed_runs

    return out, avg_time


from HybridTensor.triton.select_attn_v1 import select_attn
from HybridTensor.utils.utils import generate_BH_index
from HybridTensor.triton.references.attention_proj_sparse import qkv_proj_sparse, out_proj_sparse

def _sim_cache_update(k, v, qkv, seq_len):
    k[:, -1, ...] = qkv[:, :, 1]
    v[:, -1, ...] = qkv[:, :, 2]

def mha_inference_simulation(B, in_features, seq_len, head_density, active_density):
    ''' 
    Simulates the execultion time of a standard MHA layer and a selective MHA layer with sparse projection and select_attn.
    
    Parameters:
    -   B: batch size
    -   in_features: number of features
    -   seq_len: sequence length
    -   head_density: the percentage of heads that are active per batch 
    -   active_density: the percentage of active heads per layer (aggregate active heads in all batches)
    
    '''
    # Test parameters
    H = in_features // 128  # Number of heads
    G = 1  # Group size
    M = 1  # Sequence length for queries
    Mk = seq_len  # Sequence length for keys/values
    Kq = 128  # Embedding size for queries
    Kkv = 128  # Embedding size for keys/values

    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = 'cuda:0'
    dtype = torch.float16

    x = torch.rand(B, in_features, dtype=dtype).to(device)
    proj_dim = 3 * in_features

    # Define the Linear layer
    qkv_project = torch.nn.Linear(in_features, proj_dim, dtype=dtype).to(device)
    out_project = torch.nn.Linear(in_features, in_features, dtype=dtype).to(device)

    weight = torch.randn(3, H, Kkv, in_features, device=device, dtype=dtype)
    bias = torch.randn(3, H, Kkv, device=device, dtype=dtype)

    n_active_heads = math.ceil(H * active_density)
    head_idx = torch.randperm(H, device=device, dtype=torch.int32)[:n_active_heads]

    batch_idx = torch.stack([
        torch.arange(B, dtype=torch.int32, device=device)
        for _ in range(n_active_heads)
    ])

    print(f"Batch size: {B}, Total heads: {H}, Features: {in_features}, Seq len: {seq_len}")
    print(f"Total active heads: {n_active_heads}")
    print(f"Head density in SelectAttn: {head_density}")
    
    print("====================================")

    # Inference simulation
    qkv, qkv_project_time = cuda_profiler(qkv_project, x)
    print(f"qkv projection time: {qkv_project_time:.3f} ms")

    qkv_sel, qkv_sel_proj_time = cuda_profiler(qkv_proj_sparse, x, weight, head_idx, batch_idx, bias)
    print(f"qkv projection time: {qkv_sel_proj_time:.3f} ms")

    # Generate random tensors for q, k, v
    q = torch.randn(B, M, G, H, Kq, dtype=dtype, device=device)
    k = torch.randn(B, Mk, G, H, Kkv, dtype=dtype, device=device)
    v = torch.randn(B, Mk, G, H, Kkv, dtype=dtype, device=device)

    # need to update kv cache with the new k, v
    _sim_cache_update(k, v, qkv, seq_len)
    _, kv_cache_update_time = cuda_profiler(_sim_cache_update, k, v, qkv, seq_len)
    print(f"KV cache update time: {kv_cache_update_time:.3f} ms")

    scale = 1 / (Kq ** 0.5)
    batch_head_index_1 = generate_BH_index(B, H, math.ceil(H * 1))

    triton_sel_output, attn_time = cuda_profiler(select_attn, q, k, v, scale, batch_head_index_1)
    print(f"Attention time: {attn_time:.3f} ms")

    batch_head_index_2 = generate_BH_index(B, H, math.ceil(H * head_density))
    triton_sel_output_2, select_attn_time = cuda_profiler(select_attn, q, k, v, scale, batch_head_index_2)
    print(f"SelectAttn time: {select_attn_time:.3f} ms")

    triton_sel_output_2, view_time = cuda_profiler(triton_sel_output_2.view, B, in_features)

    # Out projection
    out, out_project_time = cuda_profiler(out_project, triton_sel_output_2)
    print(f"out projection time: {out_project_time:.3f} ms")

    standard_time = qkv_project_time + attn_time + out_project_time
    select_time = qkv_project_time + select_attn_time + out_project_time
    select_time_sparse_project = qkv_sel_proj_time + select_attn_time + out_project_time

    print("====================================")
    print(f"Standard time: {standard_time:.3f} ms")
    print(f"Select time: {select_time:.3f} ms")
    print(f"Select time with sparse project: {select_time_sparse_project:.3f} ms")
    print("====================================")
    print(f"Selective Speedup: {standard_time / select_time:.3f}")
    print(f"Selective Speedup with sparse project: {standard_time / select_time_sparse_project:.3f}")
    
    # free cuda memory
    del qkv, qkv_sel, q, k, v, triton_sel_output, triton_sel_output_2, out
    torch.cuda.empty_cache()
    
