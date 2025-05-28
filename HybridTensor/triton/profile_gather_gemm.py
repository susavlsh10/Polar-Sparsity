import torch
import os
import csv

# from HybridTensor.triton.heuristics.gather_gemm_col_h import gather_matmul_col
from HybridTensor.triton.gather_gemm_col import gather_matmul_col
from HybridTensor.triton.references.gemm import triton_matmul
from HybridTensor.triton.triton_utils import get_autotune_config, benchmark_fwd, torch_sel_gemm_col
from HybridTensor.utils.utils import arg_parser, sparse_index, get_gpu_name
from HybridTensor.utils.profiling import cuda_profiler

if __name__ == "__main__":
    args = arg_parser()
    torch.manual_seed(0)
    args.hidden_features = args.in_features * 4
    A = torch.randn(args.batch_size, args.in_features, device='cuda', dtype=torch.float16)
    B = torch.randn(args.in_features, args.hidden_features, device='cuda', dtype=torch.float16)
    B_col_major = B.t().contiguous()
    index_vec = sparse_index(args.index_size, args.hidden_features)[0]
    if args.bias:
        print("Using bias")
        bias = torch.randn(args.hidden_features, device='cuda', dtype=torch.float16)
    else:
        print("No bias")
        bias = None
    print(f"args: {args}")
    print(f"Index size: {args.index_size}, Activated {args.index_size/(args.in_features * 4)*100}% neurons")
    
    index_sizes_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    index_sizes = [int(args.hidden_features * ratio) for ratio in index_sizes_ratio]
    # index_sizes = [((size + 127) // 128) * 128 for size in index_sizes]  # Round up to the nearest multiple of 128
    results = []
    print(f"Index sizes: {index_sizes}")
    
    linear_layer = torch.nn.Linear(args.in_features, args.hidden_features, bias = args.bias, device = 'cuda', dtype = torch.float16)
    out_b, dense_triton_time = cuda_profiler(triton_matmul, A, B, bias=None, warmup_runs=1, timed_runs=10)
    out_b, cublass_time = cuda_profiler(linear_layer, A, warmup_runs=1, timed_runs=10)
    # out_b, cublass_time = benchmark_fwd(A, B, bias = None, function = torch.linear, print_result = args.print_results)
    for idx, index_size in enumerate(index_sizes):
        
        index_vec = sparse_index(index_size, args.hidden_features)[0]
        
        out_a_gather, gather_gemm_time = benchmark_fwd(A, B_col_major, bias = bias, index_vec=index_vec, function=gather_matmul_col, print_result = args.print_results)
        out_a_gather_torch,torch_sel_gemm_time  = benchmark_fwd(A, B, bias = bias, index_vec=index_vec, function=torch_sel_gemm_col, print_result = args.print_results)

        speedup = cublass_time / gather_gemm_time
        print(f"Speedup: {speedup:.2f}")
        results.append({
            "Neuron Activation": index_sizes_ratio[idx],
            "Dense (Cublass) Time (ms)": round(cublass_time, 4),
            "Dense (Triton) Time (ms)": round(dense_triton_time, 4),
            "Gather GEMM Time (ms)": round(gather_gemm_time, 4),
            "Naive(Torch Gather) Time (ms)": round(torch_sel_gemm_time, 4),
            "Speedup": round(speedup, 4)
        })
    args.results_dir = f"results/triton_kernels/{get_gpu_name()}/"
    os.makedirs(args.results_dir, exist_ok=True)
    csv_file_path = os.path.join(args.results_dir, f"gather_gemm_profiling_{args.in_features}_{args.batch_size}.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_file_path}")