import torch
import numpy as np
import os
import csv

from HybridTensor.triton.select_attn_v1 import select_attn
from HybridTensor.triton.references.attn_decode import attention as triton_attn
from flash_attn import flash_attn_with_kvcache
from HybridTensor.utils.utils import arg_parser, generate_random_BH_index, get_gpu_name
from HybridTensor.utils.profiling import cuda_profiler



def perform_attention_with_selection(q, k, v, k_cache, v_cache, batch_head_index, causal, cache_seqlens):
    # Expand head_idx so it matches the dimensions of the tensor
    # For q: shape (B, 1, num_heads, head_dim)
    head_idx = batch_head_index.to(torch.int64)
    q_selected = torch.gather(
        q, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(q.size(0), q.size(1), head_idx.size(1), q.size(-1))
    )

    # For k_cache and v_cache: shape (B, seq_len, num_heads, head_dim)
    k_cache_selected = torch.gather(
        k_cache, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(k_cache.size(0), k_cache.size(1), head_idx.size(1), k_cache.size(-1))
    )
    v_cache_selected = torch.gather(
        v_cache, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(v_cache.size(0), v_cache.size(1), head_idx.size(1), v_cache.size(-1))
    )

    # Similarly for k and v: shape (B, 1, num_heads, head_dim)
    k_selected = torch.gather(
        k, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(k.size(0), k.size(1), head_idx.size(1), k.size(-1))
    )
    v_selected = torch.gather(
        v, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(v.size(0), v.size(1), head_idx.size(1), v.size(-1))
    )

    # Perform attention
    out = flash_attn_with_kvcache(
        q=q_selected, 
        k=k_selected, 
        v=v_selected, 
        k_cache=k_cache_selected, 
        v_cache=v_cache_selected, 
        causal=causal, 
        cache_seqlens=cache_seqlens
    )
    
    return out

def triton_attention_with_selection(q, k_cache, v_cache, batch_head_index, scale_float):
    # Expand head_idx so it matches the dimensions of the tensor
    # For q: shape (B, 1, num_heads, head_dim)
    head_idx = batch_head_index.to(torch.int64)
    q_selected = torch.gather(
        q, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(q.size(0), q.size(1), head_idx.size(1), q.size(-1))
    )

    # For k_cache and v_cache: shape (B, seq_len, num_heads, head_dim)
    k_cache_selected = torch.gather(
        k_cache, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(k_cache.size(0), k_cache.size(1), head_idx.size(1), k_cache.size(-1))
    )
    v_cache_selected = torch.gather(
        v_cache, 
        dim=2, 
        index=head_idx.unsqueeze(1).unsqueeze(-1).expand(v_cache.size(0), v_cache.size(1), head_idx.size(1), v_cache.size(-1))
    )

    out = triton_attn(q_selected.unsqueeze(2), k_cache_selected.unsqueeze(2), v_cache_selected.unsqueeze(2), scale_float)
    # triton_attn_out, dense_triton_time = cuda_profiler(triton_attn, q_s, k_s, v_s, scale_float, warmup_runs=1, timed_runs=10)
    
    return out



if __name__ == "__main__":
    args = arg_parser()
    gpu_name = get_gpu_name()
    print(f"Starting profiling for select attention")
    args.embed_dim = args.in_features
    args.results_dir = f"results/triton_kernels/{gpu_name}/"
    batch_size = args.batch_size
    seq_len = args.seq_len
    
    attn_topk_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    device = torch.device(args.device)
    num_heads = args.embed_dim // 128
    head_dim = args.embed_dim // num_heads
    q = torch.randn(args.batch_size, 1, num_heads, head_dim, dtype = torch.float16).to(device)
    k_cache = torch.randn(args.batch_size, args.seq_len, num_heads, head_dim, dtype = torch.float16).to(device)
    v_cache = torch.randn(args.batch_size, args.seq_len, num_heads, head_dim, dtype = torch.float16).to(device)
    k = torch.randn(args.batch_size, 1, num_heads, head_dim, dtype = torch.float16).to(device)
    v = torch.randn(args.batch_size, 1, num_heads, head_dim, dtype = torch.float16).to(device)
    print(f"KV cache built")
    
    # dense flash attention 
    cache_seqlens = args.seq_len - 1 
    causal = False
    
    attn_topk = 0.5
    print(f"Number of heads selected = {int(num_heads * attn_topk)}")
    # # naive select attention
    batch_head_index = generate_random_BH_index(args.batch_size, num_heads, int(num_heads * attn_topk), device = device)
    
    # triton select attention
    q_s = q.unsqueeze(2) #torch.randn(B, M, G, H, Kq, dtype=dtype, device=device)
    k_s = k_cache.unsqueeze(2) #torch.randn(B, Mk, G, H, Kkv, dtype=dtype, device=device)
    v_s = v_cache.unsqueeze(2) #torch.randn(B, Mk, G, H, Kkv, dtype=dtype, device=device)
    # Scale for attention
    scale_float = 1.0 / np.sqrt(128)
    
    # Dense triton flash attention 
    triton_attn_out, dense_triton_time = cuda_profiler(triton_attn, q_s, k_s, v_s, scale_float, warmup_runs=1, timed_runs=10)
    print(f'Dense triton attention time: {dense_triton_time} ms')


    results = []

    # triton select attention
    # naive_out, naive_time = cuda_profiler(triton_attention_with_selection, q, k_cache, v_cache, batch_head_index, scale_float)
    # print(f"Naive select attention time: {naive_time:.3f} ms")
    
    # sel_attn_out, sel_attn_time = cuda_profiler(select_attn, q_s, k_s, v_s, scale_float, batch_head_index, args.seq_len, warmup_runs=1, timed_runs=10)
    # print(f'Average time taken for triton selected attention: {sel_attn_time} ms')
    
    
    # Iterate over attn_topk_list
    for attn_topk in attn_topk_list:
        print(f"Running for attn_topk = {attn_topk}")
        num_selected_heads = int(num_heads * attn_topk)
        print(f"Number of heads selected = {num_selected_heads}")
        
        # Generate batch head index for selection
        batch_head_index = generate_random_BH_index(args.batch_size, num_heads, num_selected_heads, device=device)
        
        # Naive select attention
        naive_out, naive_time = cuda_profiler(triton_attention_with_selection, q, k_cache, v_cache, batch_head_index, scale_float)
        print(f"Naive select attention time: {naive_time:.3f} ms")
        
        # Triton selected attention
        sel_attn_out, sel_attn_time = cuda_profiler(select_attn, q_s, k_s, v_s, scale_float, batch_head_index, args.seq_len, warmup_runs=1, timed_runs=10)
        print(f'Average time taken for triton selected attention: {sel_attn_time} ms')
        
        # Append results
        results.append({
            "attn_topk": attn_topk,
            "dense_time": dense_triton_time,
            "naive_time": naive_time,
            "sel_attn_time": sel_attn_time
        })
    
    # Write results to CSV
    os.makedirs(args.results_dir, exist_ok=True)
    file_name = f"select_attention_profiling_{args.in_features}_{args.batch_size}_{args.seq_len}.csv"
    csv_file = os.path.join(args.results_dir, file_name)
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["attn_topk", "dense_time", "naive_time", "sel_attn_time"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {csv_file}")