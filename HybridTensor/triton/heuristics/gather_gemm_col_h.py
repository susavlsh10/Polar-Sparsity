# python -m HybridTensor.triton.heuristics.gather_gemm_col_h --batch_size 8 --in_features 4096 --hidden_features 16384 --index_size 512


import torch

import triton
import triton.language as tl
from HybridTensor.triton.triton_utils import get_autotune_config, benchmark_fwd, torch_sel_gemm_col
from HybridTensor.triton.activations import leaky_relu, relu

from HybridTensor.utils.utils import sparse_index, arg_parser
from HybridTensor.triton.heuristics.heuristics import HeuristicSelector

heuristic_dir = "configs/gemm"
selector = HeuristicSelector(heuristic_dir, type="col")

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs

@triton.jit
def matmul_gather_kernel_col(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        index_vec_ptr,  # Pointer to the index vector for selected columns in B
        bias_ptr,
        # Matrix dimensions
        M, N, K, gather_N,
        # Strides
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  # B is column-major
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr,  #
        USE_BIAS: tl.constexpr
):
    """Kernel for computing the matmul C = A x B_select, 
    where B_select contains columns of B selected using index_vec.
    A has shape (M, K) in row-major and B has shape (K, N) in column-major.
    index_vec is used to select the columns of B for the matmul.
    """
    # -----------------------------------------------------------
    # Map program ids to block of C
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(gather_N, BLOCK_SIZE_N)  # Adjusted for the gathered columns
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gather_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointer arithmetic for A (row-major, K contiguous)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Gather selected column indices from B using the index_vec
    gather_idx = tl.load(index_vec_ptr + offs_bn)  # Gather indices from the index_vec
    b_ptrs = b_ptr + (gather_idx[None, :] * stride_bk + offs_k[:, None] * stride_bn)
    
    if USE_BIAS:
        bias = tl.load(bias_ptr + gather_idx, mask=offs_bn < gather_N, other=0.0)

    # -----------------------------------------------------------
    # Initialize the accumulator for C in FP32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B with masking for out-of-bounds K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Perform dot product and accumulate
        accumulator = tl.dot(a, b, accumulator)
        # Advance pointers for next iteration over K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bn  # Move along K dimension

    if USE_BIAS:
        accumulator += bias[None, :]
        
    # Optional activation
    accumulator = relu(accumulator)
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    # elif ACTIVATION == "relu":
    #     accumulator = relu(accumulator)
    
    # Convert the accumulator back to FP16
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the result into matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < gather_N)
    tl.store(c_ptrs, c, mask=c_mask)
    
def gather_matmul_col(a, b, index_vec, bias = None, activations="", out=None):
    # Check constraints.
    # b is expected to be in column major format.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if bias is not None:
        assert bias.size(0) == b.shape[0], "Incompatible bias dimensions"
        
    use_bias = bias is not None

    M, K = a.shape
    N, K = b.shape
    gather_N = index_vec.shape[0]   # Number of columns to gather from B, total neuron activations 
    # Allocates output.
    if out is None:
        out = torch.empty((M, gather_N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    cfg = selector.get_config(M, K, N, gather_N)
    
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(gather_N, META['BLOCK_SIZE_N']), )
    grid = lambda META: (triton.cdiv(M, cfg['BLOCK_SIZE_M']) * triton.cdiv(gather_N, cfg['BLOCK_SIZE_N']), )
    
    matmul_gather_kernel_col[grid](
        a, b, out, index_vec, bias, #
        M, N, K, gather_N, #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        out.stride(0), out.stride(1),  #
        ACTIVATION='',  #
        USE_BIAS = use_bias,
        BLOCK_SIZE_M=cfg['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=cfg['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=cfg['BLOCK_SIZE_K'],
        GROUP_SIZE_M=cfg['GROUP_SIZE_M'],
        num_warps=cfg['num_warps'],
        num_ctas=cfg['num_ctas'],
        num_stages=cfg['num_stages']
        )
    
    return out

if __name__ == '__main__':
    args = arg_parser()
    A = torch.randn(args.batch_size, args.in_features, device='cuda', dtype=torch.float16)
    B = torch.randn(args.in_features, args.hidden_features, device='cuda', dtype=torch.float16)
    B_col_major = B.t().contiguous()

    
    # index_vec = sparse_index(args.index_size, args.hidden_features)[0]
    
    router_output = torch.rand([args.hidden_features], device='cuda', dtype=torch.float16)
    _, index_vec = router_output.topk(args.index_size, dim=0, sorted=False)
    
    if args.bias:
        print("Using bias")
        bias = torch.randn(args.hidden_features, device='cuda', dtype=torch.float16)
    else:
        print("No bias")
        bias = None
    print(f"args: {args}")
    print(f"Index size: {args.index_size}, Activated {args.index_size/(args.in_features * 4)*100}% neurons")
    
    # Without CUDA Graph
    out_b, cublass_time = benchmark_fwd(A, B, bias = None, function = torch.matmul, print_result = args.print_results)
    out_a_gather, gather_gemm_time = benchmark_fwd(A, B_col_major, bias = bias, index_vec=index_vec, function=gather_matmul_col, print_result = args.print_results)
    out_a_gather_torch,torch_sel_gemm_time  = benchmark_fwd(A, B, bias = bias, index_vec=index_vec, function=torch_sel_gemm_col, print_result = args.print_results)

    speedup = cublass_time / gather_gemm_time

    # check results
    if args.check_results:
        print("Checking results")
        print("Kernel gather output: ", out_a_gather)
        print("Torch gather output: ", out_a_gather_torch)
        # Check if the results are the same
        if  torch.allclose(out_a_gather, out_a_gather_torch, atol=0.5): #, "Gathered output does not match torch.matmul output"
            print("Results match ✅")
        else:
            print("Results do not match ❌")
            # max difference 
            print("Max difference: ", torch.max(torch.abs(out_a_gather - out_a_gather_torch)))
    
    print(f"Speedup: {speedup:.2f}")
    
    # ----------------------------
    # CUDA Graph capture testing for gather_matmul_col
    # ----------------------------
    print("Testing CUDA Graph capture...")
    
    # Warm-up run to compile the kernel
    out_cuda = gather_matmul_col(A, B_col_major, index_vec, bias=bias)
    torch.cuda.synchronize()
    
    # Allocate static buffers for CUDA Graph capture
    static_out = torch.empty_like(out_cuda)
    static_A = A.clone()
    static_B = B_col_major.clone()
    static_index_vec = index_vec.clone()
    if bias is not None:
        static_bias = bias.clone()
    else:
        static_bias = None
    
    # Warm up static buffers
    _ = gather_matmul_col(static_A, static_B, static_index_vec, bias=static_bias, out=static_out)
    torch.cuda.synchronize()
    
    # Capture on a non-default stream
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        g = torch.cuda.CUDAGraph()
        g.capture_begin()
        gather_matmul_col(static_A, static_B, static_index_vec, bias=static_bias, out=static_out)
        g.capture_end()
    torch.cuda.synchronize()
    
    # Replay the graph and compare with a regular run
    g.replay()
    torch.cuda.synchronize()
    cuda_graph_out = static_out.clone()
    
    regular_out = gather_matmul_col(A, B_col_major, index_vec, bias=bias)
    if torch.allclose(cuda_graph_out, regular_out, atol=1e-3):
        print("CUDA Graph output matches regular output")
    else:
        print("CUDA Graph output does not match regular output")
    
    # Test with updated inputs
    print("Testing CUDA Graph with updated inputs...")
    new_A = torch.randn_like(A)
    new_B = torch.randn_like(B)
    new_B_col_major = new_B.t().contiguous()
    if bias is not None:
        new_bias = torch.randn_like(bias)
    else:
        new_bias = None
    router_output_new = torch.rand([args.hidden_features], device='cuda', dtype=torch.float16)
    _, new_index_vec = router_output_new.topk(args.index_size, dim=0, sorted=False)
    
    static_A.copy_(new_A)
    static_B.copy_(new_B_col_major)
    static_index_vec.copy_(new_index_vec)
    if static_bias is not None:
        static_bias.copy_(new_bias)
    
    g.replay()
    torch.cuda.synchronize()
    cuda_graph_out_new = static_out.clone()
    
    regular_out_new = gather_matmul_col(new_A, new_B_col_major, new_index_vec, bias=new_bias)
    if torch.allclose(cuda_graph_out_new, regular_out_new, atol=1e-3):
        print("Updated CUDA Graph output matches regular output")
    else:
        print("Updated CUDA Graph output does not match regular output")
        
    # Execute the graph
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.iterations):
        g.replay()
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time = start.elapsed_time(end) / args.iterations
    print(f"Average time per genearation (CUDA GRAPH): {elapsed_time:.4f} ms")