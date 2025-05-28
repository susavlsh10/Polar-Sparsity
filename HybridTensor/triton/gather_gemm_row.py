import torch

import triton
import triton.language as tl
from .triton_utils import get_autotune_config, benchmark_fwd, torch_sel_gemm_row
from .activations import leaky_relu, relu

from HybridTensor.utils.utils import sparse_index, arg_parser

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'index_size'],
)

@triton.jit
def matmul_gather_kernel_row(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_vec_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K, index_size,
    # The stride variables
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Compute block indices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    if USE_BIAS:
        # Load the bias vector
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(index_size, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load indices for the current block
        index_vec = tl.load(index_vec_ptr + offs_k, mask=offs_k < index_size, other=0)

        # Compute pointers to A and B
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (index_vec[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < index_size), other=0.0)
        b = tl.load(b_ptrs, mask=(index_vec[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        # Accumulate partial results
        accumulator += tl.dot(a, b)
    
    if USE_BIAS:
        # Add the bias
        accumulator += bias[None, :]  # Broadcast bias over rows

    # Cast accumulator to desired type and store the result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def gather_matmul_row(a, b, index_vec, bias = None, activations="", out = None):
    # Check constraints.
    # a and b are expected to be in row major format.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, _ = a.shape
    K, N = b.shape
    index_size = index_vec.shape[0]
    # Allocates output.
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if bias is not None:
        assert bias.size(0) == N, "Incompatible bias dimensions"

    use_bias = bias is not None
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_gather_kernel_row[grid](
        a, b, out, index_vec, bias, #
        M, N, K, index_size, #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        out.stride(0), out.stride(1),  #
        ACTIVATION='',  #
        USE_BIAS = use_bias
        )
    
    return out

if __name__ == '__main__':
    args = arg_parser()
    torch.manual_seed(0)
    
    # modeling the ffn2 kernel
    A = torch.randn(args.batch_size, args.hidden_features, device='cuda', dtype=torch.float16)
    B = torch.randn(args.hidden_features, args.in_features, device='cuda', dtype=torch.float16)

    index_vec = sparse_index(args.index_size, args.hidden_features)[0]
    A_select = A[:, index_vec]
    
    if args.bias:
        print("Using bias")
        bias = torch.randn(args.in_features, device='cuda', dtype=torch.float16)
    else:
        print("Not using bias")
        bias = None
    
    print(f"args: {args}")
        
    out_b, cublass_time = benchmark_fwd(A, B, bias = None, function = torch.matmul, print_result = args.print_results)
    out_a_gather, gather_gemm_time = benchmark_fwd(A_select, B, bias = bias, index_vec=index_vec, function=gather_matmul_row, print_result = args.print_results)
    out_a_gather_torch,torch_sel_gemm_time  = benchmark_fwd(A_select, B, bias = bias, index_vec=index_vec, function=torch_sel_gemm_row, print_result = args.print_results)

    # C = torch.empty((args.batch_size, args.in_features), device=A.device, dtype=torch.float16)
    # out_a_gather_alloc, gather_gemm_time_alloc = benchmark_fwd(A_select, B, index_vec=index_vec, function=gather_matmul_row, print_result = args.print_results, out = C)


    speedup = cublass_time / gather_gemm_time
    
    # check results
    if args.check_results:
        print("Checking results")
        # print("Cublass output: ", out_b)
        # print("Kernel gather output: ", out_a_gather)
        # print("Torch gather output: ", out_a_gather_torch)
    
        # assert torch.allclose(out_b, out_a_gather, atol=1e-3), "Results do not match"
        if  torch.allclose(out_a_gather, out_a_gather_torch, atol=0.5): #, "Gathered output does not match torch.matmul output"
            print("Results match ✅")
        else:
            print("Results do not match ❌")
    print(f"Speedup: {speedup:.2f}")