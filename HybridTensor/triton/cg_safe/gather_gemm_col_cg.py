import torch
import triton
import triton.language as tl
from HybridTensor.triton.triton_utils import get_autotune_config, benchmark_fwd, torch_sel_gemm_col
from HybridTensor.triton.activations import leaky_relu, relu
from HybridTensor.utils.utils import sparse_index, arg_parser
from HybridTensor.utils.profiling import cuda_profiler

@triton.autotune(
    configs=get_autotune_config(),
    # key=['M', 'gather_N', 'K'],
    key=['M', 'K'],
)
@triton.jit
def matmul_gather_kernel_col(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        index_vec_ptr,  # Pointer to the index vector for selected columns in B
        bias_ptr,
        # Matrix dimensions
        M, N, K, index_size,
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
    # in_size = tl.load(index_size)
    in_size = index_size
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(in_size, BLOCK_SIZE_N)  # Adjusted for the gathered columns
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Exit early if this block's column start exceeds active columns.
    if pid_n * BLOCK_SIZE_N >= in_size:
        return

    # -----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % in_size
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointer arithmetic for A (row-major, K contiguous)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Gather selected column indices from B using the index_vec
    gather_idx = tl.load(index_vec_ptr + offs_bn)  # Gather indices from the index_vec
    b_ptrs = b_ptr + (gather_idx[None, :] * stride_bk + offs_k[:, None] * stride_bn)
    
    if USE_BIAS:
        bias = tl.load(bias_ptr + gather_idx, mask=offs_bn < in_size, other=0.0)

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
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < in_size)
    tl.store(c_ptrs, c, mask=c_mask)

def gather_matmul_col(a, b, index_vec, index_size, bias=None, activations="", out=None):
    # a must be contiguous; b is expected in column-major format.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if bias is not None:
        assert bias.size(0) == b.shape[0], "Incompatible bias dimensions"
    use_bias = bias is not None

    M, K = a.shape
    N, K = b.shape
    # Static gather_N equals the length of index_vec (total number of columns)
    # gather_N = index_vec.numel()
    

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Use a static grid based on the total number of columns.
    # grid = lambda META: (tl.cdiv(M, META['BLOCK_SIZE_M']) * tl.cdiv(gather_N, META['BLOCK_SIZE_N']), )
    # Compute grid dimensions on the host.
    grid = lambda META: (
        ((M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M']) *
        ((N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N']),
    )
    
    matmul_gather_kernel_col[grid](
        a, b, out, index_vec, bias,
        M, N, K, index_size,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        ACTIVATION=activations,
        USE_BIAS=use_bias
    )
    return out

def torch_sel_gemm_col(a, b, index_vec, index_size, bias=None, out=None):
    """
    Computes the selective GEMM in torch to match the Triton kernel.
    Only the first `index_size` columns (as given by index_vec) are computed;
    the remaining columns in the output (of shape determined by index_vec) are zeros.
    Assumes ReLU activation.
    """
    # Total static number of columns (gather_N)
    gather_N = index_vec.numel()
    M = a.shape[0]
    if out is None:
        out = torch.zeros((M, gather_N), device=a.device, dtype=a.dtype)
    
    # Compute only for the active indices (first `index_size` elements of index_vec)
    selected_indices = index_vec[:index_size]
    active_result = torch.matmul(a, b[:, selected_indices])
    if bias is not None:
        active_result = active_result + bias[selected_indices]
    
    active_result = torch.nn.functional.relu(active_result)
    
    # Place the computed results in the first index_size columns; the rest remain zero.
    out[:, :index_size] = active_result
    return out


if __name__ == '__main__':
    args = arg_parser()
    torch.manual_seed(0)
    A = torch.randn(args.batch_size, args.in_features, device='cuda', dtype=torch.float16)
    B = torch.randn(args.in_features, args.hidden_features, device='cuda', dtype=torch.float16)
    # Ensure B is column-major.
    B_col_major = B.t().contiguous()
    
    # Create a static index vector (length equals total columns in B).
    index_vec = torch.empty((args.hidden_features,), device='cuda', dtype=torch.int32)
    active_indices = sparse_index(args.index_size, args.hidden_features)[0]
    index_vec[:args.index_size] = active_indices
    if args.index_size < args.hidden_features:
        index_vec[args.index_size:] = 0  # Fill the rest with dummy values.
    
    if args.bias:
        print("Using bias")
        bias = torch.randn(args.hidden_features, device='cuda', dtype=torch.float16)
    else:
        print("No bias")
        bias = None

    print(f"args: {args}")
    print(f"Index size: {args.index_size}, Activated {args.index_size/(args.in_features * 4)*100}% neurons")
    index_size = torch.tensor(args.index_size, device='cuda', dtype=torch.int32)
    
    # Benchmark standard matmul.
    out_b, cublass_time = cuda_profiler(torch.matmul, A, B)
    print(f"Cublass time: {cublass_time:.2f} ms")
    # Benchmark the CUDA Graph safe selective GEMM.
    out_a_gather, gather_gemm_time = cuda_profiler(gather_matmul_col, A, B_col_major, index_vec, index_size, bias=bias)
    print(f"Gather kernel time: {gather_gemm_time:.2f} ms")
    # Benchmark the torch-based selective GEMM.
    out_a_gather_torch, torch_sel_gemm_time = cuda_profiler(torch_sel_gemm_col, A, B, index_vec, index_size, bias=bias)
    print(f"Torch gather time: {torch_sel_gemm_time:.2f} ms")

    speedup = cublass_time / gather_gemm_time

    if args.check_results:
        print("Checking results")
        print("Kernel gather output: ", out_a_gather)
        print("Torch gather output: ", out_a_gather_torch)
        if torch.allclose(out_a_gather, out_a_gather_torch, atol=0.5):
            print("Results match ✅")
        else:
            print("Results do not match ❌")
            print("Max difference: ", torch.max(torch.abs(out_a_gather - out_a_gather_torch)))
    
    print(f"Speedup: {speedup:.2f}")
