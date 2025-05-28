# python -m HybridTensor.modules.SelectiveMLP --batch_size 8 --index_size 512
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
import torch.distributed as dist

# import fused_dense_cuda  # from apex

import fused_dense_lib as fused_dense_cuda

from flash_attn.utils.distributed import reduce_scatter, all_reduce
from einops import rearrange

# from HybridTensor.modules.MLP import SelectiveMLPFunc
from HybridTensor.modules.references.fused_dense import ColumnParallelLinear, RowParallelLinear, fused_mlp_func
from HybridTensor.modules.references.MLP import SelectiveMLPTriton
from HybridTensor.utils.utils import arg_parser, sparse_index
from HybridTensor.utils.profiling import cuda_profiler

# compiles the kernels for the first time, takes time
from HybridTensor.triton.gather_gemm_col import gather_matmul_col
from HybridTensor.triton.gather_gemm_row import gather_matmul_row

# needs to be compiled before running
from HybridTensor.triton.heuristics.gather_gemm_col_h import gather_matmul_col as gather_matmul_col_h
from HybridTensor.triton.heuristics.gather_gemm_row_h import gather_matmul_row as gather_matmul_row_h

# from HybridTensor.triton.cg_safe.gather_gemm_col_cg import gather_matmul_col
# from HybridTensor.triton.cg_safe.gather_gemm_row_cg import gather_matmul_row


def SelectiveMLPFunc(x, fc1_w, fc2_w, index_vec, bias1 = None, bias2 = None, activation='relu', use_heuristic=True):
    if use_heuristic:
        out = gather_matmul_col_h(x, fc1_w, index_vec, bias = bias1, activations=activation)
        out = gather_matmul_row_h(out, fc2_w, index_vec, bias = bias2)
    else:
        out = gather_matmul_col(x, fc1_w, index_vec, bias = bias1, activations=activation)
        out = gather_matmul_row(out, fc2_w, index_vec, bias = bias2)
    return out


# cg safe version
# def SelectiveMLPFunc(x, fc1_w, fc2_w, index_vec, index_size, bias1 = None, bias2 = None, activation='relu', use_heuristic=True):
#     out = gather_matmul_col(x, fc1_w, index_vec, index_size, bias = bias1, activations=activation)
#     out = gather_matmul_row(out, fc2_w, index_vec, index_size, bias = bias2)
#     return out

class MLPRouter(nn.Module):
    def __init__(self, embed_dim, low_rank_dim, out_dim, act_th, device=None, dtype=None):
        """
        Initializes the MHARouter class.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            low_rank_dim (int): Dimensionality of the intermediate layer.
            out_dim (int): Number of neurons.
        """
        super(MLPRouter, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(embed_dim, low_rank_dim, bias=False, **factory_kwargs)
        self.fc2 = nn.Linear(low_rank_dim, out_dim, bias=False, **factory_kwargs)
        self.act_th = act_th
        self.num_neurons = out_dim
        self.largest = self.num_neurons + 1

    def forward(self, x):
        """
        Forward pass of the MHARouter.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def _select_neurons_topk(self, x, topk=None):
        neurons = self.forward(x)
        
        neurons_nonzero = torch.nn.ReLU()(neurons)
        _, index_vec = neurons_nonzero.sum(dim=0).topk(topk, dim=0, sorted=False)
        # index_vec, _ = index_vec.sort()
        return index_vec
    
    def _select_neurons(self, x, th=None):
        '''
        Threshold based selection of neurons, not CG safe 
        '''
        if th is None:
            th = self.act_th
        
        neurons = self.forward(x)
        activated = (neurons > th).sum(dim=0)
        index_vec = activated.nonzero().flatten()
        return index_vec
    
    def _select_neurons_cuda_safe(self, x, th=None):
        '''
        This function is used with threshold and is used for CG safe version of the code
        '''
        if th is None:
            th = self.act_th
        neurons = self.forward(x)
        activated = (neurons > th).sum(dim=0)
        
        indices = torch.arange(self.num_neurons, device=activated.device)
        selected = torch.where(activated > th, indices, torch.full_like(indices, self.largest))
        
        index_vec, _ = torch.sort(selected)
        index_size = ((index_vec < self.largest).sum()).to(torch.int32)
        
        return index_size, index_vec
    


class ParallelMLPRouter(nn.Module):
    """
    Parallel Sparse Predictor for MHA layer.
    """

    def __init__(
        self,
        embed_dim,
        low_rank_dim,
        out_dim,
        act_th,
        process_group,
        sequence_parallel=False,
        device=None,
        dtype=None,
    ):
        """
        Initializes the ParallelMHARouter class.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            low_rank_dim (int): Dimensionality of the intermediate layer.
            out_dim (int): Output dimensionality (typically number of neurons).
            process_group (torch.distributed.ProcessGroup): Process group for parallelism.
            sequence_parallel (bool, optional): Whether to use sequence parallelism. Defaults to False.
            device (torch.device, optional): Device to run the module on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module parameters. Defaults to None.
        """
        super(ParallelMLPRouter, self).__init__()
        assert process_group is not None, "ParallelMHARouter requires a process group."

        factory_kwargs = {"device": device, "dtype": dtype}
        self.process_group = process_group
        self.embed_dim = embed_dim
        self.act_th = act_th

        self.fc1 = nn.Linear(
            embed_dim, low_rank_dim, bias=False, **factory_kwargs
        )
        self.fc2 = ColumnParallelLinear(
            low_rank_dim,
            out_dim,
            process_group,
            bias=False,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        
    # def _select_neurons(self, neurons, th=None):
    #     if th is None:
    #         th = self.act_th
    #     activated = (neurons > th).sum(dim=0)
    #     index_vec = activated.nonzero().flatten()
    #     return index_vec

    def forward(self, x):
        """
        Forward pass of the ParallelMHARouter.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, out_dim).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _select_neurons(self, x, th=None):
        if th is None:
            th = self.act_th
        
        neurons = self.forward(x)
        activated = (neurons > th).sum(dim=0)
        index_vec = activated.nonzero().flatten()
        return index_vec
    
    def _select_neurons_topk(self, x, topk=None):
        neurons = self.forward(x)
        
        neurons_nonzero = torch.nn.ReLU()(neurons) #.squeeze(1)
        # print(f"neurons_nonzero shape: {neurons_nonzero.shape}")
        # print(f"Top k neurons: {topk}")
        _, index_vec = neurons_nonzero.sum(dim=0).topk(topk, dim=0, sorted=False)
        # index_vec, _ = index_vec.sort()
        return index_vec

class SelectiveMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation='relu',
        layer_idx=None,
        bias1=True,
        bias2=True,
        return_residual=False,
        checkpoint_lvl=0,
        use_heuristic=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.activation_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        # self.fc2_weight_t = self.fc2.weight.t().contiguous()
        self.fc2_weight_t = None
        self.use_heuristic = use_heuristic

    def _init_weights(self):
        # if weights are updated, we need to update the transpose
        self.fc2_weight_t = self.fc2.weight.t().contiguous()
        
    def forward(self, x, index_vec=None, index_size=None):
        
        if index_vec is not None:
            # sparse forward,
            
            # update on first run 
            if self.fc2_weight_t is None:
                self.fc2_weight_t = self.fc2.weight.t().contiguous()
                
                # Remove the original parameter to free memory.
                self.fc2.weight = None
                del self.fc2._parameters['weight']
            
            x = x.view(-1, x.size(-1))
            # x = x.squeeze(1)
            y = SelectiveMLPFunc(x = x, fc1_w = self.fc1.weight,
                                fc2_w = self.fc2_weight_t, index_vec = index_vec,
                                bias1 = self.fc1.bias, bias2 = self.fc2.bias,
                                activation=self.activation, use_heuristic=self.use_heuristic)
            
        else:
            # dense forward
            
            y = self.fc1(x)
            y = self.activation_fn(y)
            
            if self.fc2_weight_t is not None:
                y = torch.matmul(y, self.fc2_weight_t)
            else:
                y = self.fc2(y)
            
        return y if not self.return_residual else (y, x)

class ParallelSelectiveMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        activation="relu",
        layer_idx=None,
        process_group: ProcessGroup = None,
        bias1=True,
        bias2=True,
        return_residual=False,
        sequence_parallel=False,
        use_heuristic=True,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    ):
        """
        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ["gelu_approx", "relu"]
        assert process_group is not None
        # assert sp_kwargs != None, "sparse predictor parameters are not passed in."
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.activation = activation
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic
        self.fc1 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias=bias1, **factory_kwargs
        )
        self.fc2 = RowParallelLinear(
            hidden_features, out_features, process_group, bias=bias2, **factory_kwargs
        )
        self.layer_idx = layer_idx

        self.fc2_weight_t = self.register_buffer("fc2_weigth_t", None)
        self.return_residual = return_residual
        self.fc2_weight_t = None    
        self.use_heuristic = use_heuristic
        self.reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        
        # self._init_weights()
    
    def _init_weights(self):
        # ffn2 weights needs to be in row major format to select from rows
        self.fc2_weight_t = self.fc2.weight.t().contiguous()
    
    def forward(self, x, residual = None, index_vec = None):

        # do_token_generation = x.size(1) == 1
        # index_vec = None
        # with torch.cuda.stream(self.curr_stream):
        if index_vec is not None:
            # assert x.size(1) == 1
            if self.fc2_weight_t is None:
                self.fc2_weight_t = self.fc2.weight.t().contiguous()
            
            x = x.view(-1, x.size(-1))
            # x = rearrange(x, "b 1 d -> b d") # slightly more expensive to use rearrange
            
            out = SelectiveMLPFunc(x = x, fc1_w = self.fc1.weight,
                                fc2_w = self.fc2_weight_t, index_vec = index_vec,
                                bias1 = self.fc1.bias, bias2 = self.fc2.bias,
                                activation=self.activation, use_heuristic=self.use_heuristic)
            # out = rearrange(out, "b d -> b 1 d")
            # out = out.view(-1, 1, out.size(-1))
       
        else:   # normal mlp
            if self.heuristic == "auto":
                dtype = (
                    x.dtype
                    if not torch.is_autocast_enabled()
                    else torch.get_autocast_gpu_dtype()
                )
                if self.activation == "gelu_approx":
                    cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                    heuristic = (
                        0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
                    )
                else:
                    heuristic = 0
            else:
                heuristic = self.heuristic
            out = fused_mlp_func(
                x,
                self.fc1.weight,
                self.fc2.weight,
                self.fc1.bias,
                self.fc2.bias,
                activation=self.activation,
                save_pre_act=self.training,
                checkpoint_lvl=self.checkpoint_lvl,
                heuristic=heuristic,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
            )
        
        if self.process_group.size() > 1:
            # out = self.reduce_fn(out, self.process_group) # has some overhead,
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.process_group)
        
        return out if not self.return_residual else (out, x)
        # return out
    
    def sp_forward(self, x, residual = None, index_vec = None):
        if self.heuristic == "auto":
            dtype = (
                x.dtype
                if not torch.is_autocast_enabled()
                else torch.get_autocast_gpu_dtype()
            )
            if self.activation == "gelu_approx":
                cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                heuristic = (
                    0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
                )
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        curr_stream = torch.cuda.current_stream()
        do_token_generation = x.size(1) == 1
        # mlp_logit = None
        
        # with torch.cuda.stream(self.curr_stream):
        if index_vec != None:
            assert x.size(1) == 1

            if self.fc2_weight_t is None:
                self.fc2_weight_t = self.fc2.weight.t().contiguous()

            out = SelectiveMLPFunc(
                rearrange(x, "b 1 d -> b d"),
                self.fc1.weight,
                self.fc2_weight_t,
                index_vec,
                self.fc1.bias,
                self.fc2.bias,
                activation=self.activation,
            )
            out = rearrange(out, "b d -> b 1 d")
        else:
            out = fused_mlp_func(
                x,
                self.fc1.weight,
                self.fc2.weight,
                self.fc1.bias,
                self.fc2.bias,
                activation=self.activation,
                save_pre_act=self.training,
                checkpoint_lvl=self.checkpoint_lvl,
                heuristic=heuristic,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
            )
        
        
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        if self.sp_router:
            curr_stream.record_event(self.event_mlp)
        
        # handle = torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=self.process_group, async_op=True)
        out = reduce_fn(out, self.process_group)
        

        if self.sp_router:
            with torch.cuda.stream(self.sp_stream):
                self.sp_stream.wait_event(self.event_mlp)
                if do_token_generation:
                    mlp_logit = self.sp(rearrange(residual, "b 1 d -> b d"))
                self.sp_stream.record_event(self.event_mlp_sp)

            # check this again, we might not have to synchronize here, we can synchronize in the next layer
            curr_stream.wait_event(self.event_mlp_sp)   
        
        return out
    
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.activation = activation
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    args = arg_parser()
    
    bias = True if args.bias > 0 else False
    x = torch.randn(args.batch_size, args.in_features, device="cuda", dtype=torch.float16)
    index_vec, _ = sparse_index(args.index_size, args.in_features*4)

    '''
    selective_mlp = SelectiveMLPTriton(args.in_features, args.hidden_features, bias=bias, device="cuda", dtype=torch.float16, activation="relu")
    
    out, mlp_time = cuda_profiler(selective_mlp, x, index_vec)
    
    out_col, col_time = cuda_profiler(gather_matmul_col, x, selective_mlp.fc1_w, index_vec, activations=selective_mlp.activation)
    out_row, row_time = cuda_profiler(gather_matmul_row, out_col, selective_mlp.fc2_w, index_vec)
    sum_time = col_time + row_time
    
    print(f"Index size {args.index_size}, Activated {args.index_size/(args.in_features * 4)*100}% neurons")
    
    print(f"Gather Col Time: {col_time} ms")
    print(f"Gather Row Time: {row_time} ms")
    # print(f"Sum Time: {sum_time} ms")
    
    print(f"SelectiveMLP Time: {mlp_time} ms")
    '''
    
    in_features = args.in_features
    hidden_features = in_features * 4
    out_features = in_features
    device = torch.device("cuda")
    
    model = SelectiveMLP(
        in_features, hidden_features, out_features, device=device, dtype=torch.float16, activation="relu", use_heuristic=True
    ).to(device)
    
    router = MLPRouter(in_features, 1024, hidden_features, act_th = 0.5, device=device, dtype=torch.float16).to(device)

    # Warm-up GPU
    def warmup():
        for _ in range(10):
            _ = model(x, index_vec)
            _ = model(x, None)
            _ = router._select_neurons_topk(x, args.index_size)

    warmup()

    # Measure SelectiveMLPFunc speed
    _, router_time = cuda_profiler(router._select_neurons_topk, x, args.index_size)
    _, selective_time = cuda_profiler(model, x, index_vec)
    # Measure dense forward speed
    _, dense_time = cuda_profiler(model, x, None)

    print(f"Router time per run: {router_time:.6f} ms")
    print(f"SelectiveMLPFunc time per run: {selective_time:.6f} ms")
    print(f"Dense forward time per run: {dense_time:.6f} ms")
    print(f"Speedup: {dense_time / selective_time:.2f}x")
    router_selective_time = router_time + selective_time
    print(f"Router + SelectiveMLPFunc time per run: {router_selective_time:.6f} ms")
    print(f"Speedup: {dense_time / router_selective_time:.2f}x")
    ############################################
    # CUDA Graph capture tests for the MLP model
    ############################################
    print("\n=== CUDA Graph Tests ===")
    # --- Selective forward (sparse mode) ---
    print("Testing CUDA Graph for Selective forward (with index_vec)...")
    static_x = x.clone()
    static_index_vec = index_vec.clone()
    # Warm-up run to allocate memory
    static_out_sel = model(static_x, index_vec=static_index_vec)
    torch.cuda.synchronize()

    # Capture on a non-default stream
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        g_sel = torch.cuda.CUDAGraph()
        g_sel.capture_begin()
        static_out_sel = model(static_x, index_vec=static_index_vec)
        g_sel.capture_end()
    torch.cuda.synchronize()

    # Replay and check accuracy
    g_sel.replay()
    torch.cuda.synchronize()
    cuda_sel_out = static_out_sel.clone()
    regular_sel_out = model(x, index_vec=index_vec)
    if torch.allclose(cuda_sel_out, regular_sel_out, atol=1e-3):
        print("Selective forward CUDA Graph output matches regular output")
    else:
        print("Selective forward CUDA Graph output does NOT match regular output")

    def replay_sel():
        g_sel.replay()
    _, selective_time_cuda = cuda_profiler(replay_sel)
    print(f"Selective forward CUDA Graph time per run: {selective_time_cuda:.6f} ms")