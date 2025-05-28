# python run_sparse_transformer_block.py --in_features 8192 --batch_size 32 --seq_len 1920 --index_size 8192 --attn_topk 0.5

import torch
import time
from HybridTensor.utils.utils import arg_parser, generate_random_BH_index
from HybridTensor.utils.profiling import cuda_profiler
from HybridTensor.utils.generation import InferenceParams
from HybridTensor.utils.utils import sparse_index
from HybridTensor.utils.utils import _get_device
from HybridTensor.models.create_sparse_model import create_block

class Config:
    def __init__(self, in_features=8192):
        self.hidden_size = in_features
        self.num_attention_heads = in_features // 128
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale_attn_weights = True
        self.mup_scale_qk_dot_by_d = False
        self.mup_attn_multiplier = 1.0
        self.scale_attn_by_inverse_layer_idx = False
        self.attn_dwconv = False
        self.qkv_proj_bias = True
        self.out_proj_bias = True
        self.rotary_emb_fraction = 0.0
        self.rotary_emb_base = 10000.0
        self.rotary_emb_scale_base = None
        self.rotary_emb_interleaved = False
        self.use_alibi = False
        self.window_size = (-1, -1)
        self.use_flash_attn = True
        self.fused_bias_fc = True
        self.mlp_sparse = True
        self.att_sparse = True
        self.attn_pdrop = 0.1
        self.n_inner = None  # Can be overridden
        self.activation_function = "relu"
        self.fused_mlp = True
        self.mlp_checkpoint_lvl = 0
        self.sequence_parallel = False
        self.layer_norm_epsilon = 1e-5
        self.residual_in_fp32 = False
        self.fused_dropout_add_ln = True
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.prenorm = True
        self.parallel_block = False

class SparseConfig:
    def __init__(self):
        self.mlp_low_rank_dim =  1024  
        self.attn_low_rank_dim = 128    # not used
        self.attn_topk = 0.5
        
if __name__ =="__main__":
    # Instantiate sample configs
    args = arg_parser()
    
    config = Config()
    sp_config = SparseConfig()
    sp_config.attn_topk = args.attn_topk
    
    config.hidden_size = args.in_features
    config.num_attention_heads = args.in_features // 128
    config.use_heuristic = False    # use pre-compiled heuristic or complie new one during runtime
    
    # Example device and dtype
    device = _get_device(args.device)
    dtype = torch.float16

    # Test create_block
    sparse_block = create_block(config, sp_config, layer_idx=0, process_group=None, device=device, dtype=dtype)
    sparse_block.eval()
    sparse_block.mlp_topk = args.index_size
    sparse_block.mlp.use_heuristic = False
    
    regular_config = config
    regular_config.att_sparse = False
    regular_config.mlp_sparse = False
    regular_block = create_block(regular_config, None, layer_idx=0, process_group=None, device=device, dtype=dtype)
    regular_block.eval()
    
    # inference simulation with select block
    max_seqlen = args.seq_len + 128
    max_batch_size = args.batch_size
    in_features = args.in_features
    head_dim = 128
    batch_size = args.batch_size
    seq_len = args.seq_len
    index_size = args.index_size
    
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
        # output_sparse = sparse_block(original_seq, mixer_kwargs=mixer_kwargs)
        # output_regular = regular_block(original_seq, mixer_kwargs=mixer_kwargs)
        
        # simulate the kv cache
        kv = torch.rand(batch_size, seq_len, 2, heads, head_dim, device='cuda', dtype=torch.float16)
        
        # need to update inference_params to reflect the new sequence length
        sparse_block.mixer._update_kv_cache(kv, inference_params)
        regular_block.mixer._update_kv_cache(kv, inference_params)
        mixer_kwargs["inference_params"].seqlen_offset = seq_len

        # Decode stage  
        input_x = torch.randn(batch_size, 1, in_features, device='cuda', dtype=torch.float16)
        
        out_decode_sparse = sparse_block(input_x, mixer_kwargs=mixer_kwargs)
        
        mixer_kwargs["inference_params"].seqlen_offset = seq_len
        
        out_decode_regular = regular_block(input_x, mixer_kwargs=mixer_kwargs)
        
        # mesure decode stage time in ms
        print("Without CUDA Graphs")
        out_decode_regular, regular_time = cuda_profiler(regular_block, input_x, mixer_kwargs=mixer_kwargs, warmup_runs=1, timed_runs=2)
        print(f"Regular time: {regular_time} ms")
        
        out_decode_sparse, sparse_time = cuda_profiler(sparse_block, input_x, mixer_kwargs=mixer_kwargs, warmup_runs=1, timed_runs=2)
        print(f"Sparse time: {sparse_time} ms")
        
        speedup = regular_time / sparse_time
        print(f"Speedup: {speedup}")
        
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

        print()
        print("With CUDA Graphs")
        print(f"Regular block time (CUDA Graphs): {regular_graph_time} ms")
        print(f"Sparse block time (CUDA Graphs): {sparse_graph_time} ms")
        print(f"Speedup (CUDA Graphs): {regular_graph_time/sparse_graph_time}")
        
        # Compare Outputs from Eager and CUDA Graph Versions
        if args.check_results:
            if isinstance(out_decode_regular, tuple):
                out_decode_regular = out_decode_regular[0]
            regular_match = torch.allclose(out_decode_regular, output_regular_static, rtol=1e-3, atol=1e-5)
            reg_diff = (out_decode_regular - output_regular_static).abs().max()
            # print both the outputs results
            # print(f"out_decode_regular: {out_decode_regular}")
            # print(f"output_regular_static: {output_regular_static}")
            
            print("\nComparison for Regular Block:")
            print(f"Outputs match: {regular_match}")
            print(f"Max difference: {reg_diff}")

            if isinstance(out_decode_sparse, tuple):
                out_decode_sparse = out_decode_sparse[0]
            sparse_match = torch.allclose(out_decode_sparse, output_sparse_static, rtol=1e-3, atol=1e-5)
            spa_diff = (out_decode_sparse - output_sparse_static).abs().max()
            print("\nComparison for Sparse Block:")
            print(f"Outputs match: {sparse_match}")
            print(f"Max difference: {spa_diff}")


