# python run_sparse_attn.py --in_features 8192 --batch_size 16 --seq_len 1920 --attn_topk 0.5

import math
import torch
from HybridTensor.modules.SelectiveMHA import SMHA, _update_kv_cache
from HybridTensor.utils.utils import arg_parser, generate_random_BH_index
from HybridTensor.utils.profiling import cuda_profiler
from HybridTensor.utils.generation import InferenceParams

if __name__ == "__main__":
    args = arg_parser()
    
    max_seqlen = args.seq_len + 128
    max_batch_size = args.batch_size
    device = torch.device(f"cuda:{args.device}")
    
    # simulates SelectiveMHA inference generation stage
    inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=max_batch_size)
    nheads = args.in_features // 128
    softmax_scale = 1 / (128 ** 0.5)
    rotary_emb_dim = 0
    
    mha = SMHA(
        embed_dim=args.in_features,
        num_heads=nheads,
        num_heads_kv=None,
        causal=True,
        layer_idx=0,
        use_flash_attn=True,
        softmax_scale=softmax_scale,
        return_residual=False,
        rotary_emb_dim=rotary_emb_dim,
        device=device,
        dtype=torch.float16,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    with torch.no_grad():
        # prefill stage to generate kv cache for all batches
        og_x = torch.randn(args.batch_size, args.seq_len, args.in_features, device=device, dtype=torch.float16, requires_grad=False)
        
        # simulate kv cache, bug in flash_attn for larger batches
        kv = torch.randn(args.batch_size, args.seq_len, 2, nheads, 128,  device=device, dtype=torch.float16, requires_grad=False)
        _ = _update_kv_cache(kv, inference_params, 0)
        
        # increment the sequence length to move to the generation stage
        inference_params.seqlen_offset += args.seq_len
        
        input_x = torch.randn(args.batch_size, 1, args.in_features, device=device, dtype=torch.float16, requires_grad=False)
        selected_heads = math.ceil(nheads * args.head_density)
        
        # generate batch_head_idx for SelectiveMHA
        # batch_head_index = generate_BH_index(args.batch_size, nheads, selected_heads, device=device)
        batch_head_index = generate_random_BH_index(args.batch_size, nheads, selected_heads, device=device)
        
        # generatation stage Standard MHA, batch_head_idx=None uses dense attention
        out, standard_time_ms = cuda_profiler(mha, input_x, inference_params=inference_params, batch_head_idx=None)
        print(f"Standard MHA time: {standard_time_ms:.3f} ms")
        
        # generatation stage SelectiveMHA, pass batch_head_idx to use selective head attention
        out, select_time_ms = cuda_profiler(mha, input_x, inference_params=inference_params, batch_head_idx=batch_head_index)
        print(f"SelectMHA time: {select_time_ms:.3f} ms")
        
        speedup = standard_time_ms / select_time_ms
        print(f"Speedup: {speedup:.3f}")
        
