import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from flash_attn.utils.distributed import get_dim_for_local_rank
from flash_attn.utils.distributed import all_reduce

try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear, fused_dense_func
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

from flash_attn.modules.mha import SelfAttention, FlashSelfAttention, LinearResidual, FlashCrossAttention, CrossAttention
from flash_attn.modules.mha import get_alibi_slopes #, _update_kv_cache
from flash_attn.utils.generation import InferenceParams

# from HybridTensor.modules.references.mha_dejavu import ParallelTracker # use this in the full implementation
# from HybridTensor.modules.references.mha_dejavu import ParallelMHASparseAttMlp
# from HybridTensor.triton.references.attention_proj_sparse import qkv_proj_sparse
# from HybridTensor.triton.select_attn import select_attn
# from HybridTensor.triton.select_attn_64b_kernel import select_attn
from HybridTensor.triton.attn_interface import flash_attn_with_kvcache_triton
from HybridTensor.triton.select_attn_v1 import select_attn
from HybridTensor.utils.utils import arg_parser, generate_BH_index, generate_random_BH_index
from HybridTensor.utils.profiling import cuda_profiler


class MHARouter(torch.nn.Module):
    def __init__(self, embed_dim, low_rank_dim = None, out_dim = None, top_k = 0.5, device = None, dtype = None):
        super(MHARouter, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.model_dim = embed_dim
        self.num_heads = out_dim
        self.topk = top_k
        
        self.linear1 = torch.nn.Linear(embed_dim, out_dim, bias = True, **factory_kwargs)
        
    def forward(self, x):
        out = self.linear1(x)
        return out
    
    def _select_heads(self, x, topk = None):
        if topk is None:
            topk = int(self.topk * self.num_heads)
        else:
            topk = int(self.num_heads * topk)
        head_scores = self.forward(x)
        _, selected_heads = torch.topk(head_scores, topk, dim=1)
        
        return selected_heads

class ParallelMHARouter(torch.nn.Module):
    def __init__(self, embed_dim, low_rank_dim, out_dim, top_k, process_group, sequence_parallel=False, device = None, dtype = None):
        super(ParallelMHARouter, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.model_dim = embed_dim
        self.num_heads = out_dim
        self.topk = top_k
        world_size = torch.distributed.get_world_size(process_group)
        self.local_heads = out_dim // world_size
        
        self.linear1 = ColumnParallelLinear(
            embed_dim,
            out_dim,
            process_group,
            bias=True,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        
    def forward(self, x):
        out = self.linear1(x)
        return out
    
    def _select_heads(self, x, topk = None):
        if topk is None:
            topk = int(self.topk * self.local_heads)
        else:
            topk = int(self.local_heads * topk)
        head_scores = self.forward(x)
        # head_scores = head_scores.squeeze(1)
        # print(f"Head Scores.shape: {head_scores.shape}")
        _, selected_heads = torch.topk(head_scores, topk, dim=1)
        # print(f"Selected Heads: {selected_heads}")
        return selected_heads

def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]

class SMHA(nn.Module):
    """Multi-head self-attention and cross-attention with Triton decode kernels + Selective Attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        use_triton=True,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        self.use_triton = use_triton
        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), device=device)
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        )
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        if not self.cross_attn:
            self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = wqkv_cls(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(
                    qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim
                )
            else:
                self.dwconv_q = nn.Conv1d(
                    embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
                )
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)

        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    
    def _update_kvcache_attention_triton(self, q, kv, inference_params, batch_head_idx=None):
        """
        The rotary embeddings have to be applied before calling this function. The KV cache is update here.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = self._update_kv_cache(kv, inference_params)
            
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
                        
            context = flash_attn_with_kvcache_triton(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                None, # kv[:, :, 0],
                None, #kv[:, :, 1],
                rotary_cos=None,
                rotary_sin=None,
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                rotary_interleaved= False, #self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
                alibi_slopes=alibi_slopes,
                batch_head_idx=batch_head_idx,
            )
            return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                alibi_slopes=alibi_slopes,
            )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        batch_head_idx=None,
        use_triton=True,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            batch_head_idx: (batch, num_heads). The index of the heads to be selected. Only applicable for Selective Head/Group Attention.
            use_triton: whether to use triton kernels for attention in decode. If False, use the original flash attention implementation.
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv
        # use_triton = self.use_triton if use_triton is None else use_triton
        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                # prefill stage
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    if use_triton:
                        # print("Using the (prefill) triton flash attention implementation")
                        context = self._update_kvcache_attention_triton(
                            qkv[:, :, 0], qkv[:, :, 1:], inference_params, batch_head_idx
                        )
                    else:
                        # print("Using the (prefill) original flash attention implementation")
                        context = self._update_kvcache_attention(
                            qkv[:, :, 0], qkv[:, :, 1:], inference_params
                        )
            else:
                # decode stage  
                # print("Using triton kernels for attention")
                if use_triton:
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(
                            qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                        )
                    context = self._update_kvcache_attention_triton(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params, batch_head_idx
                    )
                else:
                    # print("Using the original flash attention implementation")
                    context = self._apply_rotary_update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
                
        else:   # cross-attention or MQA/GQA
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                # prefill
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    if use_triton:
                        context = self._update_kvcache_attention_triton(
                            q, kv, inference_params, batch_head_idx
                        )
                    else:
                        context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                # decode 
                # print("Using triton kernels for attention")
                if use_triton:
                    if self.rotary_emb_dim > 0:
                        q, kv = self.rotary_emb(
                            q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                        )
                    context = self._update_kvcache_attention_triton(
                        q, kv, inference_params, batch_head_idx
                    )
                else:
                    # print("Using the original gqa flash attention implementation")
                    context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        # print(f"Context.shape: {context.shape}")
        # print(f"Context: {context}")
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

class ParallelSMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.process_group = process_group
        self.world_size = process_group.size()
        self.local_rank = torch.distributed.get_rank(process_group)

        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"

        self.num_heads_per_rank = get_dim_for_local_rank(
            self.num_heads, self.world_size, self.local_rank
        )
        self.num_heads_kv_per_rank = get_dim_for_local_rank(
            self.num_heads_kv, self.world_size, self.local_rank
        )
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            num_heads_local = math.ceil(self.num_heads / self.world_size)
            alibi_slopes = torch.tensor(
                get_alibi_slopes(num_heads)[
                    self.local_rank * num_heads_local : (self.local_rank + 1) * num_heads_local
                ],
                device=device,
            )
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            qkv_dim,
            process_group,
            bias=qkv_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim * (self.num_heads // self.num_heads_kv + 2),
            **factory_kwargs,
        )
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            process_group,
            bias=out_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim,
            **factory_kwargs,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv_per_rank,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if inference_params.seqlen_offset == 0 or not self.use_flash_attn:
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            context = flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                alibi_slopes=alibi_slopes,
            )
            return context

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is not None:
            qkv = rearrange(qkv, "(b s) ... -> b s ...", s=seqlen)
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        if self.num_heads_kv == self.num_heads:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
            else:
                context = self._apply_rotary_update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:   # GQA/MQA
            q = rearrange(
                qkv[..., : self.num_heads_per_rank * self.head_dim],
                "... (h d) -> ... h d",
                d=self.head_dim,
            )
            kv = rearrange(
                qkv[..., self.num_heads_per_rank * self.head_dim :],
                "... (two hkv d) -> ... two hkv d",
                two=2,
                d=self.head_dim,
            )
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "b s h d -> b s (h d)")
        if seqlen is not None:
            context = rearrange(context, "b s d -> (b s) d")
        out = self.out_proj(context)
        return out

class SelectMHA(nn.Module):
    """Multi-head, Group-query self-attention using select attention"""
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=True,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = True # use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), device=device)
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        )
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        
        self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)

        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.softmax_scale = softmax_scale
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """Update kv cache in inference_params."""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)
    
    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        batch_head_idx=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
            batch_head_idx: Tensor of indices specifying which batch and head indices to select.
                Shape: (batch_size, top_k)
            inference_params: for generation.
        """
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]

        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            # Self-attention, no MQA/GQA
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(
                    qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
                
            if inference_params is None or inference_params.seqlen_offset == 0:
                # Inference stage without inference_params
                if inference_params is not None:
                    # Update kv cache during prefill
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)

                context = self.inner_attn(qkv, **kwargs)
            
            else:
                # Generation stage
                if batch_head_idx is None:
                    # Apply select attention without kv cache update
                    context = self._update_kvcache_attention(q = qkv[:, :, 0], kv = qkv[:, :, 1:], inference_params = inference_params)
                else:
                    # Apply select attention with kv cache update    
                    context = self._update_kvcache_select_attn(q = qkv[:, :, 0], kv = qkv[:, :, 1:], inference_params = inference_params, batch_head_idx = batch_head_idx)

        else:
            raise NotImplementedError("SelectMHA currently supports only self-attention without MQA/GQA.")
        
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

    def _update_kvcache_select_attn(self, q, kv, inference_params, batch_head_idx):
        """
        Apply select attention during generation stage.

        q: (batch_size, seqlen=1, n_heads, head_dim)
        kv: (batch_size, seqlen=1, 2, n_heads, head_dim)
        batch_head_idx: Tensor of indices specifying which batch and head indices to select.
            Shape:  (batch_size, top_k)
        
        # currently only supports batches with same seqlen
        # different seqlen requires a simple update in the select_attn kernel to load the seqlen, future work
        """
        # check batch_head_idx shape
        # assert batch_head_idx.shape[0] == 2, "batch_head_idx must have shape (N_selected, 2)"
        # check batch_head_idx is not None
        assert batch_head_idx is not None, "batch_head_idx must not be None"
        
        # update kv cache
        kv_cache = self._update_kv_cache(kv, inference_params)
        # inference_params.seqlen_offset += 1 # if seqlen_offset is int
        
        batch = q.shape[0]
        # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        
        # make sure seqlen_offset accounts for the current token
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset + 1 # +1 for the current token
        )
        
        # need to reshape or view keys and value with shape (batch_size, seqlen, 1, n_heads, head_dim)
        q = q.unsqueeze(2)
        k_cache = kv_cache[:, :, 0].unsqueeze(2)
        v_cache = kv_cache[:, :, 1].unsqueeze(2)
        
        # Call select_attn
        context = select_attn(
            q, 
            k_cache,
            v_cache,
            self.softmax_scale,
            batch_head_idx,
            cache_seqlens)
        
        # context: (batch_size, seqlen_q=1, G=1, H, head_dim)
        # context = context.squeeze(2)  # Remove G dimension
        batch_size = batch_head_idx.shape[0]
        context = context.view(batch_size, 1, self.num_heads, self.head_dim)
        
        return context
    
    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            # alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            alibi_slopes = None
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_attn.softmax_scale,
                causal=self.inner_attn.causal,
                alibi_slopes=alibi_slopes,
            )
    
    # dummy function for testing
    def _select_attn(self, q, kv, inference_params, batch_head_idx):
        """
        Apply select attention during generation stage.

        q: (batch_size, seqlen=1, n_heads, head_dim)
        kv: (batch_size, seqlen=1, 2, n_heads, head_dim)
        batch_head_idx: Tensor of indices specifying which batch and head indices to select.
            Shape:  (N_selected, 2)
        
        # currently only supports batches with same seqlen
        # different seqlen requires a simple update in the select_attn kernel to load the seqlen, future work
        """
        # check batch_head_idx shape
        assert batch_head_idx.shape[1] == 2, "batch_head_idx must have shape (N_selected, 2)"
        
        # update kv cache
        # kv_cache = self._update_kv_cache(kv, inference_params)
        # inference_params.seqlen_offset += 1 # if seqlen_offset is int
        
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        
        # make sure seqlen_offset accounts for the current token
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset # +1 for the current token
        )
        
        # need to reshape or view keys and value with shape (batch_size, seqlen, 1, n_heads, head_dim)
        q = q.unsqueeze(2)
        k_cache = kv_cache[:, :, 0].unsqueeze(2)
        v_cache = kv_cache[:, :, 1].unsqueeze(2)
        
        # Call select_attn
        context = select_attn(
            q, 
            k_cache,
            v_cache,
            self.softmax_scale,
            batch_head_idx,
            cache_seqlens)
        
        # context: (batch_size, seqlen_q=1, G=1, H, head_dim)
        context = context.squeeze(2)  # Remove G dimension
        
        return context


# SelectiveGQA: Future work

class ParallelSelectMHA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=True,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=True,
        use_flash_attn=True,
        return_residual=False,
        checkpointing=False,
        sequence_parallel=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        self.process_group = process_group
        self.world_size = process_group.size()
        self.local_rank = torch.distributed.get_rank(process_group)
        
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"

        self.num_heads_per_rank = get_dim_for_local_rank(
            self.num_heads, self.world_size, self.local_rank
        )
        self.num_heads_kv_per_rank = get_dim_for_local_rank(
            self.num_heads_kv, self.world_size, self.local_rank
        )
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            num_heads_local = math.ceil(self.num_heads / self.world_size)
            alibi_slopes = torch.tensor(
                get_alibi_slopes(num_heads)[
                    self.local_rank * num_heads_local : (self.local_rank + 1) * num_heads_local
                ],
                device=device,
            )
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            qkv_dim,
            process_group,
            bias=qkv_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim * (self.num_heads // self.num_heads_kv + 2),
            **factory_kwargs,
        )

        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )

        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.softmax_scale = softmax_scale
        
        # replace this with no reduce
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            process_group,
            bias=out_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim,
            **factory_kwargs,
        )
        
        self.mha_router = None
        self.mlp_router = None
        # We'll use an extra stream for concurrency
        self.current_stream = None
        self.sparse_stream = torch.cuda.Stream(device="cuda", priority=0)
        self.main_stream = torch.cuda.Stream(device="cuda", priority=-5)
        self.mha_router_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.mlp_router_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.main_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        # self.local_head_idx = generate_random_BH_index(1, self.num_heads_per_rank,self.num_heads_per_rank)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv_per_rank,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """Update kv cache in inference_params."""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)
    
    def forward(
        self,
        x,
        seqlen=None,
        inference_params=None,
        batch_head_idx=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
            batch_head_idx: Tensor of indices specifying which batch and head indices to select.
                Shape: (N_selected,)
            inference_params: for generation.
        """
        
        router_inputs = x.squeeze(1)
        self.current_stream = torch.cuda.current_stream()
        self.main_stream.wait_stream(self.current_stream )
        self.sparse_stream.wait_stream(self.current_stream )
        
        is_decode = inference_params is not None and inference_params.seqlen_offset > 0
        
        # if self.mha_router and is_decode:
        #     with torch.cuda.stream(self.sparse_stream):
        #         batch_head_idx = self.mha_router._select_heads(router_inputs)
        #         self.sparse_stream.record_event(self.mha_router_event)
        
        
        qkv = self.Wqkv(x)
        if seqlen is not None:
            qkv = rearrange(qkv, "(b s) ... -> b s ...", s=seqlen)
            
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]

        if self.num_heads_kv == self.num_heads:
            # Self-attention, no MQA/GQA
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
            
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(
                    qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None or inference_params.seqlen_offset == 0:
                # Inference stage without inference_params, prefill stage
                if inference_params is not None:
                    # Update kv cache during prefill
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)

                context = self.inner_attn(qkv, **kwargs)
            else:
                # Generation stage
                
                # apply rotary embeddings
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                
                # Apply select attention with kv cache update    
                context = self._update_kvcache_select_attn(qkv[:, :, 0], qkv[:, :, 1:], inference_params, batch_head_idx)
        else:   # cross-attention, MQA/GQA
            raise NotImplementedError("SelectMHA currently supports only self-attention without MQA/GQA.")

        context = rearrange(context, "b s h d -> b s (h d)")
        if seqlen is not None:
            context = rearrange(context, "b s d -> (b s) d")
        # out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        # out = self.out_proj(context)
        
        out = fused_dense_func(context, self.out_proj.weight, self.out_proj.bias)
        
        # if is_decode:
        #     if self.mlp_router:
        #         with torch.cuda.stream(self.sparse_stream):
        #             index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
        #             self.sparse_stream.record_event(self.mlp_router_event)
        
        #     with torch.cuda.stream(self.main_stream):
        #         out = all_reduce(out, self.process_group)
        #         self.main_stream.record_event(self.main_event)
        
        #     self.current_stream.wait_event(self.mlp_router_event)
        #     self.current_stream.wait_event(self.main_event)
            
        #     # index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
        #     # out = all_reduce(out, self.process_group)
            
        #     return out, index_vec
        # else:
        #     out = all_reduce(out, self.process_group)
        out = all_reduce(out, self.process_group)
        return out if not self.return_residual else (out, x)
        # return out
    
    def _update_kvcache_select_attn(self, q, kv, inference_params, batch_head_idx = None):
        """
        Apply select attention during generation stage.

        q: (batch_size, seqlen=1, n_heads, head_dim)
        kv: (batch_size, seqlen=1, 2, n_heads, head_dim)
        batch_head_idx: Tensor of indices specifying which batch and head indices to select.
            Shape:  (batch_size, top_k)
        """
        # check batch_head_idx shape
        # assert batch_head_idx.shape[1] == 2, "batch_head_idx must have shape (N_selected, 2)"
        
        # if batch_head_idx is None:
        #     batch_head_idx = self.local_head_idx
        #     print("Using local_head_idx, router not used.")
        # batch_head_idx = self.local_head_idx
        # print("Using local_head_idx, router not used.")
        
        # update kv cache
        kv_cache = self._update_kv_cache(kv, inference_params)
        
        batch = q.shape[0]
        # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset + 1 # +1 for the current token
        )
        # need to reshape or view keys and value with shape (batch_size, seqlen, 1, n_heads, head_dim)
        q = q.unsqueeze(2)
        k_cache = kv_cache[:, :, 0].unsqueeze(2)
        v_cache = kv_cache[:, :, 1].unsqueeze(2)
        
        self.current_stream.wait_event(self.mha_router_event)
        
        assert batch_head_idx is not None, "batch_head_idx must not be None"
        # Call select_attn
        context = select_attn(
            q, 
            k_cache,
            v_cache,
            self.softmax_scale,
            batch_head_idx,
            cache_seqlens
        )
        
        # context: (batch_size, seqlen_q=1, G=1, H, head_dim)
        # context = context.squeeze(2)  # Remove G dimension
        context = context.view(batch, 1, self.num_heads_kv_per_rank, self.head_dim)
        return context

''' 
PYTHONWARNINGS="ignore" python -m HybridTensor.modules.SelectiveMHA --batch_size 8 --in_features 8192 --seq_len 512 --head_density 0.25
'''


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
    
    # build SelectiveMHA
    select_mha = SelectMHA(
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
    
    standard_mha = SMHA(
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
        
        # out, time_ms = cuda_profiler(select_mha, og_x, inference_params=inference_params)
        # print(f"MHA Prefill time: {time_ms:.3f} ms")
        # out = select_mha(og_x, inference_params=inference_params)
        
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
        # generatation stage Standard MHA
        out, standard_time_ms = cuda_profiler(standard_mha, input_x, inference_params=inference_params)
        print(f"Standard MHA time: {standard_time_ms:.3f} ms")
        
        # generatation stage SelectiveMHA
        out, select_time_ms = cuda_profiler(select_mha, input_x, inference_params=inference_params, batch_head_idx=batch_head_index)
        print(f"SelectMHA time: {select_time_ms:.3f} ms")
        
        speedup = standard_time_ms / select_time_ms
        print(f"Speedup: {speedup:.3f}")
        
