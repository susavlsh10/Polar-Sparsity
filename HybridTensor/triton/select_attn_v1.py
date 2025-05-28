from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple
import pytest
import torch
import sys

import triton
import triton.language as tl

def _strides(x: torch.Tensor, *stride_names: str):
    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


@triton.jit
def load_dequantize_k_v_group(
    K_block_ptr,
    V_block_ptr,
    K_scale_shift_block_ptr,
    V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    PACKED_D_PER_GROUP: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
):
    #Load K/V for a given block. In case of int4-quantized K/V, 
    # dequantize them after loading. If quantization is group-wise, 
    # use group_id to advance the pointers to the current group.

    # Advance to the current quantization group
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))

    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

    if PACKED_PER_VAL > 1:
        # K/V are quantized, load quantization coefficients and dequantize
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))

        k_scale_shift = tl.load(
            K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ()
        )
        v_scale_shift = tl.load(
            V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ()
        )

        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL).to(dtype)
        k_t = dequantize(
            tl.trans(k),
            tl.trans(k_scale),
            tl.trans(k_shift),
            PACKED_PER_VAL,
        ).to(dtype)
        k = tl.trans(k_t)
    return k, v

@triton.jit
def cast_uint32_to_half2(scale_shift):
    # Extract two float16 packed into one int32
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift

@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    # PACKED_PER_VAL is the number of values packed into 
    # each element x_. For example, for int4 quantization 
    #and x_ of type int32, PACKED_PER_VAL is 8.

    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = (
        x_[:, None, :] >> offsets[None, :, None]
    )  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

    quant_offset = tl.view(
        quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
    )
    # Trick - instead of converting int4 to float16 we view it as float16
    # and then multiply by 32768 * 512 == 2**24
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant

def get_split_k(B: int, G: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k


@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Seq_len,
    batch_head_index, # [B, top_k]
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    Z,
    N_CTX_Q,
    N_CTX_K,
    BLOCK_N_PER_SPLIT,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_SEQ_LEN: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr = 1,
    N_GROUPS: tl.constexpr = 1,
    top_k: tl.constexpr = 1,
):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    """
    tl.static_assert(
        (PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32))
        or (PACKED_PER_VAL == 8 and tl.constexpr(K.dtype.element_ty == tl.int32)),
        f"Only 4-bit quantization is supported, K/V should have dtype int32 in "
        f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
    )
    tl.static_assert(
        (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )

    QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1
    PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS

    # experimental
    # Cast strides and indexing variables to tl.int64 to prevent integer overflows

    start_m = tl.program_id(0).to(tl.int32)
    off_zhg = tl.program_id(1).to(tl.int64)
    splitk_idx = tl.program_id(2).to(tl.int64)
    
    off_g = off_zhg % G         # group, when G == 1, off_g = 0
    off_btk = off_zhg // G      # when G == 1, off_btk = off_zhg
    
    # batch_idx = off_btk // top_k
    idx_in_topk = off_btk % top_k
    off_z = off_btk // top_k    # batch index

    # Load off_h (head index) from batch_head_index
    batch_head_index_ptr = batch_head_index + off_z * top_k + idx_in_topk
    off_h = tl.load(batch_head_index_ptr, mask=True, other=0).to(tl.int64)
    
    # if we want to update kv cache inside the kernel, load the new key and value from the arguments (pointers, needs to be added) and update the cache in K and V by writing to seq_len position in the cache
    # this will only update the kv cache for the selected heads, the cache will not be updated for all heads 
    # ... #TODO ? # 

    # Compute lo and hi
    lo = (splitk_idx * BLOCK_N_PER_SPLIT).to(tl.int32)
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z).to(tl.int32)
    else:
        kv_len = N_CTX_K.to(tl.int32)
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len).to(tl.int32)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )

    k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
    # Additional shift by 1 along the last dimension in the quantized case, since
    # the first element along that dim contains packed quantization coefficients.
    K_block_ptr = tl.make_block_ptr(
        base=k_base + stride_kk * QUANTIZED * N_GROUPS,
        shape=(PACKED_D_PER_GROUP, hi),
        strides=(stride_kk, stride_kn),
        offsets=(0, lo),
        block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
        order=(0, 1),
    )
    v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg
    V_block_ptr = tl.make_block_ptr(
        base=v_base + stride_vk * QUANTIZED * N_GROUPS,
        shape=(hi, PACKED_D_PER_GROUP),
        strides=(stride_vn, stride_vk),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
        order=(1, 0),
    )

    if QUANTIZED:
        # Pointers to quantization coefficients
        K_scale_shift_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(1, hi),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        V_scale_shift_block_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(hi, 1),
            strides=(stride_vn, stride_vk),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
    else:
        K_scale_shift_block_ptr = None
        V_scale_shift_block_ptr = None

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)  # noqa: F821

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(  # noqa: F821
        tl.advance(Q_block_ptr, (0, 0)), boundary_check=(0,))  
    q = (q * qk_scale).to(q.dtype)

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        k, v = load_dequantize_k_v_group(
            K_block_ptr,
            V_block_ptr,
            K_scale_shift_block_ptr,
            V_scale_shift_block_ptr,
            BOUNDS_CHECKS_N,
            PACKED_PER_VAL,
            PACKED_D_PER_GROUP,
            Q.dtype.element_ty,
            0,
        )

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)  # noqa: F821

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if PACKED_PER_VAL > 1:
            K_scale_shift_block_ptr = tl.advance(
                K_scale_shift_block_ptr, (0, BLOCK_N)
            )
            V_scale_shift_block_ptr = tl.advance(
                V_scale_shift_block_ptr, (BLOCK_N, 0)
            )

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )
    tl.store(
        tl.advance(O_block_ptr, (0, 0)),
        acc,
        boundary_check=(0,),
    )
    # Write metadata for split-K reduction
    Metadata_ptr = (
        Metadata
        + off_zhg * stride_mzhg
        + splitk_idx * stride_ms
        + start_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )
    tl.store(Metadata_ptr, m_i)
    tl.store(Metadata_ptr + stride_m2, l_i)
    
@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Out,  # [B, H, M, K]
    LSE,  # [B, H, M]
    batch_head_index, # [B, top_k]
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_ok,
    stride_lse_zhg,
    stride_lse_m,
    M_ceil:tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    split_k:tl.constexpr,
    splitK_pow2:tl.constexpr,
    use_mask:tl.constexpr,
    top_k:tl.constexpr,
):        

    off_zhg = tl.program_id(0).to(tl.int64)
    off_m = tl.program_id(1).to(tl.int64)
    off_k = tl.program_id(2).to(tl.int64)
    
    off_g = off_zhg % G
    off_btk = off_zhg // G

    idx_in_topk = off_btk % top_k
    off_z = off_btk // top_k

    # Load off_h (head index) from batch_head_index
    batch_head_index_ptr = batch_head_index + off_z * top_k + idx_in_topk
    off_h = tl.load(batch_head_index_ptr, mask=True, other=0).to(tl.int64)
    
    # Read chunk
    spk_idx = tl.arange(0, splitK_pow2)
    kidx = tl.arange(0, BLOCK_SIZE)

    Metadata_ptr = (
        Metadata
        + stride_mzhg * off_zhg
        + spk_idx * stride_ms
        + off_m * stride_mm
    )

    o_ptr = (
        Out_splitK
        + off_zhg * stride_osk_zhg
        + stride_osk_m * off_m
        + off_k * BLOCK_SIZE
        + stride_osk_s * spk_idx[:, None]
        + kidx[None, :] * stride_osk_k
    )

    # read max values of each splitK
    if use_mask:
        spk_mask = spk_idx < split_k
        l_m = tl.load(Metadata_ptr, mask=spk_mask, other=float("-inf"))
        l_sum = tl.load(Metadata_ptr + stride_m2, mask=spk_mask, other=0.0)
        acc = tl.load(o_ptr, mask=spk_mask[:,None], other=0.0)
    else:
        l_m = tl.load(Metadata_ptr)
        l_sum = tl.load(Metadata_ptr + stride_m2)
        acc = tl.load(o_ptr)

    g_m = tl.max(l_m, axis=0)
    alpha = tl.math.exp2(l_m - g_m)

    # read sum
    l_sum *= alpha
    g_sum = tl.sum(l_sum, axis=0)
    acc = acc * alpha[:, None]
    acc_out = tl.sum(acc, axis=0) / g_sum
    Out_ptr = (
        Out
        + stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + off_k * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    tl.store(Out_ptr, acc_out)
    l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m
    tl.store(l_ptrs, (g_m + tl.math.log2(g_sum)) / 1.44269504)

class select_attention(torch.autograd.Function):

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }
    SUPPORTED_MAX_K = 128
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "triton_splitKF"

    @staticmethod
    def forward(cls, q, k, v, scale_float, batch_head_index, seq_len=None):

        cls.SPLIT_K: Optional[int] = None
        cls.BLOCK_M = 16
        cls.BLOCK_N = 64

        cls.NUM_GROUPS = 1  # Default quantization is row-wise

        # attn_bias = inp.attn_bias
        # seq_len = None

        # Transpose in the case of MQA/GQA
        mqa_swap_seqlen_head = False
        if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
            mqa_swap_seqlen_head = True
            assert q.shape[1] == 1
            q = q.transpose(1, 3)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]

        if k.dtype == torch.int32:
            # Quantized K/V
            PACKED_PER_VAL = 8
            Lk = (k.shape[-1] - cls.NUM_GROUPS) * 8
        else:
            Lk = k.shape[-1]
            PACKED_PER_VAL = 1

        B, Mk_original, G, H, Kkv = k.shape
        B, M, G, H, Kq = q.shape
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"
        # print(f"B = {B}, M = {M}, G = {G}, H = {H}, Kkv = {Kkv}, Kq = {Kq}")
        
        # Ensure batch_head_index is a torch tensor of dtype torch.int32
       # Adjust batch_head_index to the new shape and flatten it
        batch_head_index = batch_head_index.to(q.device)
        batch_head_index = batch_head_index.to(torch.int32)
        # print(f"batch_head_index= {batch_head_index}")
        # print(f"batch_head_index.shape = {batch_head_index.shape}")
        B, top_k = batch_head_index.shape
        # batch_head_index_flat = batch_head_index.flatten()  # Shape (B * top_k,)        # Is this CG safe? 
        N_selected = B * top_k
        
        if seq_len is not None and isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.to(q.device)

        # Determine if seq_len is an integer or tensor
        if isinstance(seq_len, int):
            Mk = seq_len  # Fixed sequence length for all batches
            use_seq_len = False
        elif seq_len is not None:   # seq_len is a tensor
            Mk = Mk_original  # Use original Mk from k.shape[1]
            use_seq_len = True
        else:   # seq_len is None, use Mk_original for all batches
            Mk = Mk_original
            use_seq_len = False

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = get_split_k(B, G, H, Mk)

        M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        o_splitk = torch.zeros(
            [B * G * H, split_k, M_ceil, Kq], dtype=torch.float32, device=q.device
        )
        metadata = torch.empty(
            [B * G * H, 2, split_k, M_ceil], dtype=torch.float32, device=q.device
        )
        lse = torch.empty((B * G * H, M), device=q.device, dtype=torch.float32)
        
        # remove this if not needed
        # o_splitk.zero_()
        # metadata.zero_()
        # lse.zero_()
        
        # grid = (triton.cdiv(M, BLOCK_M), B * G * H, split_k)
        # grid = (triton.cdiv(M, BLOCK_M), N_selected * G, split_k)
        grid = (triton.cdiv(M, BLOCK_M), N_selected, split_k)   # TODO: fix this for GQA
        
        num_warps = 1
        split_size = (Mk + split_k - 1) // split_k
        
        # use_seq_len = seq_len is not None

        # print(f"B = {B}, G = {G}, H = {H}, split_k = {split_k}, M_ceil = {M_ceil}, Kq = {Kq}, num_of_wgs = {G * G * H * split_k}")

        _fwd_kernel_splitK[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=scale_float,
            Out_splitK=o_splitk,
            Metadata=metadata,
            Seq_len=seq_len,
            batch_head_index=batch_head_index,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            Z=B,
            H=H,
            G=G,
            N_CTX_Q=M,
            N_CTX_K=Mk,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_seq_len,
            USE_SEQ_LEN=use_seq_len,
            num_warps=num_warps,
            num_stages=1,
            PACKED_PER_VAL=PACKED_PER_VAL,
            N_GROUPS=cls.NUM_GROUPS if PACKED_PER_VAL > 1 else 1,
            top_k=top_k,
        )

        if mqa_swap_seqlen_head:
            out = torch.zeros(
                (B, H, G, M, Kq), device=q.device, dtype=q.dtype
            ).transpose(1, 3)
        else:
            out = torch.zeros((B, M, G, H, Kq), device=q.device, dtype=q.dtype)

        # Merge together
        splitK_pow2 = triton.next_power_of_2(split_k)
        use_mask = splitK_pow2 > split_k
        if B * G * H * M >= 512:
            k_block_num = 1
        else:
            k_block_num = 2
        assert out.shape[-1] % k_block_num == 0
        k_block_size = out.shape[-1] // k_block_num
        # grid = (B * G * H, M, k_block_num)
        grid = (N_selected, M, k_block_num)  # TODO: fix this for GQA
        
        _splitK_reduce[grid](
            o_splitk,
            metadata,
            out,
            lse,
            batch_head_index=batch_head_index,
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            **_strides(out, "oz", "om", "og", "oh", "ok"),
            **_strides(lse, "lse_zhg", "lse_m"),
            M_ceil=M_ceil,
            BLOCK_SIZE=k_block_size,
            G=G,
            H=H,
            # TODO: Tune num_warps
            split_k=split_k,
            splitK_pow2=splitK_pow2,
            use_mask=use_mask,
            num_warps=4,
            top_k=top_k,
        )

        lse = lse.reshape([B, G, H, M])
        if mqa_swap_seqlen_head:
            # H/M dimensions have been swapped
            out = out.transpose(1, 3)
            lse = lse.transpose(2, 3)
        if q.ndim == 4:
            # BMGHK -> BMHK
            assert G == 1
            out = out[:, :, 0]
            lse = lse[:, 0]
        if Mk == 0:
            out.zero_()
        if mqa_swap_seqlen_head:
            out = out.reshape(B, -1, M * G, Kq).transpose(1, 2).contiguous()
        else:
            out = out.reshape(B, H * G, -1, Kq).contiguous()

        return out

select_attn = select_attention.apply