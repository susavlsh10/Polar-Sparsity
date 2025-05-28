import math
import torch
import torch.nn as nn
from functools import partial

from einops import rearrange

from transformers import GPT2Config
from collections import namedtuple
from HybridTensor.modules.SelectiveMHA import SMHA, SelectMHA, ParallelSelectMHA, MHARouter, ParallelMHARouter
from HybridTensor.modules.SelectiveMLP import SelectiveMLP, ParallelSelectiveMLP, MLPRouter, ParallelMLPRouter
from HybridTensor.modules.SelectiveBlock import SelectBlock
# from HybridTensor.modules.SelectiveBlock_v1 import SelectBlock
import torch.nn.functional as F
from flash_attn.utils.distributed import (
    all_gather,
    all_gather_raw,
    get_dim_for_local_rank,
    sync_shared_params,
)

from collections.abc import Sequence
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import FusedMLP, ParallelFusedMLP, GatedMlp, ParallelGatedMlp, Mlp, ParallelMLP
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.modules.block import Block

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import GenerationMixin
from flash_attn.models.opt import remap_state_dict_hf_opt

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

from HybridTensor.models.helper import remap_state_dict_gpt2, shard_state_dict_tp

def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    attn_scale_power = 0.5 if not getattr(config, "mup_scale_qk_dot_by_d", False) else 1.0
    softmax_scale = 1.0 if not config.scale_attn_weights else (head_dim ** (-attn_scale_power))
    softmax_scale *= getattr(config, "mup_attn_multiplier", 1.0)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    if dwconv:
        assert process_group is None, "TensorParallel MHA does not support dwconv yet"
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_alibi = getattr(config, "use_alibi", False)
    use_triton = getattr(config, "use_triton", True)    # toggle cuda or triton decode kernels
    window_size = getattr(config, "window_size", (-1, -1))
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    if not fused_bias_fc:
        assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    
    mlp_sparse = getattr(config, "mlp_sparse", False)
    att_sparse = getattr(config, "att_sparse", False)
    num_heads = getattr(config, "num_attention_heads", None)
    n_head_kv = getattr(config, "n_head_kv", num_heads)
    
    if num_heads != n_head_kv:
        att_sparse = False
    
    if process_group is None:
        mha_cls = SMHA # SelectMHA if att_sparse else MHA 
    else:
        mha_cls = ParallelSelectMHA if att_sparse else ParallelMHA
    
    # mha_cls = SelectMHA if process_group is None else ParallelSelectMHA
    serial_kwargs = (
        {"fused_bias_fc": fused_bias_fc, "dwconv": dwconv} if process_group is None else {}
    )
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", False),
        }
        if process_group is not None
        else {}
    )
    num_heads_kv = getattr(config, "n_head_kv", None)
    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        num_heads_kv=num_heads_kv,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_alibi=use_alibi,
        window_size=window_size,
        use_flash_attn=use_flash_attn,
        **serial_kwargs,
        **parallel_kwargs,
        **factory_kwargs,
    )
    return mixer_cls

def create_mlp_cls_old(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
        ]
    assert fused_mlp == True, "Not supported not fused mlp for now"

    mlp_sparse = getattr(config, "mlp_sparse", False)
    use_heuristic = getattr(config, "use_heuristic", True)

    mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
    # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
    if isinstance(mlp_checkpoint_lvl, Sequence):
        assert layer_idx is not None
        mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]

    if fused_mlp:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        # activation = (
        #     "gelu_approx"
        #     if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"]
        #     else "relu"
        # )
        
        if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]:
            activation = "gelu_approx"
        else:
            activation = "relu" # config.activation_function
        
        if process_group is None:
            mlp_cls = SelectiveMLP if mlp_sparse else FusedMLP
        else:
            mlp_cls = ParallelSelectiveMLP if mlp_sparse else ParallelFusedMLP

        parallel_kwargs = (
            {
                "process_group": process_group,
                "sequence_parallel": getattr(config, "sequence_parallel", True),
            }
            if process_group is not None
            else {}
        )
        
        sparsity_kwargs = (
            {
                "use_heuristic": use_heuristic,
            }
            if mlp_sparse
            else {}
        )

        mlp_cls = partial(
            mlp_cls,
            hidden_features=inner_dim,
            activation=activation,
            checkpoint_lvl=mlp_checkpoint_lvl,
            # layer_idx=layer_idx,
            **parallel_kwargs,
            **factory_kwargs,
            **sparsity_kwargs,
        )

    else:
        raise RuntimeError("MLP type not supported")
    return mlp_cls

def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    """
    Create an MLP class that supports both sparse MLPs (via fused mlp) and GatedMLPs.
    If the activation function is one of "glu", "swiglu", or "geglu", then GatedMlp is used
    (and mlp_sparse is ignored). Otherwise, fused_mlp is used to decide between sparse and
    dense implementations.
    """
    from functools import partial
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)


    # Check for gated activations
    if config.activation_function in ["glu", "swiglu", "geglu"]:
        # For gated activations we do not support sparsity yet. 
        activation = (
            F.sigmoid if config.activation_function == "glu"
            else (F.silu if config.activation_function == "swiglu" else F.gelu)
        )
        mlp_cls = GatedMlp if process_group is None else ParallelGatedMlp
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": getattr(config, "sequence_parallel", True)}
            if process_group is not None else {}
        )
        mlp_multiple_of = getattr(config, "mlp_multiple_of", 128)
        mlp_cls = partial(
            mlp_cls,
            hidden_features=config.n_inner,
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            multiple_of=mlp_multiple_of,
            **parallel_kwargs,
            **factory_kwargs,
        )
        return mlp_cls

    # For non-gated activations:
    fused_mlp = getattr(config, "fused_mlp", False)
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only supports approximate activation_function sqrelu"
        )
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    
    if fused_mlp:
        # Ensure valid activation function.
        assert config.activation_function in [
            "gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh", "relu", "sqrelu"
        ]
        # Support checkpoint level (possibly a list)
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        if isinstance(mlp_checkpoint_lvl, (list, tuple)):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        # Choose activation string.
        if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]:
            activation = "gelu_approx"
        else:
            activation = "relu"
        # Determine inner dim.
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        mlp_sparse = getattr(config, "mlp_sparse", False)
        use_heuristic = getattr(config, "use_heuristic", True)
        if process_group is None:
            mlp_cls = SelectiveMLP if mlp_sparse else FusedMLP
        else:
            mlp_cls = ParallelSelectiveMLP if mlp_sparse else ParallelFusedMLP
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": getattr(config, "sequence_parallel", True)}
            if process_group is not None else {}
        )
        sparsity_kwargs = {"use_heuristic": use_heuristic} if mlp_sparse else {}
        mlp_cls = partial(
            mlp_cls,
            hidden_features=inner_dim,
            activation=activation,
            checkpoint_lvl=mlp_checkpoint_lvl,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **parallel_kwargs,
            **factory_kwargs,
            **sparsity_kwargs,
        )
        return mlp_cls

    elif fused_dense_sqrelu_dense:
        if process_group is not None:
            assert fused_mlp, "Tensor Parallel is not implemented for FusedDenseSqreluDense"
        assert FusedDenseSqreluDense is not None
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        if isinstance(mlp_checkpoint_lvl, (list, tuple)):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedDenseSqreluDense,
            hidden_features=config.n_inner,
            checkpoint_lvl=mlp_checkpoint_lvl,
            **factory_kwargs,
        )
        return mlp_cls

    else:
        # Non-fused, non-sparse branch.
        assert config.activation_function in [
            "gelu", "gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh", "relu", "sqrelu"
        ]
        if config.activation_function == "relu":
            activation = partial(F.relu, inplace=True)
        elif config.activation_function == "sqrelu":
            activation = sqrelu_fwd
        else:
            approximate = "tanh" if config.activation_function in [
                "gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"
            ] else "none"
            activation = partial(F.gelu, approximate=approximate)
        mlp_sparse = getattr(config, "mlp_sparse", False)
        mlp_cls = Mlp if process_group is None else ParallelMLP
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": getattr(config, "sequence_parallel", True)}
            if process_group is not None else {}
        )
        mlp_cls = partial(
            mlp_cls,
            hidden_features=config.n_inner,
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **parallel_kwargs,
            **factory_kwargs,
        )
        return mlp_cls

def create_mlp_router_cls(config, sp_config = None, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_neurons = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    
    # this can be made different per layer by adding mlp_low_rank_dim_{layer_idx} in the sp_config
    low_rank_dim = getattr(sp_config, "mlp_low_rank_dim", 1024)
    
    # per layer activation threshold
    act_th = getattr(config, "mlp_act_th", 0.5)
    
    if process_group is None:
        mlp_router_cls = MLPRouter
    else:
        mlp_router_cls = ParallelMLPRouter
    
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", True),
        }
        if process_group is not None
        else {}
    )
    
    mlp_router_cls = partial(mlp_router_cls, 
                             low_rank_dim = low_rank_dim,
                             out_dim = num_neurons,
                             act_th = act_th,
                             **parallel_kwargs,
                             **factory_kwargs)

    return mlp_router_cls

def create_mha_router_cls(config, sp_config = None, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_heads = config.num_attention_heads
    n_head_kv = getattr(config, "n_head_kv", num_heads)
    if num_heads != n_head_kv:
        out_dim = n_head_kv
    else:
        out_dim = num_heads
        
    low_rank_dim = getattr(sp_config, "attn_low_rank_dim", 128) # optional, default to 128
    
    # per layer activation topk, to make this different per layer, add a different attn_topk_{layer_idx} in the sp_config
    attn_topk = getattr(sp_config, "attn_topk", 0.5)
    if process_group is None:
        mha_router_cls = MHARouter
    else:
        mha_router_cls = ParallelMHARouter
    
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", True),
        }
        if process_group is not None
        else {}
    )

    
    mha_router_cls = partial(mha_router_cls, 
                             low_rank_dim = low_rank_dim,
                             out_dim = out_dim,
                             top_k = attn_topk,
                             **parallel_kwargs,
                             **factory_kwargs)

    return mha_router_cls

def create_block(config, sp_config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs) 
    
    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=config.layer_norm_epsilon,
        **factory_kwargs,
    )
    
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    parallel_block = getattr(config, "parallel_block", False)
    mlp_sparse = getattr(config, "mlp_sparse", False)
    att_sparse = getattr(config, "att_sparse", False)
    block_sparse = mlp_sparse or att_sparse
    
    if not parallel_block:
        if block_sparse:
            mha_router_cls = create_mha_router_cls(config, sp_config, layer_idx, process_group=process_group, **factory_kwargs) if att_sparse else None
            mlp_router_cls = create_mlp_router_cls(config, sp_config, layer_idx, process_group=process_group, **factory_kwargs) if mlp_sparse else None
            
            block = SelectBlock(
                config.hidden_size,
                mixer_cls,
                mlp_cls,
                mlp_router = mlp_router_cls,
                mha_router = mha_router_cls,
                norm_cls=norm_cls,
                prenorm=prenorm,
                resid_dropout1=resid_dropout1,
                resid_dropout2=config.resid_pdrop,
                fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
                residual_in_fp32=residual_in_fp32,
                sequence_parallel=sequence_parallel and process_group is not None,
                mark_shared_params=process_group is not None,
            )
        else:
            block = Block(
                config.hidden_size,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                prenorm=prenorm,
                resid_dropout1=resid_dropout1,
                resid_dropout2=config.resid_pdrop,
                fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
                residual_in_fp32=residual_in_fp32,
                sequence_parallel=sequence_parallel and process_group is not None,
                mark_shared_params=process_group is not None,
            )

    else:
        # not implemented
        raise RuntimeError("ParallelBlock not implemented")
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        config,
        sp_config,
        *args,
        strict=True,
        device=None,
        dtype=None,
        world_size=1,
        rank=0,
        **kwargs,
    ):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, sp_config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            state_dict = remap_state_dict_gpt2(state_dict, config)
        elif model_name.startswith("facebook/opt"):
            state_dict = remap_state_dict_hf_opt(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        # logger.info(load_return)
        return model


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                )


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, sp_config=None, process_group=None, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.process_group = process_group
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)

        if process_group is None:
            self.embeddings = GPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                word_embed_proj_dim=word_embed_proj_dim,
                **factory_kwargs,
            )
        else:
            self.embeddings = ParallelGPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                process_group=process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )


        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(
                    config, sp_config, layer_idx=i, process_group=process_group, **factory_kwargs
                )
                for i in range(config.num_hidden_layers)
            ]
        )     
        

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if layer_norm_fn is None:
                raise ImportError("Triton is not installed")
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            # self.ln_f = nn.LayerNorm(
            #     config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs
            # )
            self.ln_f = norm_cls(
                config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs
            )
            

        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True

        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
            )
        )
        self.tie_weights()

        self.sparse = False
        if config.mlp_sparse or config.att_sparse:
            self.sparse = True

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention layers need to know the seqlen.
        embedding_kwargs = (
            {"combine_batch_seqlen_dim": True}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, **embedding_kwargs
        )
        residual = None
        mixer_kwargs = (
            {"seqlen": input_ids.shape[1]}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        else:
            mixer_kwargs["inference_params"] = None

        # else:
        for layer in self.layers:
            if self.prenorm:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    mixer_kwargs=mixer_kwargs,
                )
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if hidden_states.shape != residual.shape:
                    hidden_states = hidden_states.view(residual.shape)
                    
                hidden_states = layer_norm_fn(
                    hidden_states,
                    self.ln_f.weight,
                    self.ln_f.bias,
                    residual=residual,
                    x1=None,
                    eps=self.ln_f.eps,
                    dropout_p=self.drop_f.p if self.training else 0.0,
                    prenorm=False,
                    is_rms_norm=isinstance(self.ln_f, RMSNorm)
                )
        return hidden_states


class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPT2Config, sp_config = None, process_group=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.process_group = process_group
        
        self.transformer = GPTModel(
            config, sp_config, process_group=process_group, **factory_kwargs
        )
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = (
            config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        )
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(
                config.n_embd, embed_dim, bias=False, **factory_kwargs
            )
        else:
            self.project_out = None
        mup_width_scale = getattr(config, "mup_width_scale", 1.0)
        mup_output_multiplier = getattr(config, "mup_output_multiplier", 1.0)
        self.output_scale = mup_output_multiplier * mup_width_scale
        
        if process_group is None:
            self.lm_head = nn.Linear(
                embed_dim, vocab_size, bias=False, **factory_kwargs
            )
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                embed_dim,
                vocab_size,
                process_group,
                bias=False,
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )

        self.norm_head = getattr(config, "norm_head", False)
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight  # llama does not use tied weights
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)
            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        assert (
            input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if self.output_scale != 1.0:
            hidden_states = hidden_states * self.output_scale
        if not self.norm_head:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_head_weight = F.normalize(self.lm_head.weight)
            if isinstance(self.lm_head, ColumnParallelLinear) and self.lm_head.sequence_parallel:
                hidden_states = all_gather(hidden_states, self.lm_head.process_group)
            lm_logits = F.linear(hidden_states, lm_head_weight, bias=self.lm_head.bias)
        # During inference, we want the full logit for sampling
        if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, "(n b) ... d -> b ... (n d)", b=b)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(
                f"transformer.layers.{n_layers - 1}.norm2.weight"
            )
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(
                        f"transformer.layers.{l - 1}.norm2.weight"
                    )
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict[f"transformer.layers.0.norm1.weight"] = ln_weight
            state_dict[f"transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)