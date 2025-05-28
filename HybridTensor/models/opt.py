from HybridTensor.utils.activations import OPT_MODELS
import torch
import math
from einops import rearrange

from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.models.opt import remap_state_dict_hf_opt
from HybridTensor.modules.SelectiveRouters import create_mlp_router_state_dict, create_attn_router_state_dict
from HybridTensor.models.create_sparse_model import GPTLMHeadModel as GPTLMHeadModelSparse
from flash_attn.models.gpt import GPTLMHeadModel

from transformers.models.opt import OPTConfig
from flash_attn.models.opt import opt_config_to_gpt2_config

class SparseConfig:
    def __init__(self):
        self.mlp_low_rank_dim = 1024
        self.attn_low_rank_dim = 128
        self.mlp_act_th = 0.5
        self.attn_topk = 0.3

def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    shared_state_dict = {}

    def shard_first_dim(new, old, key):
        x = old[key]
        dim = x.shape[0] // world_size
        new[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(new, old, key):
        x = old[key]
        dim = x.shape[-1] // world_size
        new[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(new, old, key):
        x = rearrange(old[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        new[key] = rearrange(
            x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ..."
        )

    shard_first_dim(shared_state_dict, state_dict, "transformer.embeddings.word_embeddings.weight")
    
    if "lm_head.weight" in state_dict:
        shard_first_dim(shared_state_dict, state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(shared_state_dict, state_dict, "transformer.embeddings.position_embeddings.weight")
        
    for i in range(config.num_hidden_layers):
        # attention
        shard_qkv_headdim(shared_state_dict, state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(shared_state_dict, state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        shard_last_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mixer.out_proj.weight")
        
        # mlp
        shard_first_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        shard_first_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc1.bias")
        shard_last_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc2.weight")
        
        if rank == 0:
            shared_state_dict[f"transformer.layers.{i}.mlp.fc2.bias"] = state_dict[f"transformer.layers.{i}.mlp.fc2.bias"]
            shared_state_dict[f"transformer.layers.{i}.mixer.out_proj.bias"] = state_dict[f"transformer.layers.{i}.mixer.out_proj.bias"]

        shared_state_dict[f"transformer.layers.{i}.norm1.weight"] = state_dict[f"transformer.layers.{i}.norm1.weight"]
        shared_state_dict[f"transformer.layers.{i}.norm1.bias"] = state_dict[f"transformer.layers.{i}.norm1.bias"]
        shared_state_dict[f"transformer.layers.{i}.norm2.weight"] = state_dict[f"transformer.layers.{i}.norm2.weight"]
        shared_state_dict[f"transformer.layers.{i}.norm2.bias"] = state_dict[f"transformer.layers.{i}.norm2.bias"]
        
        # routers
        
        # mlp router
        shared_state_dict[f"transformer.layers.{i}.mlp_router.fc1.weight"] = state_dict[f"transformer.layers.{i}.mlp_router.fc1.weight"]
        shard_first_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mlp_router.fc2.weight")
        
        # mha router
        shard_first_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mha_router.linear1.weight")
        shard_first_dim(shared_state_dict, state_dict, f"transformer.layers.{i}.mha_router.linear1.bias")

    shared_state_dict[f"transformer.ln_f.weight"] = state_dict["transformer.ln_f.weight"]
    shared_state_dict[f"transformer.ln_f.bias"] = state_dict["transformer.ln_f.bias"]
        
    # shared_state_dict[f"transformer.ln_f.weight"] = state_dict["transformer.final_layer_norm.weight"]
    # shared_state_dict[f"transformer.ln_f.bias"] = state_dict["transformer.final_layer_norm.bias"]
        
    return shared_state_dict

'''
def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[0] // world_size
        state_dict[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[-1] // world_size
        state_dict[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(
            x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ..."
        )

    shard_first_dim(state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        shard_first_dim(state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(state_dict, "transformer.embeddings.position_embeddings.weight")
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mixer.out_proj.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mixer.out_proj.bias")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mlp.fc2.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mlp.fc2.bias")
    return state_dict


'''

def build_sparse_opt(args, model_name, mlp_ckpt_dir, attn_ckpt_dir, device = None, dtype=torch.float16, process_group = None, world_size = None, rank = None):
    # dtype = torch.float16
    
    config = OPTConfig.from_pretrained(model_name)
    config = opt_config_to_gpt2_config(config)
    
    if device in ('cpu', torch.device('cpu')):
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        config.use_flash_attn = False
        config.fused_bias_fc = False
    else:
        config.fused_mlp = True
        config.fused_dropout_add_ln = True
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.sequence_parallel = False
    
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.pad_vocab_size_multiple = 8
    config.mlp_sparse = True
    config.att_sparse = True
    
    config.use_heuristic = True
    if config.use_heuristic:
        print("Using pre-compiled heuristic")
    else:
        print("Compiling new heuristic during runtime")
        
    spconfig = SparseConfig()
    spconfig.mlp_act_th = 0.5   # sets the threshold for the MLP routers for all layers
    spconfig.attn_topk = args.attn_topk    # sets the topk for the attention routers for all layers
    
    # build model
    print("Bulding Model with sparse routers")
    model_sparse = GPTLMHeadModelSparse(config = config, sp_config = spconfig, process_group = process_group, device = device, dtype=dtype)
    # print(model_sparse)
    
    # load pretrained weights into the sparse model
    state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
    state_dict = remap_state_dict_hf_opt(state_dict, config)
    
    # load the routers into the model
    if mlp_ckpt_dir is not None and attn_ckpt_dir is not None:
        mlp_router_state_dict = create_mlp_router_state_dict(mlp_ckpt_dir)
        attn_router_state_dict = create_attn_router_state_dict(attn_ckpt_dir)
            
        # merge the state dict
        merged_state_dict = {**state_dict, **mlp_router_state_dict, **attn_router_state_dict}
        
        if process_group is not None:
            merged_state_dict = shard_state_dict_tp(merged_state_dict, config, world_size, rank)
        
        model_sparse.load_state_dict(merged_state_dict, strict=True)
    else:
        if process_group is not None:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        model_sparse.load_state_dict(state_dict, strict=False)
    
    return model_sparse

def build_dense_opt(model_name, device = None, dtype=torch.float16, process_group = None, world_size = None, rank = None):
    dtype = torch.float16
    
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    # config.fused_dropout_add_ln = True
    config.sequence_parallel = False
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.pad_vocab_size_multiple = 8

    # build model
    print("Bulding Dense Model")
    model = GPTLMHeadModel.from_pretrained(model_name, config, process_group = process_group, world_size = world_size, rank = rank, device=device, dtype=dtype)

    return model