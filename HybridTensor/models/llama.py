from transformers import LlamaConfig, LlamaTokenizer

import torch
import torch.nn as nn
from HybridTensor.models.create_sparse_model import GPTLMHeadModel
from HybridTensor.modules.SelectiveRouters import create_mlp_router_state_dict, create_attn_router_state_dict

# from flash_attn.models.gpt import GPTLMHeadModel
from transformers import AutoConfig, AutoTokenizer
from flash_attn.utils.pretrained import state_dict_from_pretrained

from flash_attn.models.llama import (
    config_from_checkpoint,
    inv_remap_state_dict_hf_llama,
    llama_config_to_gpt2_config,
    remap_state_dict_hf_llama,
    remap_state_dict_meta_llama,
    state_dicts_from_checkpoint,
)

class SparseConfig:
    def __init__(self):
        self.mlp_low_rank_dim = 1024
        self.attn_low_rank_dim = 128
        self.mlp_act_th = 0.5
        self.attn_topk = 0.3

def build_dense_llama(model_name: str, device = None, dtype=torch.float16, process_group = None, world_size = None, rank = None, **kwargs):
    config = llama_config_to_gpt2_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True
    config.prenorm = True
    
    state_dict = state_dict_from_pretrained(model_name, device='cpu', dtype=dtype)
    state_dict = remap_state_dict_hf_llama(state_dict, config)

    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model

def build_sparse_llama(args, model_name: str, attn_ckpt_dir: str, device = None, dtype=torch.float16, process_group = None, world_size = None, rank = None, **kwargs):
    config = llama_config_to_gpt2_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True
    config.prenorm = True
    
    spconfig = SparseConfig()
    spconfig.attn_topk = args.attn_topk
    config.mlp_sparse = False
    config.att_sparse = True
    
    state_dict = state_dict_from_pretrained(model_name, device='cpu', dtype=dtype)
    state_dict = remap_state_dict_hf_llama(state_dict, config)

    model = GPTLMHeadModel(config, sp_config= spconfig, device=device, dtype=dtype)
    
    if attn_ckpt_dir is not None:
        attn_router_state_dict = create_attn_router_state_dict(attn_ckpt_dir)
        merged_state_dict = {**state_dict, **attn_router_state_dict}
    
    # TODO: Add code for tensor parallel state dict sharding
    
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    
    return model