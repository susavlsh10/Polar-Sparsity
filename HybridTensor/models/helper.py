import math
import re
from collections import OrderedDict

from einops import rearrange


def remap_state_dict_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("wte.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict[
        "transformer.embeddings.word_embeddings.weight"
    ]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(
            r"^h.(\d+).ln_(1|2).(weight|bias)", r"transformer.layers.\1.norm\2.\3", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f"h.{d}.mlp.c_fc.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc1.weight"] = W1.t()
        W2 = state_dict.pop(f"h.{d}.mlp.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc2.weight"] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub(
            r"^h.(\d+).mlp.c_fc.bias", r"transformer.layers.\1.mlp.fc1.bias", key
        )
        key = re.sub(
            r"^h.(\d+).mlp.c_proj.bias", r"transformer.layers.\1.mlp.fc2.bias", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f"h.{d}.attn.bias")  # We don't store this bias
        Wqkv = state_dict.pop(f"h.{d}.attn.c_attn.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = Wqkv.t()
        Wout = state_dict.pop(f"h.{d}.attn.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mixer.out_proj.weight"] = Wout.t()

    def key_mapping_attn(key):
        key = re.sub(
            r"^h.(\d+).attn.c_attn.bias", r"transformer.layers.\1.mixer.Wqkv.bias", key
        )
        key = re.sub(
            r"^h.(\d+).attn.c_proj.bias",
            r"transformer.layers.\1.mixer.out_proj.bias",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


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
