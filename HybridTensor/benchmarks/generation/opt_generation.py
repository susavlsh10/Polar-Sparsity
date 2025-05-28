import re
import time

import pytest
import torch
import argparse

from einops import rearrange

from HybridTensor.benchmarks.generation.gen_util import tokenize_dataset, get_random_batch
from HybridTensor.utils.activations import OPT_MODELS
from datasets import load_dataset

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.opt import opt_config_to_gpt2_config, remap_state_dict_hf_opt
from flash_attn.utils.generation import update_graph_cache
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import AutoTokenizer, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

def test_opt_generation(model_name):
    """Check that our implementation of OPT generation matches the HF implementation:
    the scores in fp16 should be around the same as the HF scores in fp16, when compared to
    the HF scores in fp32.
    """
    print(f"\nMODEL: {model_name}")
    verbose = False
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    # OPT tokenizer requires use_fast=False
    # https://huggingface.co/docs/transformers/model_doc/opt
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    eos_token_id = tokenizer.eos_token_id

    input_ids = tokenizer("Hello, my dog is cute and he", return_tensors="pt").input_ids.to(
        device=device
    )
    max_length = 25
    # input_ids = torch.randint(0, 100, (2, 10), dtype=torch.long, device='cuda')
    # max_length = input_ids.shape[1] + 40

    # Slow generation for reference
    sequences = []
    scores = []
    cur_input_ids = input_ids
    with torch.inference_mode():
        scores.append(model(cur_input_ids).logits[:, -1])
        sequences.append(scores[-1].argmax(dim=-1))
        for _ in range(input_ids.shape[1] + 1, max_length):
            cur_input_ids = torch.cat([cur_input_ids, rearrange(sequences[-1], "b -> b 1")], dim=-1)
            scores.append(model(cur_input_ids).logits[:, -1])
            sequences.append(scores[-1].argmax(dim=-1))
            if eos_token_id is not None and (sequences[-1] == eos_token_id).all():
                break
    sequences = torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1)
    scores = tuple(scores)

    print("Without CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=True,
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    if verbose:
        print(out.sequences)
    print(tokenizer.batch_decode(out.sequences.tolist()))
    if getattr(config, "use_flash_attn", False):
        # Capture graph outside the timing loop
        batch_size, seqlen_og = input_ids.shape
        model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
        print("With CUDA graph")
        torch.cuda.synchronize()
        start = time.time()
        out_cg = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=True,
        )
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
        if verbose:
            print(out_cg.sequences)
        print(tokenizer.batch_decode(out_cg.sequences.tolist()))

    del model

    model_hf = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device=device)
    model_hf.eval()
    print("HF fp16")
    torch.cuda.synchronize()
    start = time.time()
    out_hf = model_hf.generate(
        input_ids=input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    del model_hf

    model_ref = OPTForCausalLM.from_pretrained(model_name).to(device=device)
    model_ref.eval()
    print("HF fp32")
    torch.cuda.synchronize()
    start = time.time()
    out_ref = model_ref.generate(
        input_ids=input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    del model_ref
    print(tokenizer.batch_decode(out_ref.sequences.tolist()))

    if verbose:
        print(
            f"Scores max diff: {(torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item()}"
        )
        print(
            f"Scores mean diff: {(torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().mean().item()}"
        )
        print(
            f"HF fp16 max diff: {(torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item()}"
        )
        print(
            f"HF fp16 mean diff: {(torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)).abs().mean().item()}"
        )

    assert torch.all(out.sequences == sequences)
    assert torch.allclose(
        torch.stack(out.scores, dim=1), torch.stack(scores, dim=1), rtol=rtol, atol=atol
    )
    assert torch.all(out.sequences == out_ref.sequences)
    assert torch.all(out.sequences == out_hf.sequences)

    assert (torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item() < 3 * (
        torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)
    ).abs().max().item()


def arg_parser():
    parser = argparse.ArgumentParser(description='Inference benchmarking')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_index', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--index_size', type=int, default=8192)
    parser.add_argument('--head_density', type=float, default=0.25)
    parser.add_argument('--print_results', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--check_results', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = arg_parser()
    model_name = OPT_MODELS[args.model_index-1]
    # test_opt_generation(model_name)
    
    print(f"\nMODEL: {model_name}\n")
    verbose = False
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    # OPT tokenizer requires use_fast=False
    # https://huggingface.co/docs/transformers/model_doc/opt
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    eos_token_id = tokenizer.eos_token_id

    # input_ids = tokenizer("In a distant galaxy, a spaceship", return_tensors="pt").input_ids.to(
    #     device=device
    # )
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    tokens = tokenize_dataset(dataset, tokenizer)
    input_ids = get_random_batch(tokens, args.batch_size, args.seq_len)
    input_ids = input_ids.to(device=device)
    max_length = args.seq_len + 20
    
    # input_ids = torch.randint(0, 100, (2, 10), dtype=torch.long, device='cuda')
    # max_length = input_ids.shape[1] + 40

    # warm up
    _ = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
    )

    print("Without CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
    )
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) * 1000
    print(f"Prompt processing + decoding time: {elapsed_time:.0f} ms")
    
    # Compute throughput and latency per token
    num_tokens_generated = out.sequences.shape[1] - input_ids.shape[1]
    throughput = (args.batch_size * num_tokens_generated) / (elapsed_time / 1000)
    latency_per_token = elapsed_time / num_tokens_generated  # ms per token
    
    # print(f"Number of tokens generated: {num_tokens_generated}")
    print(f"Throughput: {throughput:.1f} tokens/second")
    print(f"Latency per token: {latency_per_token:.1f} ms")
    
    
    if args.print_results:
        # print(out.sequences)
        print(tokenizer.batch_decode(out.sequences.tolist()))
    
    # ============================================================================= #
    
    print("\n")
    if getattr(config, "use_flash_attn", False):
        # Capture graph outside the timing loop
        batch_size, seqlen_og = input_ids.shape
        model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
        print("With CUDA graph")
        torch.cuda.synchronize()
        start = time.time()
        out_cg = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
        )
        torch.cuda.synchronize()
        elapsed_time = (time.time() - start) * 1000
        print(f"Prompt processing + decoding time: {elapsed_time:.0f} ms")
        
        # Compute throughput and latency per token
        num_tokens_generated = out.sequences.shape[1] - input_ids.shape[1]
        latency_per_token = elapsed_time / num_tokens_generated  # ms per token
        throughput = (args.batch_size * num_tokens_generated) / (elapsed_time / 1000)
        
        # print(f"Number of tokens generated: {num_tokens_generated}")
        print(f"Throughput: {throughput:.1f} tokens/second")
        print(f"Latency per token: {latency_per_token:.1f} ms")

        if args.print_results:
            # print(out_cg.sequences)
            print(tokenizer.batch_decode(out_cg.sequences.tolist()))