import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_dataset(dataset, tokenizer):
    # Tokenize and concatenate all texts (without adding special tokens)
    all_tokens = []
    for example in dataset:
        tokens = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
    return all_tokens

def get_random_batch(tokens, batch_size, seq_length):
    total = len(tokens)
    batch = []
    for _ in range(batch_size):
        start = random.randint(0, total - seq_length)
        batch.append(tokens[start : start + seq_length])
    return torch.tensor(batch)

'''
# Load dataset and tokenizer
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokens = tokenize_dataset(dataset, tokenizer)

# Define parameters
batch_size = 8
seq_length = 2000

random_batch = get_random_batch(tokens, batch_size, seq_length)
print("Batch shape:", random_batch.shape)  # Expected: (8, 128)

'''

