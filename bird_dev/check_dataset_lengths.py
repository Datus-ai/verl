#!/usr/bin/env python3
"""Check actual token lengths in BIRD dataset"""

import sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
tokenizer_path = "/root/onethingai-fs/Qwen3-4B"
print(f"Loading tokenizer from: {tokenizer_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    print("\nNote: Run this script on the server where the model is located")
    sys.exit(1)

# Load dataset
dataset_path = "/root/data/bird_dev_rl/train.parquet"
print(f"\nLoading dataset from: {dataset_path}")

try:
    df = pd.read_parquet(dataset_path)
    print(f"✅ Dataset loaded: {len(df)} samples")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Analyze prompt lengths
print("\n" + "=" * 80)
print("Analyzing Prompt Lengths")
print("=" * 80)

prompt_lengths = []
for idx, row in df.iterrows():
    if idx >= 100:  # Sample first 100
        break
    prompt = row.get('prompt', '')
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_lengths.append(len(tokens))

    if idx < 5:  # Show first 5
        print(f"Sample {idx}: {len(tokens)} tokens")

print(f"\nPrompt Length Statistics (first 100 samples):")
print(f"  Min:     {min(prompt_lengths)}")
print(f"  Max:     {max(prompt_lengths)}")
print(f"  Mean:    {sum(prompt_lengths) / len(prompt_lengths):.1f}")
print(f"  Median:  {sorted(prompt_lengths)[len(prompt_lengths)//2]}")

# Check if any prompts exceed limit
max_prompt_config = 9472
exceeding = [l for l in prompt_lengths if l > max_prompt_config]
if exceeding:
    print(f"\n⚠️  WARNING: {len(exceeding)} prompts exceed max_prompt_length ({max_prompt_config})")
    print(f"   Longest prompt: {max(exceeding)} tokens")
else:
    print(f"\n✅ All prompts fit within max_prompt_length ({max_prompt_config})")

print("\n" + "=" * 80)
