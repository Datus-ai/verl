#!/usr/bin/env python3
"""Diagnose verl configuration for multi-turn rollout"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

# Load the config
config_path = Path(__file__).parent / "agent_tool_trainer.yaml"
cfg = OmegaConf.load(config_path)

print("=" * 80)
print("VERL Configuration Diagnosis")
print("=" * 80)

# Data config
print("\n[Data Config]")
print(f"  max_prompt_length:    {cfg.data.max_prompt_length}")
print(f"  max_response_length:  {cfg.data.max_response_length}")
print(f"  train_batch_size:     {cfg.data.train_batch_size}")

# Rollout config
print("\n[Rollout Config]")
print(f"  prompt_length:        {cfg.actor_rollout_ref.rollout.prompt_length}")
print(f"  response_length:      {cfg.actor_rollout_ref.rollout.response_length}")
print(f"  max_model_len:        {cfg.actor_rollout_ref.rollout.max_model_len}")
print(f"  max_num_batched_tokens: {cfg.actor_rollout_ref.rollout.max_num_batched_tokens}")

# Multi-turn config
print("\n[Multi-Turn Config]")
mt = cfg.actor_rollout_ref.rollout.multi_turn
print(f"  enable:                      {mt.enable}")
print(f"  max_assistant_turns:         {mt.max_assistant_turns}")
print(f"  max_tool_response_length:    {mt.max_tool_response_length}")

# Token budget calculation
print("\n[Token Budget Analysis]")
prompt_len = cfg.actor_rollout_ref.rollout.prompt_length
response_len = cfg.actor_rollout_ref.rollout.response_length
max_model = cfg.actor_rollout_ref.rollout.max_model_len
max_turns = mt.max_assistant_turns
tool_resp_len = mt.max_tool_response_length

# IMPORTANT: prompt_length and response_length are CONFIG values, not actual usage
# In verl, max_new_tokens = prompt_length + response_length - len(prompt_ids)
base_budget = prompt_len + response_len

print(f"  prompt_length (config):     {prompt_len}")
print(f"  response_length (config):   {response_len}")
print(f"  Base budget (sum):          {base_budget}")
print(f"  max_model_len:              {max_model}")

# Estimate actual multi-turn usage (assuming typical BIRD dataset)
# Typical initial prompt: ~1500 tokens
# Typical response per turn: ~1500 tokens
# Tool result per turn: ~512 tokens (max)
estimated_initial_prompt = 1500
estimated_response_per_turn = 1500

print(f"\n  Estimated multi-turn usage (typical BIRD dataset):")
print(f"    Initial prompt:        ~{estimated_initial_prompt} tokens")
print(f"    Response per turn:     ~{estimated_response_per_turn} tokens")
print(f"    Tool result per turn:  ~{tool_resp_len} tokens")
print(f"    Max turns:             {max_turns}")

cumulative = estimated_initial_prompt
for turn in range(1, max_turns + 1):
    cumulative += estimated_response_per_turn + tool_resp_len
    max_new_tokens = base_budget - cumulative
    status = "✅" if max_new_tokens >= 0 else "❌"
    print(f"      Turn {turn}: cumulative={cumulative}, max_new_tokens={max_new_tokens} {status}")

total_estimated = estimated_initial_prompt + max_turns * (estimated_response_per_turn + tool_resp_len)
print(f"    Total estimated:       {total_estimated} tokens")

print(f"\n[Validation]")
print(f"  Base budget: {base_budget}")
print(f"  Estimated total usage: {total_estimated}")

if base_budget < 0:
    print("  ❌ CRITICAL: Base budget is negative!")
elif base_budget < total_estimated:
    print(f"  ⚠️  WARNING: Base budget ({base_budget}) < estimated usage ({total_estimated})")
    print(f"     This will cause 'max_new_tokens < 0' error!")
    required_base = total_estimated + 2048  # Add 2K buffer
    print(f"\n  [Suggested Fix]")
    print(f"    Set: max_prompt_length: {required_base // 2}")
    print(f"         max_response_length: {required_base // 2}")
elif total_estimated > max_model:
    print(f"  ⚠️  WARNING: Estimated usage ({total_estimated}) > max_model_len ({max_model})")
    print(f"     Increase max_model_len to at least {total_estimated + 2048}")
else:
    print(f"  ✅ Configuration looks valid for typical cases")
    print(f"     Remaining budget: {base_budget - total_estimated} tokens")

print("\n" + "=" * 80)
