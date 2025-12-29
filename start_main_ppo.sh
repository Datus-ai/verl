#!/bin/sh
export CUDA_RT_DIR=/root/miniconda3/envs/verl-sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH="$CUDA_RT_DIR:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu"

python -m verl.trainer.main_ppo --config-path /root/verl/verl/trainer/config --config-name agent_tool_trainer.yaml