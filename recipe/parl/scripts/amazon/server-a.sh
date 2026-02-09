#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve checkpoints/personalized_baselines/amazon_parl_a/global_step_1068/merged \
  --tokenizer Qwen/Qwen3-8B \
  --port 8000 \
  --seed 42 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --api-key EMPTY \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.92 \
  --served-model-name qwen3-parl-a
