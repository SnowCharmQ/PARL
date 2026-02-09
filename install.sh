#!/bin/bash

uv pip install gputil
uv pip install -e .
uv pip install vllm==0.11.2

uv pip install flash-attn==2.6.3 --no-build-isolation --no-cache
uv pip install evaluate rank_bm25 sacrebleu rouge_score absl-py bert_score
uv pip install fastapi uvicorn pydantic
uv pip install wandb nvitop