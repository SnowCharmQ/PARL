#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0

CONFIG_PATH=recipe/parl/config

python -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name news_parl_b.yaml \
    2>&1 | tee logs/news_parl_b.log