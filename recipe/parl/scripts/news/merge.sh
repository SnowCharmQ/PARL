#!/bin/bash

METHODS=(
    "news_parl_a"
    "news_parl_b"
)

for METHOD in "${METHODS[@]}"; do

    BASE_DIR="checkpoints/personalized_baselines/${METHOD}"

    MODEL_DIRS=(
        ${BASE_DIR}/global_step_*
        ${BASE_DIR}/best_checkpoint
    )

    for step_dir in "${MODEL_DIRS[@]}"; do
        step_name=$(basename "${step_dir}")
        
        echo "${step_name}"
        
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir ${BASE_DIR}/${step_name}/actor \
            --target_dir ${BASE_DIR}/${step_name}/merged
    done

done