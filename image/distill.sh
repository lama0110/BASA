#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
export DATAPATH="${DATAPATH:-./data_000000}"
export PRECISION="${PRECISION:-bf16}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export DOWN_FACTOR="${DOWN_FACTOR:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-500}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export REPORT_TO="${REPORT_TO:-wandb}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./BASA}"

if [ "${DOWN_FACTOR}" -eq 1 ]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/basa_local_${WINDOW_SIZE}}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/basa_local_${WINDOW_SIZE}_down_${DOWN_FACTOR}}"
fi

accelerate launch --config_file deepspeed_config.yaml distill.py \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --data_root="${DATAPATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --mixed_precision="${PRECISION}" \
  --dataloader_num_workers=8 \
  --resolution=1024 \
  --train_batch_size="${BATCH_SIZE}" \
  --gradient_accumulation_steps="${GRAD_ACCUM}" \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="${REPORT_TO}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="${TRAIN_STEPS}" \
  --validation_epochs=1 \
  --seed="0" \
  --checkpointing_steps=5000 \
  --use_cached_prompt_embed \
  --use_cached_latent \
  --gradient_checkpointing \
  --down_factor="${DOWN_FACTOR}" \
  --window_size="${WINDOW_SIZE}" \
  --pool_size=4 \
  --train_shift_steps=20 \
  --validation_num_inference_steps=20
