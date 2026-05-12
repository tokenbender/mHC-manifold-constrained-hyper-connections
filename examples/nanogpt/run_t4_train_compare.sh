#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_ITERS="${MAX_ITERS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS="${EVAL_ITERS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
WANDB_LOG="${WANDB_LOG:-False}"

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="$3"

  echo "training ${name} -> ${out_dir}"
  (
    cd "${SCRIPT_DIR}"
    python train.py "${config}" \
    "out_dir='${out_dir}'" \
    "max_iters=${MAX_ITERS}" \
    "eval_interval=${EVAL_INTERVAL}" \
    "eval_iters=${EVAL_ITERS}" \
    "batch_size=${BATCH_SIZE}" \
    "gradient_accumulation_steps=${GRAD_ACCUM}" \
    "device='${DEVICE}'" \
    "dtype='${DTYPE}'" \
    "wandb_log=${WANDB_LOG}" \
    "compile_model=False"
  )
}

run_variant "baseline" "config/train_fineweb10B_t4.py" "out-t4-baseline"
run_variant "hc" "config/train_fineweb10B_hc_t4.py" "out-t4-hc"
run_variant "mhc" "config/train_fineweb10B_mhc_t4.py" "out-t4-mhc"
