#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-75}"
EVAL_ITERS="${EVAL_ITERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
N_LAYER="${N_LAYER:-4}"
N_HEAD="${N_HEAD:-4}"
N_EMBD="${N_EMBD:-192}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
WANDB_LOG="${WANDB_LOG:-False}"
COMPILE_MODEL="${COMPILE_MODEL:-False}"
DATA_LOADER="${DATA_LOADER:-memmap}"

py_bool() {
  case "${1}" in
    true|True|TRUE|1|yes|YES|on|ON) echo "True" ;;
    false|False|FALSE|0|no|NO|off|OFF) echo "False" ;;
    *) echo "${1}" ;;
  esac
}

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="$3"
  local log_path="${LOG_DIR}/t4-mini-${name}.log"

  echo "[$(date -Is)] start ${name} -> ${out_dir}"
  (
    cd "${SCRIPT_DIR}"
    stdbuf -oL -eL python -u train.py "${config}" \
      "out_dir='${out_dir}'" \
      "max_iters=${MAX_ITERS}" \
      "eval_interval=${EVAL_INTERVAL}" \
      "eval_iters=${EVAL_ITERS}" \
      "batch_size=${BATCH_SIZE}" \
      "gradient_accumulation_steps=${GRAD_ACCUM}" \
      "block_size=${BLOCK_SIZE}" \
      "n_layer=${N_LAYER}" \
      "n_head=${N_HEAD}" \
      "n_embd=${N_EMBD}" \
      "device='${DEVICE}'" \
      "dtype='${DTYPE}'" \
      "wandb_log=$(py_bool "${WANDB_LOG}")" \
      "compile_model=$(py_bool "${COMPILE_MODEL}")" \
      "data_loader='${DATA_LOADER}'"
  ) 2>&1 | tee "${log_path}"
  echo "[$(date -Is)] end ${name} -> ${out_dir}"
}

run_variant "baseline" "config/train_fineweb10B_mini_t4.py" "out-t4-mini-baseline"
run_variant "hc" "config/train_fineweb10B_hc_mini_t4.py" "out-t4-mini-hc"
run_variant "mhc" "config/train_fineweb10B_mhc_mini_t4.py" "out-t4-mini-mhc"
