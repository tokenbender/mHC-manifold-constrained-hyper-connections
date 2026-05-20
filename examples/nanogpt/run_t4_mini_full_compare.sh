#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_BENCH="${RUN_BENCH:-true}"
RUN_PLOT="${RUN_PLOT:-true}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-75}"
EVAL_ITERS="${EVAL_ITERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
N_LAYER="${N_LAYER:-4}"
N_HEAD="${N_HEAD:-4}"
N_EMBD="${N_EMBD:-192}"
WANDB_LOG="${WANDB_LOG:-False}"
COMPILE_MODEL="${COMPILE_MODEL:-False}"
DATA_LOADER="${DATA_LOADER:-memmap}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-1}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-32}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERS="${NUM_ITERS:-20}"
REPORT_DIR="${REPORT_DIR:-${REPO_DIR}/reports/t4-mini-$(date +%Y%m%d-%H%M%S)}"

CKPT_BASELINE="${CKPT_BASELINE:-${SCRIPT_DIR}/out-t4-mini-baseline/ckpt.pt}"
CKPT_HC="${CKPT_HC:-${SCRIPT_DIR}/out-t4-mini-hc/ckpt.pt}"
CKPT_MHC="${CKPT_MHC:-${SCRIPT_DIR}/out-t4-mini-mhc/ckpt.pt}"

mkdir -p "${REPORT_DIR}/benchmarks"

py_bool_cli() {
  case "${1}" in
    true|True|TRUE|1|yes|YES|on|ON) echo "true" ;;
    false|False|FALSE|0|no|NO|off|OFF) echo "false" ;;
    *) echo "${1}" ;;
  esac
}

if [[ "${RUN_TRAIN}" == "true" ]]; then
  MAX_ITERS="${MAX_ITERS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_ITERS="${EVAL_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  BLOCK_SIZE="${BLOCK_SIZE}" \
  N_LAYER="${N_LAYER}" \
  N_HEAD="${N_HEAD}" \
  N_EMBD="${N_EMBD}" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  WANDB_LOG="${WANDB_LOG}" \
  COMPILE_MODEL="${COMPILE_MODEL}" \
  DATA_LOADER="${DATA_LOADER}" \
  bash "${SCRIPT_DIR}/run_t4_mini_compare.sh"
else
  echo "training skipped: RUN_TRAIN=${RUN_TRAIN}"
fi

for ckpt in "${CKPT_BASELINE}" "${CKPT_HC}" "${CKPT_MHC}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
done

run_benchmark() {
  local name="$1"
  local ckpt="$2"
  local config="$3"

  echo "[$(date -Is)] benchmark ${name}"
  stdbuf -oL -eL python -u "${SCRIPT_DIR}/benchmark_inference.py" \
    --ckpt "${ckpt}" \
    --config "${config}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch-size "${BENCH_BATCH_SIZE}" \
    --prompt-len "${PROMPT_LEN}" \
    --gen-len "${GEN_LEN}" \
    --num-warmup "${NUM_WARMUP}" \
    --num-iters "${NUM_ITERS}" \
    --compile "$(py_bool_cli "${COMPILE_MODEL}")" \
    --output-json "${REPORT_DIR}/benchmarks/${name}.json" \
    --output-csv "${REPORT_DIR}/benchmarks/${name}.csv"
}

if [[ "${RUN_BENCH}" == "true" ]]; then
  run_benchmark "baseline" "${CKPT_BASELINE}" "${SCRIPT_DIR}/config/train_fineweb10B_mini_t4.py"
  run_benchmark "hc" "${CKPT_HC}" "${SCRIPT_DIR}/config/train_fineweb10B_hc_mini_t4.py"
  run_benchmark "mhc" "${CKPT_MHC}" "${SCRIPT_DIR}/config/train_fineweb10B_mhc_mini_t4.py"
  python "${SCRIPT_DIR}/summarize_benchmarks.py" "${REPORT_DIR}/benchmarks"
else
  echo "benchmark skipped: RUN_BENCH=${RUN_BENCH}"
fi

python "${SCRIPT_DIR}/summarize_training_runs.py" \
  --runs \
  "baseline=${SCRIPT_DIR}/out-t4-mini-baseline" \
  "hc=${SCRIPT_DIR}/out-t4-mini-hc" \
  "mhc=${SCRIPT_DIR}/out-t4-mini-mhc" \
  --output-dir "${REPORT_DIR}"

if [[ "${RUN_PLOT}" == "true" ]]; then
  BENCH_ARGS=()
  if [[ -f "${REPORT_DIR}/benchmarks/summary.csv" ]]; then
    BENCH_ARGS=(--benchmark-summary "${REPORT_DIR}/benchmarks/summary.csv")
  fi
  python "${SCRIPT_DIR}/plot_training_comparison.py" \
    --runs \
    "baseline=${SCRIPT_DIR}/out-t4-mini-baseline" \
    "hc=${SCRIPT_DIR}/out-t4-mini-hc" \
    "mhc=${SCRIPT_DIR}/out-t4-mini-mhc" \
    "${BENCH_ARGS[@]}" \
    --output-dir "${REPORT_DIR}/figures"
fi

echo "report dir: ${REPORT_DIR}"
