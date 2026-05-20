#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_BENCH="${RUN_BENCH:-true}"
STRICT_CKPT="${STRICT_CKPT:-true}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_ITERS="${MAX_ITERS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS="${EVAL_ITERS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-1}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-32}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERS="${NUM_ITERS:-20}"
COMPILE="${COMPILE:-false}"
WANDB_LOG="${WANDB_LOG:-False}"
REPORT_DIR="${REPORT_DIR:-${REPO_DIR}/reports/t4-$(date +%Y%m%d-%H%M%S)}"

CKPT_BASELINE="${CKPT_BASELINE:-${SCRIPT_DIR}/out-t4-baseline/ckpt.pt}"
CKPT_HC="${CKPT_HC:-${SCRIPT_DIR}/out-t4-hc/ckpt.pt}"
CKPT_MHC="${CKPT_MHC:-${SCRIPT_DIR}/out-t4-mhc/ckpt.pt}"

mkdir -p "${REPORT_DIR}"

if [[ "${RUN_TRAIN}" == "true" ]]; then
  echo "training baseline/hc/mhc"
  MAX_ITERS="${MAX_ITERS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_ITERS="${EVAL_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  WANDB_LOG="${WANDB_LOG}" \
  bash "${SCRIPT_DIR}/run_t4_train_compare.sh"
else
  echo "training skipped: RUN_TRAIN=${RUN_TRAIN}"
fi

missing=0
for ckpt in "${CKPT_BASELINE}" "${CKPT_HC}" "${CKPT_MHC}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing checkpoint: ${ckpt}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 && "${STRICT_CKPT}" == "true" ]]; then
  echo "error: required checkpoints missing; set STRICT_CKPT=false to continue and skip missing benchmarks" >&2
  exit 2
elif [[ "${missing}" -ne 0 ]]; then
  echo "warning: missing checkpoints; benchmark helper will skip missing variants" >&2
fi

if [[ "${RUN_BENCH}" == "true" ]]; then
  echo "benchmarking checkpoints"
  CKPT_BASELINE="${CKPT_BASELINE}" \
  CKPT_HC="${CKPT_HC}" \
  CKPT_MHC="${CKPT_MHC}" \
  OUT_DIR="${REPORT_DIR}/benchmarks" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  BATCH_SIZE="${BENCH_BATCH_SIZE}" \
  PROMPT_LEN="${PROMPT_LEN}" \
  GEN_LEN="${GEN_LEN}" \
  NUM_WARMUP="${NUM_WARMUP}" \
  NUM_ITERS="${NUM_ITERS}" \
  COMPILE="${COMPILE}" \
  bash "${SCRIPT_DIR}/run_t4_benchmarks.sh"

  if compgen -G "${REPORT_DIR}/benchmarks/*.json" > /dev/null; then
    python "${SCRIPT_DIR}/summarize_benchmarks.py" "${REPORT_DIR}/benchmarks"
  else
    echo "warning: no benchmark JSON files found under ${REPORT_DIR}/benchmarks; benchmark summary skipped" >&2
  fi
else
  echo "benchmark skipped: RUN_BENCH=${RUN_BENCH}"
fi

python "${SCRIPT_DIR}/summarize_training_runs.py" \
  --runs \
  "baseline=${SCRIPT_DIR}/out-t4-baseline" \
  "hc=${SCRIPT_DIR}/out-t4-hc" \
  "mhc=${SCRIPT_DIR}/out-t4-mhc" \
  --output-dir "${REPORT_DIR}"

echo
echo "report files:"
echo "  ${REPORT_DIR}/training_summary.csv"
echo "  ${REPORT_DIR}/training_summary.md"
echo "  ${REPORT_DIR}/benchmarks/summary.csv"
echo "  ${REPORT_DIR}/benchmarks/summary.md"
