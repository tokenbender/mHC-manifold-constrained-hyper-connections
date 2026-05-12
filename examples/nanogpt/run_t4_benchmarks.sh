#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/benchmarks/t4-$(date +%Y%m%d-%H%M%S)}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-32}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERS="${NUM_ITERS:-20}"
COMPILE="${COMPILE:-false}"

mkdir -p "${OUT_DIR}"

run_variant() {
  local name="$1"
  local ckpt="$2"
  local config="$3"

  if [[ -z "${ckpt}" ]]; then
    echo "warning: ${name} checkpoint env var is empty; skipping" >&2
    return 0
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "warning: ${name} checkpoint not found at ${ckpt}; skipping" >&2
    return 0
  fi

  echo "benchmarking ${name}: ${ckpt}"
  python "${SCRIPT_DIR}/benchmark_inference.py" \
    --ckpt "${ckpt}" \
    --config "${config}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --prompt-len "${PROMPT_LEN}" \
    --gen-len "${GEN_LEN}" \
    --num-warmup "${NUM_WARMUP}" \
    --num-iters "${NUM_ITERS}" \
    --compile "${COMPILE}" \
    --output-json "${OUT_DIR}/${name}.json" \
    --output-csv "${OUT_DIR}/${name}.csv"
}

run_variant "baseline" "${CKPT_BASELINE:-}" "${SCRIPT_DIR}/config/train_fineweb10B.py"
run_variant "hc" "${CKPT_HC:-}" "${SCRIPT_DIR}/config/train_fineweb10B_hc.py"
run_variant "mhc" "${CKPT_MHC:-}" "${SCRIPT_DIR}/config/train_fineweb10B_mhc.py"
run_variant "vres" "${CKPT_VRES:-}" "${SCRIPT_DIR}/config/train_fineweb10B_vres.py"
run_variant "vres_mhc" "${CKPT_VRES_MHC:-}" "${SCRIPT_DIR}/config/train_fineweb10B_vres_mhc.py"

echo "benchmark outputs: ${OUT_DIR}"
