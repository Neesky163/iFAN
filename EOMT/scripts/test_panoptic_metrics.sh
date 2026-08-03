#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EOMT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${EOMT_ROOT}/checkpoints}"

CONFIG_PATH="${CONFIG_PATH:-${CHECKPOINT_ROOT}/panoptic/configs/coco_panoptic_vitl_640.yaml}"
CKPT_PATH="${CKPT_PATH:-${CHECKPOINT_ROOT}/panoptic/inference/coco_panoptic_vitl_640.ckpt}"
DATA_PATH="${DATA_PATH:-}"
GPU="${GPU:-0}"
LOG_ROOT="${LOG_ROOT:-${EOMT_ROOT}/metric_test_results}"

[[ -n "${DATA_PATH}" ]] || { echo "DATA_PATH is required (COCO or ADE20K validation root)" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 2; }
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 2; }
[[ -d "${DATA_PATH}" ]] || { echo "Dataset not found: ${DATA_PATH}" >&2; exit 2; }
[[ "${CKPT_PATH}" == */inference/*.ckpt ]] || { echo "Only inference/*.ckpt is allowed" >&2; exit 2; }

cd "${EOMT_ROOT}"
COMMAND=(python evaluate.py
  --config "${CONFIG_PATH}"
  --checkpoint "${CKPT_PATH}"
  --data "${DATA_PATH}"
  --output "${LOG_ROOT}"
  --device cuda:0
)
[[ -n "${LIMIT:-}" ]] && COMMAND+=(--limit "${LIMIT}")
[[ "${DRY_RUN:-0}" == "1" ]] && COMMAND+=(--dry-run)
exec env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib_eomt}" \
  "${COMMAND[@]}"
