#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [step_epochs] [final_epochs] [parallel_cases] [parallel_devices] [python_cmd] [extra ghd_fitting args...]" >&2
  echo "Example: $0 /workspace/AneuG/ghd_fitting_config_prepared3_aneurysm_1op_quality_rim_ordered_v12_roundrobin.yaml 1500 15000 4 cuda:0,cuda:1,cuda:2,cuda:3 python --redo_do_points 1" >&2
  exit 1
fi

CONFIG_PATH="$1"
STEP_EPOCHS="${2:-1500}"
FINAL_EPOCHS="${3:-15000}"
PARALLEL_CASES="${4:-1}"
PARALLEL_DEVICES="${5:-cuda:0}"
PYTHON_CMD="${6:-python}"
EXTRA_ARGS=("${@:7}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIT_SCRIPT="${REPO_ROOT}/ghd_fitting.py"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${FIT_SCRIPT}" ]]; then
  echo "ghd_fitting.py not found at: ${FIT_SCRIPT}" >&2
  exit 1
fi

if ! [[ "${STEP_EPOCHS}" =~ ^[0-9]+$ && "${FINAL_EPOCHS}" =~ ^[0-9]+$ && "${PARALLEL_CASES}" =~ ^[0-9]+$ ]]; then
  echo "step_epochs, final_epochs and parallel_cases must be integers." >&2
  exit 1
fi

if (( STEP_EPOCHS <= 0 )); then
  echo "step_epochs must be > 0." >&2
  exit 1
fi

if (( FINAL_EPOCHS < STEP_EPOCHS )); then
  echo "final_epochs must be >= step_epochs." >&2
  exit 1
fi

echo "Round-robin epoch schedule"
echo "  config          = ${CONFIG_PATH}"
echo "  step_epochs     = ${STEP_EPOCHS}"
echo "  final_epochs    = ${FINAL_EPOCHS}"
echo "  parallel_cases  = ${PARALLEL_CASES}"
echo "  parallel_devices= ${PARALLEL_DEVICES}"
echo "  python_cmd      = ${PYTHON_CMD}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  echo "  extra_args      = ${EXTRA_ARGS[*]}"
fi

epoch_target="${STEP_EPOCHS}"
while (( epoch_target <= FINAL_EPOCHS )); do
  echo
  echo "=== Round-robin pass to epochs=${epoch_target} ==="
  "${PYTHON_CMD}" "${FIT_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --epochs "${epoch_target}" \
    --parallel_cases "${PARALLEL_CASES}" \
    --parallel_devices "${PARALLEL_DEVICES}" \
    "${EXTRA_ARGS[@]}"
  epoch_target=$(( epoch_target + STEP_EPOCHS ))
done

if (( (FINAL_EPOCHS % STEP_EPOCHS) != 0 )); then
  echo
  echo "=== Final pass to epochs=${FINAL_EPOCHS} ==="
  "${PYTHON_CMD}" "${FIT_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --epochs "${FINAL_EPOCHS}" \
    --parallel_cases "${PARALLEL_CASES}" \
    --parallel_devices "${PARALLEL_DEVICES}" \
    "${EXTRA_ARGS[@]}"
fi
