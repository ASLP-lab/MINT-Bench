#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="${PRESET:-mini}"
LANGUAGE="${LANGUAGE:-en}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/nodes_generator/nodes/label_plan_nodes_${PRESET}_${LANGUAGE}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/label_plans/label_plan_nodes_${PRESET}_outputs_${LANGUAGE}}"
ERROR_LOG="${ERROR_LOG:-${SCRIPT_DIR}/logs/label_plan_nodes_${PRESET}_${LANGUAGE}.log}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/label_plan_generator_multilingual.py" \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --error_log "${ERROR_LOG}" \
  "$@"
