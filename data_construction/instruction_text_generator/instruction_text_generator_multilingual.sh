#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="${PRESET:-mini}"
LANGUAGE="${LANGUAGE:-en}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/label_plan_generator/label_plans/label_plan_nodes_${PRESET}_outputs_${LANGUAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/instruction_text_pairs/plan_to_instruction_text_${PRESET}_outputs_${LANGUAGE}}"
ERROR_LOG="${ERROR_LOG:-${SCRIPT_DIR}/logs/plan_to_instruction_text_${PRESET}_outputs_${LANGUAGE}.log}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/instruction_text_generator_multilingual.py" \
  --input_path "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --error_log "${ERROR_LOG}" \
  "$@"
