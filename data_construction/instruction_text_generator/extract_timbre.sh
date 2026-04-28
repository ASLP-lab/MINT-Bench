#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESET="${PRESET:-mini}"
LANGUAGE="${LANGUAGE:-en}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/instruction_text_pairs/plan_to_instruction_text_${PRESET}_outputs_${LANGUAGE}}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/instruction_text_pairs/timbre/${LANGUAGE}_timbre.json}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/extract_timbre.py" \
  --input_dir "${INPUT_DIR}" \
  --output_path "${OUTPUT_PATH}" \
  "$@"
