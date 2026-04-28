#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/nodes"

PRESET="${PRESET:-mini}"
LANGUAGE="${LANGUAGE:-en}"
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_DIR}/label_plan_nodes_${PRESET}_${LANGUAGE}.json}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_label_plan_nodes_multilingual.py" \
  --preset "${PRESET}" \
  --language "${LANGUAGE}" \
  --output_path "${OUTPUT_PATH}" \
  "$@"
