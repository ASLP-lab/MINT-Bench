#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="${LANGUAGE:-en}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/instruction_text_pairs_timbre/${LANGUAGE}_timbre.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/instruction_text_pairs_timbre/${LANGUAGE}_timbre_subset_balanced_${NUM_SAMPLES}.json}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sample_timbre_subset_balanced.py" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  "$@"
