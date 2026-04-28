#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data_construction/instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_mini_dryrun_outputs_en}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/gen_outputs/gemini_reference_mini_outputs_en}"
MODEL_NAME="${MODEL_NAME:-gemini-2.5-flash-preview-tts}"
API_URL="${API_URL:-https://generativelanguage.googleapis.com}"
SAMPLE_RATE="${SAMPLE_RATE:-24000}"
NUM_WORKERS="${NUM_WORKERS:-5}"
SEED="${SEED:-42}"
ERROR_LOG="${ERROR_LOG:-${OUTPUT_DIR}/errors.jsonl}"
NODE_WHITELIST_JSON="${NODE_WHITELIST_JSON:-}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/geminiflash_tts_infer_from_pairs_concurrent.py"
  --input_path "${INPUT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --api_url "${API_URL}"
  --sample_rate "${SAMPLE_RATE}"
  --num_workers "${NUM_WORKERS}"
  --skip_existing
  --recursive
  --use_plan_id_as_name
  --seed "${SEED}"
  --error_log "${ERROR_LOG}"
)

if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
  cmd+=(--api_key "${GOOGLE_API_KEY}")
fi

if [[ -n "${NODE_WHITELIST_JSON}" ]]; then
  cmd+=(--node_whitelist_json "${NODE_WHITELIST_JSON}")
fi

cmd+=("$@")
"${cmd[@]}"
