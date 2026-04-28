#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data_construction/instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_mini_dryrun_outputs_en}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/gen_outputs/mimopro_reference_mini_outputs_en}"
MODEL_NAME="${MODEL_NAME:-mimo-v2.5-tts-voicedesign}"
BASE_URL="${BASE_URL:-https://api.xiaomimimo.com/v1}"
AUDIO_FORMAT="${AUDIO_FORMAT:-wav}"
ERROR_LOG="${ERROR_LOG:-${OUTPUT_DIR}/errors.jsonl}"
NODE_WHITELIST_JSON="${NODE_WHITELIST_JSON:-}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/mimo_pro_tts_voicedesign_infer.py"
  --input_path "${INPUT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --base_url "${BASE_URL}"
  --audio_format "${AUDIO_FORMAT}"
  --skip_existing
  --recursive
  --use_plan_id_as_name
  --error_log "${ERROR_LOG}"
)

if [[ -n "${MIMO_API_KEY:-}" ]]; then
  cmd+=(--api_key "${MIMO_API_KEY}")
fi

if [[ -n "${NODE_WHITELIST_JSON}" ]]; then
  cmd+=(--node_whitelist_json "${NODE_WHITELIST_JSON}")
fi

cmd+=("$@")
"${cmd[@]}"
