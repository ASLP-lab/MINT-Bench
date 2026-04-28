#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Timbre inference parameters
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/gen_outputs_timbre/gemini_reference_timbre_candidates_en}"
MODEL_NAME="${MODEL_NAME:-gemini-2.5-flash-preview-tts}"
API_URL="${API_URL:-https://generativelanguage.googleapis.com}"
SAMPLE_RATE="${SAMPLE_RATE:-24000}"
NUM_WORKERS="${NUM_WORKERS:-5}"
SEED="${SEED:-42}"
ERROR_LOG="${ERROR_LOG:-${OUTPUT_DIR}/errors.jsonl}"
TIMBRE_TAXONOMY_NODE="${TIMBRE_TAXONOMY_NODE:-aggregate/timbre}"
NUM_TIMBRE_CANDIDATES="${NUM_TIMBRE_CANDIDATES:-10}"
NUM_DEFAULT_CANDIDATES="${NUM_DEFAULT_CANDIDATES:-1}"
NODE_WHITELIST_JSON="${NODE_WHITELIST_JSON:-}"

# Show help if needed
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  cat << 'EOF'
Usage: bash gemini_tts_infer_from_pairs_timbre.sh [options]

Infer Gemini TTS audio from instruction-text pair JSON files with timbre diversity.
Supports multi-candidate generation for timbre-focused inference.

Environment Variables (set before running):
  INPUT_PATH                Input directory/file path (required)
  OUTPUT_DIR                Output directory path (required)
  MODEL_NAME                Gemini TTS model (default: gemini-2.5-flash-preview-tts)
  API_URL                   API endpoint (default: https://generativelanguage.googleapis.com)
  GOOGLE_API_KEY            API key (required if not --api_key)
  SAMPLE_RATE               Audio sample rate in Hz (default: 24000)
  NUM_WORKERS               Concurrent requests (default: 5)
  SEED                      Random seed for voice selection (default: 42)
  ERROR_LOG                 Error log file path
  TIMBRE_TAXONOMY_NODE      Node name for timbre subset (default: aggregate/timbre)
  NUM_TIMBRE_CANDIDATES     Candidates per timbre item (default: 10)
  NUM_DEFAULT_CANDIDATES    Candidates per non-timbre item (default: 1)
  NODE_WHITELIST_JSON       Optional: JSON list of taxonomy nodes to process
  PYTHON_BIN                Python executable (default: python)

Example:
  INPUT_PATH=./input \
  OUTPUT_DIR=./output \
  GOOGLE_API_KEY=your-key \
  bash gemini_tts_infer_from_pairs_timbre.sh
EOF
  exit 0
fi

# Validate required parameters
if [[ -z "${INPUT_PATH}" ]]; then
  echo "Error: INPUT_PATH must be set" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Error: OUTPUT_DIR must be set" >&2
  exit 1
fi

# Build command
cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gemini_tts_infer_from_pairs_timbre.py"
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
  --timbre_taxonomy_node "${TIMBRE_TAXONOMY_NODE}"
  --num_timbre_candidates "${NUM_TIMBRE_CANDIDATES}"
  --num_default_candidates "${NUM_DEFAULT_CANDIDATES}"
)

if [[ -n "${ERROR_LOG}" ]]; then
  cmd+=(--error_log "${ERROR_LOG}")
fi

if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
  cmd+=(--api_key "${GOOGLE_API_KEY}")
fi

if [[ -n "${NODE_WHITELIST_JSON}" ]]; then
  cmd+=(--node_whitelist_json "${NODE_WHITELIST_JSON}")
fi

cmd+=("$@")
"${cmd[@]}"