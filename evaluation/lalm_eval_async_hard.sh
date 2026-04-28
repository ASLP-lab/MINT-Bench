#!/usr/bin/env bash
set -euo pipefail

# Available hard node basenames:
# - hard__composite__complex_comp__conflict
# - hard__composite__complex_comp__dynamic
# - hard__composite__complex_comp__layered
# - hard__composite__complex_comp__static
# - hard__composite__persona__ip_portrait
# - hard__composite__persona__non_ip_archetype
# - hard__composite__persona__scenario
# - hard__composite__persona__social_role
# - hard__composite__persona__stylized_persona

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_DIR}/logs}"
INPUT_PATH="${INPUT_PATH:-${1:-}}"
DIFFICULTY="hard"
API_KEY="${API_KEY:-${MINT_BENCH_API_KEY:-${GOOGLE_API_KEY:-}}}"
API_URL="${API_URL:-${MINT_BENCH_API_URL:-https://generativelanguage.googleapis.com}}"
NUM_WORKERS="${NUM_WORKERS:-5}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Usage: INPUT_PATH=path/to/node.jsonl bash lalm_eval_async_hard.sh" >&2
  echo "   or: INPUT_PATH=path/to/payload_dir bash lalm_eval_async_hard.sh" >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "API_KEY is not set. Export API_KEY, MINT_BENCH_API_KEY, or GOOGLE_API_KEY." >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/lalm_eval_async.py" \
  --input_path "${INPUT_PATH}" \
  --difficulty "${DIFFICULTY}" \
  --results_dir "${RESULTS_DIR}" \
  --logs_dir "${LOGS_DIR}" \
  --easy_prompt_file "${SCRIPT_DIR}/prompts/easy.txt" \
  --hard_prompt_file "${SCRIPT_DIR}/prompts/hard.txt" \
  --special_prompt_file "${SCRIPT_DIR}/prompts/special.txt" \
  --api_key "${API_KEY}" \
  --api_url "${API_URL}" \
  --num_workers "${NUM_WORKERS}"
