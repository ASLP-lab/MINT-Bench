#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${STAGE:-all}"
PRESET="${PRESET:-mini}"
LANGUAGE="${LANGUAGE:-en}"
BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}}"
API_KEY="${API_KEY:-${MINT_BENCH_API_KEY:-${GOOGLE_API_KEY:-}}}"
API_URL="${API_URL:-${MINT_BENCH_API_URL:-https://generativelanguage.googleapis.com}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_CANDIDATES_PER_PLAN="${NUM_CANDIDATES_PER_PLAN:-3}"
SELECTION_SEED="${SELECTION_SEED:-}"
MAX_CHUNK_ATTEMPTS="${MAX_CHUNK_ATTEMPTS:-3}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_data_construction.py"
  --stage "${STAGE}"
  --preset "${PRESET}"
  --language "${LANGUAGE}"
  --base-dir "${BASE_DIR}"
  --max_concurrent "${MAX_CONCURRENT}"
  --batch_size "${BATCH_SIZE}"
  --num_candidates_per_plan "${NUM_CANDIDATES_PER_PLAN}"
  --max_chunk_attempts "${MAX_CHUNK_ATTEMPTS}"
)

if [[ -n "${API_KEY}" ]]; then
  cmd+=(--api_key "${API_KEY}")
fi

if [[ -n "${API_URL}" ]]; then
  cmd+=(--api_url "${API_URL}")
fi

if [[ -n "${SELECTION_SEED}" ]]; then
  cmd+=(--selection_seed "${SELECTION_SEED}")
fi

cmd+=("$@")

"${cmd[@]}"
