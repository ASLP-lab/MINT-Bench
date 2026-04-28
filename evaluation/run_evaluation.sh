#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat << 'EOF'
Usage: run_evaluation.sh --model MODEL --language LANG --generation_path PATH [options]

Run the MINT-Bench evaluation pipeline for one model and one language.

By default this runs the first two stages only:
  1) WER evaluation
  2) Instruction-following evaluation

Timbre diversity is treated as an advanced stage and is only run when
--run_timbre_advanced is specified.

The instruction-text pair set defaults to the official MINT-Bench test set:
  data_construction/instruction_text_generator/instruction_text_pairs_official
You can override --pairs_path to evaluate a DIY benchmark you built via
the data construction pipeline.

Important:
  This script changes the working directory to evaluation/ before launching
  the evaluation stages. Payload JSONL files store wav_path relative to
  evaluation/, so running from that directory keeps relative audio paths
  resolvable.

Required Arguments:
  -m, --model MODEL           Model name (e.g. gemini_reference)
  -l, --language LANG         Language code (e.g. en, zh, de)
      --generation_path PATH  Generation outputs directory for this run
                              (typically gen_outputs/<model>_<preset>_outputs_<lang>).
                              You may also pass a parent directory.

Optional Arguments:
  -p, --preset PRESET         Preset for standard generation outputs (default: mini)
  -s, --stage STAGE           Standard stage: wer | instruction | all (default: all)
      --run_timbre_advanced   Also run the advanced timbre-diversity pipeline
      --pairs_path PATH       Instruction-text pairs root or run directory
                              (default: data_construction/instruction_text_generator/instruction_text_pairs_official)
      --timbre_input_json PATH
                              Timbre subset JSON path. Required when
                              --run_timbre_advanced is set.
                              (e.g. instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json)
      --timbre_wav_dir PATH   Timbre candidate wav dir. Required when
                              --run_timbre_advanced is set.
                              (e.g. gen_outputs_timbre/<model>_timbre_candidates_<lang>)
      --api_key KEY           API key passed to LALM evaluation wrappers
      --api_url URL           API base URL for LALM evaluation
      --num_workers NUM       Worker count for LALM evaluation wrappers
      --wer_workers NUM       Worker count for WER evaluation wrapper
      --device DEVICE         Torch device for timbre diversity (default: cuda:0)
  -h, --help                  Show this help message and exit.

Environment Variables:
  API_KEY / MINT_BENCH_API_KEY / GOOGLE_API_KEY  API key for instruction-following evaluation
  PYTHON_BIN                                        Python interpreter (default: python)

Examples:
  # Default: WER + instruction-following on the official benchmark
  bash evaluation/run_evaluation.sh \
    -m gemini_reference -l en \
    --generation_path gen_outputs/gemini_reference_mini_outputs_en

  # Only WER
  bash evaluation/run_evaluation.sh \
    -m gemini_reference -l en -s wer \
    --generation_path gen_outputs/gemini_reference_mini_outputs_en

  # Evaluate a DIY benchmark you built with data_construction
  bash evaluation/run_evaluation.sh \
    -m mymodel -l en \
    --pairs_path data_construction/instruction_text_generator/instruction_text_pairs/my_diy_pairs \
    --generation_path gen_outputs/mymodel_mini_outputs_en

  # Standard flow + advanced timbre diversity
  bash evaluation/run_evaluation.sh \
    -m gemini_reference -l en \
    --generation_path gen_outputs/gemini_reference_mini_outputs_en \
    --run_timbre_advanced \
    --timbre_input_json data_construction/instruction_text_generator/instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json \
    --timbre_wav_dir gen_outputs_timbre/gemini_reference_timbre_candidates_en
EOF
}

model=""
language=""
preset="mini"
stage="all"
run_timbre_advanced=false
pairs_path=""
generation_path=""
timbre_input_json=""
timbre_wav_dir=""
api_key="${API_KEY:-${MINT_BENCH_API_KEY:-${GOOGLE_API_KEY:-}}}"
api_url="${API_URL:-${MINT_BENCH_API_URL:-https://generativelanguage.googleapis.com}}"
num_workers="${NUM_WORKERS:-5}"
wer_workers="${ARNOLD_WORKER_GPU:-8}"
device="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -m|--model)
      shift; model="${1:-}" ;;
    -l|--language)
      shift; language="${1:-}" ;;
    -p|--preset)
      shift; preset="${1:-}" ;;
    -s|--stage)
      shift; stage="${1:-}" ;;
    --run_timbre_advanced)
      run_timbre_advanced=true ;;
    --pairs_path)
      shift; pairs_path="${1:-}" ;;
    --generation_path)
      shift; generation_path="${1:-}" ;;
    --timbre_input_json)
      shift; timbre_input_json="${1:-}" ;;
    --timbre_wav_dir)
      shift; timbre_wav_dir="${1:-}" ;;
    --api_key)
      shift; api_key="${1:-}" ;;
    --api_url)
      shift; api_url="${1:-}" ;;
    --num_workers)
      shift; num_workers="${1:-}" ;;
    --wer_workers)
      shift; wer_workers="${1:-}" ;;
    --device)
      shift; device="${1:-}" ;;
    *)
      echo "Error: Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${model}" ]]; then
  echo "Error: --model is required" >&2
  show_help >&2
  exit 1
fi

if [[ -z "${language}" ]]; then
  echo "Error: --language is required" >&2
  show_help >&2
  exit 1
fi

if [[ -z "${generation_path}" ]]; then
  echo "Error: --generation_path is required (e.g. gen_outputs/<model>_<preset>_outputs_<lang>)" >&2
  show_help >&2
  exit 1
fi

if [[ "${stage}" != "wer" && "${stage}" != "instruction" && "${stage}" != "all" ]]; then
  echo "Error: --stage must be one of: wer, instruction, all" >&2
  exit 1
fi

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_path() {
  local input_path="$1"
  if [[ -z "${input_path}" ]]; then
    return 0
  fi

  # Absolute path: keep as-is.
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
    return 0
  fi

  # 1) Try relative to the current shell working directory.
  if [[ -e "${input_path}" ]]; then
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${input_path}"
    return 0
  fi

  # 2) Try relative to the mint-bench project root.
  if [[ -e "${PROJECT_ROOT}/${input_path}" ]]; then
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${PROJECT_ROOT}/${input_path}"
    return 0
  fi

  # 3) Fall back to current-dir absolute resolution for clearer downstream errors.
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${input_path}"
}

# Default to the official MINT-Bench instruction-text pair set.
if [[ -z "${pairs_path}" ]]; then
  pairs_path="${PROJECT_ROOT}/data_construction/instruction_text_generator/instruction_text_pairs_official"
fi

# Normalize incoming paths before switching cwd to evaluation/.
pairs_path="$(resolve_path "${pairs_path}")"
generation_path="$(resolve_path "${generation_path}")"
if [[ -n "${timbre_input_json}" ]]; then
  timbre_input_json="$(resolve_path "${timbre_input_json}")"
fi
if [[ -n "${timbre_wav_dir}" ]]; then
  timbre_wav_dir="$(resolve_path "${timbre_wav_dir}")"
fi

# If the user passed a parent gen dir instead of a per-run dir, expand to the
# expected per-run subdirectory.
generation_dir="${generation_path}"
expected_run="${model}_${preset}_outputs_${language}"
if [[ -d "${generation_dir}/${expected_run}" ]]; then
  generation_dir="${generation_dir}/${expected_run}"
fi

generation_dir="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${generation_dir}")"

if [[ ! -d "${generation_dir}" ]]; then
  echo "Error: generation_path does not exist: ${generation_dir}" >&2
  exit 1
fi

# Run evaluation stages from evaluation/ so relative wav_path values stored in
# payloads resolve correctly.
cd "${EVAL_DIR}"

wer_payload_dir="${EVAL_DIR}/payloads_wer"
instruction_payload_dir="${EVAL_DIR}/payloads_instruction_following"
timbre_instruction_payload_dir="${EVAL_DIR}/payloads_instruction_following_timbre"
timbre_payload_dir="${EVAL_DIR}/payloads_timbre_diversity"

run_wer_stage() {
  echo "=== [WER] Prepare payload ==="
  "${PYTHON_BIN}" "${EVAL_DIR}/prepare_wer_payload.py" \
    -m "${model}" \
    -l "${language}" \
    -p "${preset}" \
    --pairs_path "${pairs_path}" \
    --generation_path "${generation_dir}" \
    -o "${wer_payload_dir}"

  echo "=== [WER] Run evaluation ==="
  ARNOLD_WORKER_GPU="${wer_workers}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/cal_wer_multilingual.sh" \
    -m "${model}" \
    -l "${language}" \
    -p "${preset}" \
    -w "${wer_workers}"
}

run_instruction_stage() {
  if [[ -z "${api_key}" ]]; then
    echo "Error: API key is required for instruction-following evaluation." >&2
    exit 1
  fi

  echo "=== [Instruction] Prepare payload ==="
  "${PYTHON_BIN}" "${EVAL_DIR}/prepare_instruction_following_payload.py" \
    --pairs_path "${pairs_path}" \
    --generation_path "${generation_dir}" \
    --output_dir "${instruction_payload_dir}" \
    --skip_missing_audio

  payload_run_dir="${instruction_payload_dir}/${model}_${preset}_outputs_${language}"

  echo "=== [Instruction] Run easy ==="
  INPUT_PATH="${payload_run_dir}" \
  RESULTS_DIR="${EVAL_DIR}/results" \
  API_KEY="${api_key}" \
  API_URL="${api_url}" \
  NUM_WORKERS="${num_workers}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/lalm_eval_async_easy.sh"

  echo "=== [Instruction] Run hard ==="
  INPUT_PATH="${payload_run_dir}" \
  RESULTS_DIR="${EVAL_DIR}/results" \
  API_KEY="${api_key}" \
  API_URL="${api_url}" \
  NUM_WORKERS="${num_workers}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/lalm_eval_async_hard.sh"

  echo "=== [Instruction] Run special ==="
  INPUT_PATH="${payload_run_dir}" \
  RESULTS_DIR="${EVAL_DIR}/results" \
  API_KEY="${api_key}" \
  API_URL="${api_url}" \
  NUM_WORKERS="${num_workers}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/lalm_eval_async_special.sh"
}

run_timbre_advanced_stage() {
  if [[ -z "${api_key}" ]]; then
    echo "Error: API key is required for timbre advanced evaluation." >&2
    exit 1
  fi
  if [[ -z "${timbre_input_json}" || -z "${timbre_wav_dir}" ]]; then
    echo "Error: --timbre_input_json and --timbre_wav_dir are required when --run_timbre_advanced is set." >&2
    exit 1
  fi
  if [[ ! -f "${timbre_input_json}" ]]; then
    echo "Error: --timbre_input_json file does not exist: ${timbre_input_json}" >&2
    exit 1
  fi
  if [[ ! -d "${timbre_wav_dir}" ]]; then
    echo "Error: --timbre_wav_dir directory does not exist: ${timbre_wav_dir}" >&2
    exit 1
  fi

  echo "=== [Timbre] Prepare instruction payload ==="
  "${PYTHON_BIN}" "${EVAL_DIR}/prepare_instruction_following_payload_timbre.py" \
    -i "${timbre_input_json}" \
    -w "${timbre_wav_dir}" \
    -o "${timbre_instruction_payload_dir}" \
    -m "${model}"

  timbre_instruction_jsonl="${timbre_instruction_payload_dir}/$(basename "${timbre_wav_dir}").jsonl"

  echo "=== [Timbre] Run instruction-following filter ==="
  INPUT_PATH="${timbre_instruction_jsonl}" \
  RESULTS_DIR="${EVAL_DIR}/results_timbre" \
  API_KEY="${api_key}" \
  API_URL="${api_url}" \
  NUM_WORKERS="${num_workers}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/lalm_eval_async_timbre.sh"

  timbre_result_jsonl="${EVAL_DIR}/results_timbre/$(basename "${timbre_wav_dir}").jsonl"

  echo "=== [Timbre] Prepare diversity summary ==="
  "${PYTHON_BIN}" "${EVAL_DIR}/prepare_timbre_diversity_payload.py" \
    -i "${timbre_result_jsonl}" \
    -o "${timbre_payload_dir}"

  timbre_summary_json="${timbre_payload_dir}/$(basename "${timbre_wav_dir}").summary.json"

  echo "=== [Timbre] Run diversity metric ==="
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_DIR}/cal_timbre_diversity.sh" \
    -i "${timbre_summary_json}" \
    -d "${device}"
}

echo "Evaluation Configuration:"
echo "  Model:             ${model}"
echo "  Language:          ${language}"
echo "  Preset:            ${preset}"
echo "  Stage:             ${stage}"
echo "  Run timbre adv.:   ${run_timbre_advanced}"
echo "  Pairs path:        ${pairs_path}"
echo "  Generation path:   ${generation_dir}"
if [[ "${run_timbre_advanced}" == true ]]; then
  echo "  Timbre input json: ${timbre_input_json}"
  echo "  Timbre wav dir:    ${timbre_wav_dir}"
fi
echo ""

case "${stage}" in
  wer)
    run_wer_stage
    ;;
  instruction)
    run_instruction_stage
    ;;
  all)
    run_wer_stage
    run_instruction_stage
    ;;
esac

if [[ "${run_timbre_advanced}" == true ]]; then
  run_timbre_advanced_stage
fi

echo ""
echo "Evaluation pipeline completed."