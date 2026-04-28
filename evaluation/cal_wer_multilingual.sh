#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Multilingual WER Evaluation Runner
# =============================================================================
# Computes WER (Word Error Rate) for TTS audio using ASR models.
#
# Usage:
#   bash cal_wer_multilingual.sh -m <model> -l <language> [options]
#
# Required Arguments:
#   -m, --model        Model name (e.g., 'gemini_reference')
#   -l, --language     Language code (e.g., 'en', 'zh')
#
# Optional Arguments:
#   -p, --preset       Preset name (default: mini)
#   -w, --workers      Number of parallel GPU workers (default: 8)
#   -h, --help         Show help message
# =============================================================================

show_help() {
  cat << 'EOF'
Usage: cal_wer_multilingual.sh -m <model> -l <language> [options]

Compute WER for a specific model and language combination using ASR.
Parallelizes evaluation across GPUs and aggregates results.

Required Arguments:
  -m, --model MODEL   Model name (e.g., 'gemini_reference', 'elevenlabs')

  -l, --language LANG Language code. Supported: zh, en, de, es, fr, it, ja, ko, pt, ru

Optional Arguments:
  -p, --preset PRESET Output preset name (default: mini)

  -w, --workers NUM   Number of parallel GPU workers (default: 8)

  -h, --help          Show this help message

Output:
  Results are saved to:
    evaluation/payloads_wer/<model>_<preset>_outputs_<lang>/wer.out

Examples:
  bash cal_wer_multilingual.sh -m gemini_reference -l en
  bash cal_wer_multilingual.sh -m mymodel -l zh -w 4
  bash cal_wer_multilingual.sh -m mymodel -l en -p mini
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

model_name=""
language=""
preset="mini"
num_workers=0  # 0 means use env var default

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -m|--model)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --model requires an argument" >&2
        exit 1
      fi
      model_name="$1"
      ;;
    -l|--language)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --language requires an argument" >&2
        exit 1
      fi
      language="$1"
      ;;
    -p|--preset)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --preset requires an argument" >&2
        exit 1
      fi
      preset="$1"
      ;;
    -w|--workers)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --workers requires an argument" >&2
        exit 1
      fi
      if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        echo "Error: --workers must be a positive integer" >&2
        exit 1
      fi
      num_workers="$1"
      ;;
    *)
      echo "Error: Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
  shift
done

# Validate required arguments
if [[ -z "${model_name}" ]]; then
  echo "Error: --model is required" >&2
  show_help >&2
  exit 1
fi

if [[ -z "${language}" ]]; then
  echo "Error: --language is required" >&2
  show_help >&2
  exit 1
fi

# Set number of workers
if [[ ${num_workers} -eq 0 ]]; then
  num_workers="${ARNOLD_WORKER_GPU:-8}"
fi

# =============================================================================
# Setup
# =============================================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED_TTSEVAL_DIR="${EVAL_DIR}/seed-tts-eval"
PAYLOADS_ROOT="${EVAL_DIR}/payloads_wer"
OUTPUT_ROOT="${EVAL_DIR}/results"

# Pin caches under seed-tts-eval/pretrained for reproducibility
PRETRAINED_ROOT="${SEED_TTSEVAL_DIR}/pretrained"
export HF_HOME="${PRETRAINED_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${PRETRAINED_ROOT}/hf/hub"
export TRANSFORMERS_CACHE="${PRETRAINED_ROOT}/hf/transformers"
export MODELSCOPE_CACHE="${PRETRAINED_ROOT}/modelscope"
export TORCH_HOME="${PRETRAINED_ROOT}/torch"
export XDG_CACHE_HOME="${PRETRAINED_ROOT}/xdg"

# =============================================================================
# Main Evaluation
# =============================================================================

run_dir="${PAYLOADS_ROOT}/${model_name}_${preset}_outputs_${language}"
out_dir="${OUTPUT_ROOT}/${model_name}_${preset}_outputs_${language}"
input_lst="${run_dir}/wer_input.lst"
output_wer="${out_dir}/wer.out"

if [[ ! -f "${input_lst}" ]]; then
  echo "Error: Missing input list: ${input_lst}" >&2
  exit 1
fi

mkdir -p "${out_dir}"

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "${tmp_dir}"; }
trap cleanup EXIT

num=$(wc -l < "${input_lst}")
if [[ "${num}" -eq 0 ]]; then
  echo "Error: Empty input list: ${input_lst}" >&2
  exit 1
fi

echo "Processing ${num} items with ${num_workers} workers..."

num_per_thread=$(( num / num_workers + 1 ))
split -l "${num_per_thread}" --additional-suffix=.lst -d "${input_lst}" "${tmp_dir}/thread-"

tmp_out_dir="${tmp_dir}/out"
mkdir -p "${tmp_out_dir}"

for rank in $(seq 0 $((num_workers - 1))); do
  thread_lst="${tmp_dir}/thread-0${rank}.lst"
  if [[ ! -f "${thread_lst}" ]]; then
    continue
  fi

  sub_out="${tmp_out_dir}/thread-0${rank}.wer.out"
  CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON_BIN}" "${SEED_TTSEVAL_DIR}/run_wer_multilingual.py" "${thread_lst}" "${sub_out}" "${language}" &
done
wait

merge_out="${tmp_out_dir}/merge.out"
cat "${tmp_out_dir}"/thread-0*.wer.out >> "${merge_out}"

"${PYTHON_BIN}" "${SEED_TTSEVAL_DIR}/average_wer.py" "${merge_out}" "${output_wer}"

# Extract WER value for display
wer_value=$(grep "^WER:" "${output_wer}" | head -1 || echo "N/A")
echo ""
echo "WER evaluation completed: ${wer_value}"
echo "Result: ${output_wer}"

cleanup
trap - EXIT
