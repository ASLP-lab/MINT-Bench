#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Timbre Diversity Evaluation Runner
# =============================================================================
# Computes WavLM-large speaker-similarity metric over multi-candidate timbre
# generations.  Reads the summary JSON produced by
# prepare_timbre_diversity_payload.py and writes a per-run score file.
#
# Usage:
#   bash cal_timbre_diversity.sh -i <summary_json> [options]
# =============================================================================

show_help() {
  cat << 'EOF'
Usage: cal_timbre_diversity.sh -i <summary_json> [options]

Compute timbre diversity (mean pairwise speaker similarity across valid
candidate WAVs) for one summary JSON produced by
prepare_timbre_diversity_payload.py.

Required Arguments:
  -i, --input_json PATH   Summary JSON path. Usually:
                            evaluation/payloads_timbre_diversity/<run>.summary.json

Optional Arguments:
  -o, --output_dir DIR    Output directory for score files
                            (default: evaluation/results_timbre_diversity)
  -d, --device DEVICE     Torch device (default: cuda:0)
  -h, --help              Show this help message and exit.

Environment Variables:
  PYTHON_BIN              Python interpreter (default: python)

Examples:
  bash cal_timbre_diversity.sh \
    -i evaluation/payloads_timbre_diversity/gemini_reference_timbre_candidates_en.summary.json

  bash cal_timbre_diversity.sh \
    -i path/to/summary.json -d cuda:7
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

input_json=""
output_dir=""
device="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -i|--input_json)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --input_json requires an argument" >&2
        exit 1
      fi
      input_json="$1"
      ;;
    -o|--output_dir)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --output_dir requires an argument" >&2
        exit 1
      fi
      output_dir="$1"
      ;;
    -d|--device)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --device requires an argument" >&2
        exit 1
      fi
      device="$1"
      ;;
    *)
      echo "Error: Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${input_json}" ]]; then
  echo "Error: --input_json is required" >&2
  show_help >&2
  exit 1
fi

# =============================================================================
# Setup
# =============================================================================

PYTHON_BIN="${PYTHON_BIN:-python}"

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED_TTSEVAL_DIR="${EVAL_DIR}/seed-tts-eval"

if [[ -z "${output_dir}" ]]; then
  output_dir="${EVAL_DIR}/results_timbre_diversity"
fi

# Pin caches under seed-tts-eval/pretrained for reproducibility
PRETRAINED_ROOT="${SEED_TTSEVAL_DIR}/pretrained"
export HF_HOME="${PRETRAINED_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${PRETRAINED_ROOT}/hf/hub"
export TRANSFORMERS_CACHE="${PRETRAINED_ROOT}/hf/transformers"
export TORCH_HOME="${PRETRAINED_ROOT}/torch"
export XDG_CACHE_HOME="${PRETRAINED_ROOT}/xdg"

# =============================================================================
# Run
# =============================================================================

mkdir -p "${output_dir}"

echo "Timbre Diversity Configuration:"
echo "  Input:    ${input_json}"
echo "  Output:   ${output_dir}"
echo "  Device:   ${device}"
echo ""

"${PYTHON_BIN}" "${SEED_TTSEVAL_DIR}/cal_timbre_diversity.py" \
  --input_json "${input_json}" \
  --output_dir "${output_dir}" \
  --device "${device}"