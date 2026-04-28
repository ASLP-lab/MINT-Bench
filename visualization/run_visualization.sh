#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MINT-Bench Visualization Launcher
# =============================================================================
# Launches the Streamlit leaderboard for MINT-Bench evaluation results.
#
# Usage:
#   bash run_visualization.sh [results_dir] [-- streamlit_args...]
#
# Examples:
#   # Use the repo default: ../evaluation/results
#   bash run_visualization.sh
#
#   # Custom results directory
#   bash run_visualization.sh /path/to/results
#
#   # Pass extra Streamlit options (e.g. port)
#   bash run_visualization.sh /path/to/results -- --server.port 8520
#
# Environment Variables:
#   PYTHON_BIN   Python interpreter (default: python)
#   STREAMLIT_BIN  Streamlit executable (default: streamlit)
# =============================================================================

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,22p' "$0"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${SCRIPT_DIR}/app_view_evaluation_results.py"

if [[ ! -f "${APP}" ]]; then
  echo "Error: app not found: ${APP}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
STREAMLIT_BIN="${STREAMLIT_BIN:-streamlit}"

results_dir=""
streamlit_args=()
saw_split=false
for arg in "$@"; do
  if [[ "${saw_split}" == true ]]; then
    streamlit_args+=("${arg}")
    continue
  fi
  if [[ "${arg}" == "--" ]]; then
    saw_split=true
    continue
  fi
  if [[ -z "${results_dir}" ]]; then
    results_dir="${arg}"
  else
    streamlit_args+=("${arg}")
  fi
done

if [[ -z "${results_dir}" ]]; then
  results_dir="$(cd "${SCRIPT_DIR}/../evaluation/results" 2>/dev/null && pwd || true)"
  if [[ -z "${results_dir}" ]]; then
    echo "Error: default results dir not found at ${SCRIPT_DIR}/../evaluation/results" >&2
    echo "Pass an explicit path: bash run_visualization.sh /path/to/results" >&2
    exit 1
  fi
fi

if [[ ! -d "${results_dir}" ]]; then
  echo "Error: results dir does not exist: ${results_dir}" >&2
  exit 1
fi

echo "Launching MINT-Bench visualization"
echo "  app:         ${APP}"
echo "  results_dir: ${results_dir}"
if [[ ${#streamlit_args[@]} -gt 0 ]]; then
  echo "  extra args:  ${streamlit_args[*]}"
fi

exec "${STREAMLIT_BIN}" run "${APP}" "${streamlit_args[@]}" -- "${results_dir}"