# MINT-Bench Visualization

This document focuses on the visualization app for inspecting evaluation results. For the project overview and full pipeline, start from the root [README.md](../README.md).

This directory contains a Streamlit app that visualizes MINT-Bench evaluation results: a per-bucket leaderboard, per-model bar charts, and a per-node taxonomy heatmap, all combined with WER pass-rate penalty.

## Quick Start

The simplest way is to use the wrapper:

```bash
# Use the default results dir at evaluation/results
bash visualization/run_visualization.sh

# Use a custom results dir
bash visualization/run_visualization.sh /path/to/results

# Pass extra Streamlit options (e.g. port)
bash visualization/run_visualization.sh /path/to/results -- --server.port 8520
```

You can also run the Streamlit app directly:

```bash
streamlit run visualization/app_view_evaluation_results.py
# or with a custom results dir
streamlit run visualization/app_view_evaluation_results.py -- /path/to/results
```

The app resolves the results path in the following order:
1. CLI argument
2. `MINT_BENCH_RESULTS_DIR` environment variable
3. Default repo path: `../evaluation/results` relative to the app file

## Expected Input Layout

The app expects MINT-Bench evaluation results as written by the evaluation pipeline:

```text
evaluation/results/
  <model>_<preset>_outputs_<lang>/
    easy__....jsonl
    hard__....jsonl
    special__....jsonl
    wer.out
  logs/                 # ignored
```

Each per-node JSONL contains LALM eval items with a `lalm_eval` field, a `taxonomy_node`, and `target_values`. Each `<run>/wer.out` is the per-utterance WER table written by `cal_wer_multilingual.sh` with a trailing `WER: x.xx%` summary line.

## What You Get

- **Structured leaderboard** with high-level buckets:
  Overall · Timbre · Style · Easy(tag/direct/simple-comp) · Hard(complex-comp · persona) · Special(abnormal · nonverbal)
- **Model comparison bar chart** for the chosen metric.
- **Per-bucket comparison bar chart** for the chosen metric.
- **Per-node heatmap** following the full taxonomy hierarchy (difficulty → family → branch → leaf), red→amber→green color gradient over raw 1.0–5.0.
- **Raw table + CSV download** of every metric per (model, bucket).
- **Debug expander** showing the discovered runs and frozen WER thresholds.

## Scoring

For each bucket / node:

```
pass_rate = fraction of items with WER <= language threshold
            (items missing WER are treated as passed)
pen_if    = raw_if_avg    * pass_rate
pen_total = raw_total_avg * pass_rate
```

WER thresholds are frozen (from the paper) and live at the top of the script as `WER_THRESHOLDS`.

## Sidebar Controls

- **Preset**: `mini` / `full` / `smoke`
- **Language**: language code present in the results dir
- **Models**: subset of models to plot
- **Threshold display**: shows the WER threshold used for the current language

## Metric Selector

The metric selector (radio button below the leaderboard) controls both the per-bucket bar chart and the heatmap:
- `pen_total_avg` — penalized total (default)
- `pen_if_avg`   — penalized instruction-following only
- `raw_total_avg` — raw total
- `raw_if_avg`   — raw instruction-following only

## Files

- [app_view_evaluation_results.py](app_view_evaluation_results.py) — main Streamlit app.
- [run_visualization.sh](run_visualization.sh) — convenience launcher with results-path handling.

## Troubleshooting

### `No evaluation runs found under: ...`
The app could not find any directory matching `<model>_<preset>_outputs_<lang>/` under the resolved results path. Check that:
- the results path you passed is correct
- the run directories follow the expected naming convention
- `logs/` is the only excluded top-level directory

### Empty heatmap / leaderboard
Make sure the run directories actually contain `*.jsonl` files with `lalm_eval` records and `wer.out`. The "Debug summary" expander at the bottom of the page lists the runs the app discovered for the current slice.

### `streamlit: command not found`
Activate the environment that has Streamlit installed, or set:

```bash
STREAMLIT_BIN=/path/to/streamlit bash visualization/run_visualization.sh
```

## Customizing

- WER thresholds — edit `WER_THRESHOLDS` at the top of the app.
- Bucket layout — edit `BUCKET_ORDER` / `BUCKET_LABEL`.
- Heatmap taxonomy — edit `NODE_HIERARCHY`.
