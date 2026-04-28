# Instruction-Following Evaluation

This document focuses on the instruction-following stage of MINT-Bench evaluation. For the overall evaluation flow, see [EVALUATION.md](EVALUATION.md). For the project homepage overview, see [README.md](../README.md).

## Overview

The instruction-following pipeline has two stages:

1. **Prepare payload JSONL files** — Convert instruction-text pairs plus generated audio outputs into evaluator-ready payloads.
2. **Run LALM evaluation** — Score each payload item with difficulty-specific prompts and save structured JSONL results.

There are two payload preparation flows:
- **Standard benchmark evaluation** for easy / hard / special nodes
- **Timbre candidate evaluation** for the advanced timbre-diversity pipeline

## Quick Start

### Standard benchmark evaluation

#### Step 1: Prepare payload JSONL files

By default, MINT-Bench uses the official instruction-text pair set:

```text
data_construction/instruction_text_generator/instruction_text_pairs_official/
```

Pass it explicitly to `--pairs_path`. To evaluate a DIY benchmark you built via the data construction pipeline, pass your custom pair root instead.

```bash
# Official test set
python evaluation/prepare_instruction_following_payload.py \
  --pairs_path data_construction/instruction_text_generator/instruction_text_pairs_official \
  --generation_path gen_outputs/gemini_reference_mini_outputs_en \
  --skip_missing_audio

# DIY benchmark
python evaluation/prepare_instruction_following_payload.py \
  --pairs_path data_construction/instruction_text_generator/instruction_text_pairs/my_diy_pairs \
  --generation_path gen_outputs/mymodel_mini_outputs_en \
  --skip_missing_audio
```

You can also pass a parent directory containing multiple generation runs:

```bash
python evaluation/prepare_instruction_following_payload.py \
  --pairs_path data_construction/instruction_text_generator/instruction_text_pairs_official \
  --generation_path gen_outputs \
  --skip_missing_audio
```

Default output:

```text
evaluation/payloads_instruction_following/
```

Each generation directory gets its own subdirectory, and each pair JSON becomes one payload JSONL.

#### Step 2: Run LALM evaluation

Run each difficulty separately:

```bash
INPUT_PATH=evaluation/payloads_instruction_following/gemini_reference_mini_outputs_en/easy__composite__simple_comp__static.jsonl \
API_KEY=your-key \
bash evaluation/lalm_eval_async_easy.sh

INPUT_PATH=evaluation/payloads_instruction_following/gemini_reference_mini_outputs_en \
API_KEY=your-key \
bash evaluation/lalm_eval_async_hard.sh

INPUT_PATH=evaluation/payloads_instruction_following/gemini_reference_mini_outputs_en \
API_KEY=your-key \
bash evaluation/lalm_eval_async_special.sh
```

You may pass either:
- one payload JSONL file, or
- one payload directory containing many JSONL files

Results are written to:

```text
evaluation/results/
```

### Timbre candidate evaluation

This is used by the advanced timbre-diversity pipeline.

#### Step 1: Prepare timbre instruction-following payload

```bash
python evaluation/prepare_instruction_following_payload_timbre.py \
  -i data_construction/instruction_text_generator/instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json \
  -w gen_outputs_timbre/gemini_reference_timbre_candidates_en
```

Default output:

```text
evaluation/payloads_instruction_following_timbre/
```

#### Step 2: Run timbre LALM evaluation

```bash
INPUT_PATH=evaluation/payloads_instruction_following_timbre/gemini_reference_timbre_candidates_en.jsonl \
API_KEY=your-key \
bash evaluation/lalm_eval_async_timbre.sh
```

Results are written to:

```text
evaluation/results_timbre/
```

## Output Layout

### Standard instruction-following payloads

```text
evaluation/payloads_instruction_following/
  <generation_run>/
    easy__....jsonl
    hard__....jsonl
    special__....jsonl
  summary.json
```

### Standard evaluation results

```text
evaluation/results/
  <generation_run>/
    easy__....jsonl
    hard__....jsonl
    special__....jsonl
  logs/
    <generation_run>__<payload_name>.log
```

### Timbre payloads and results

```text
evaluation/payloads_instruction_following_timbre/
  <model>_timbre_candidates_<lang>.jsonl

evaluation/results_timbre/
  <model>_timbre_candidates_<lang>.jsonl
  logs/
    <model>_timbre_candidates_<lang>.log
```

## Payload Format

Each payload JSONL line contains:

```json
{
  "taxonomy_node": "easy/timbre/direct/atomic/age",
  "plan_id": "PLAN__en__easy_timbre_direct_atomic_age__0001",
  "language": "en",
  "instruction": "Speak as if you are a young child.",
  "text": "Can I have a glass of juice, please?",
  "target_values": {"age": "child"},
  "wav_path": "../gen_outputs/.../PLAN__....wav"
}
```

The evaluator appends a `lalm_eval` field with:
- `instruction_following_level`
- `instruction_following_label`
- `naturalness_bonus`
- `expressiveness_bonus`
- `final_score`
- evidence fields and brief reasoning

## Prompt Routing

The evaluator picks the prompt template based on `taxonomy_node` prefix:
- `easy/...` → `prompts/easy.txt`
- `hard/...` → `prompts/hard.txt`
- `special/...` → `prompts/special.txt`
- timbre payloads also use `easy/...` taxonomy and therefore route to `prompts/easy.txt`

## Configuration

### Environment variables

- `API_KEY` / `MINT_BENCH_API_KEY` / `GOOGLE_API_KEY` — API key for the LALM evaluator
- `API_URL` / `MINT_BENCH_API_URL` — API base URL (default: `https://generativelanguage.googleapis.com`)
- `PYTHON_BIN` — Python interpreter (default: `python`)
- `NUM_WORKERS` — maximum concurrent evaluator requests

### Common options for payload preparation

`prepare_instruction_following_payload.py` supports:
- `--pairs_path`
- `--generation_path`
- `--output_dir`
- `--model_name`
- `--audio_ext`
- `--node_whitelist_json`
- `--skip_missing_audio`
- `--summary_path`

`prepare_instruction_following_payload_timbre.py` supports:
- `-i/--input_json`
- `-w/--wav_dir`
- `-o/--output_dir`
- `-m/--model`
- `-n/--num_candidates`

### Relative wav_path resolution

Prepared payload JSONL files store `wav_path` relative to the `evaluation/` directory.

That means:
- if you use the provided wrappers and `run_evaluation.sh`, paths are resolved correctly
- `run_evaluation.sh` accepts relative paths from either your current shell working directory or the `mint-bench/` project root, then switches into `evaluation/` before launching the evaluator
- if you call the evaluator manually from an arbitrary working directory, relative audio paths may fail to resolve

The recommended approach is to run the wrappers from the repo, or let `run_evaluation.sh` switch into `evaluation/` before launching the evaluator.

## Resume Behavior

The evaluator supports resume / rerun behavior:
- valid existing results in the output JSONL are reused
- missing or invalid items are rerun
- output order remains aligned with the input payload order

This makes it safe to rerun the same command after partial failures.

## Troubleshooting

### No payload JSONL files are generated

Possible causes:
- audio files are missing and `--skip_missing_audio` filtered everything
- generation directory name does not match `<model>_<preset>_outputs_<lang>`
- the pair directory does not match the preset/language

### Difficulty mismatch error

Standard payload file names are validated against the wrapper difficulty:
- `lalm_eval_async_easy.sh` expects `easy__...jsonl`
- `lalm_eval_async_hard.sh` expects `hard__...jsonl`
- `lalm_eval_async_special.sh` expects `special__...jsonl`

Timbre payloads use a dedicated `timbre` difficulty mode and do not need an `easy__` filename prefix.

### API key error

Set one of:

```bash
export API_KEY=...
# or
export MINT_BENCH_API_KEY=...
# or
export GOOGLE_API_KEY=...
```

### Relative wav_path resolution fails

Payloads are written with paths relative to `evaluation/`. Run the wrappers from the repo using the provided scripts rather than calling `lalm_eval_async.py` with ad-hoc working directories.

## Related Files

- [prepare_instruction_following_payload.py](prepare_instruction_following_payload.py)
- [prepare_instruction_following_payload_timbre.py](prepare_instruction_following_payload_timbre.py)
- [lalm_eval_async.py](lalm_eval_async.py)
- [lalm_eval_async_easy.sh](lalm_eval_async_easy.sh)
- [lalm_eval_async_hard.sh](lalm_eval_async_hard.sh)
- [lalm_eval_async_special.sh](lalm_eval_async_special.sh)
- [lalm_eval_async_timbre.sh](lalm_eval_async_timbre.sh)
- [prompts/](prompts/)
