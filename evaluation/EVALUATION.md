# Evaluation Pipeline Overview

This document focuses on the evaluation stage. For the project overview and homepage-level quickstart, start from the root [README.md](../README.md).

The evaluation is organized into three stages:

1. **WER evaluation** — automatic speech recognition based intelligibility / transcription error
2. **Instruction-following evaluation** — LALM-based judgment of whether generated speech follows the requested instruction
3. **Timbre diversity evaluation** — advanced multi-candidate speaker-similarity analysis for timbre diversity

By default, the recommended evaluation flow runs only the first two stages. Timbre diversity is more expensive and depends on additional candidate-generation and filtering steps, so it is treated as an advanced stage.

## Prerequisites

Before running any evaluation stage, you must first generate the corresponding audio outputs.

That means:
- run the generation pipeline described in the generation documentation
- ensure the generated audio layout follows the expected naming convention
- for standard evaluation, audio basenames should match `plan_id`
- for timbre diversity, multi-candidate audio should follow the `__candXX` naming pattern
- payload JSONL files in evaluation store `wav_path` relative to `evaluation/`, so wrapper scripts and `run_evaluation.sh` intentionally run from that directory when executing evaluator stages
- `run_evaluation.sh` accepts relative paths either from the current shell working directory or from the `mint-bench/` project root, so it works whether you launch it from `mint-bench/` or from `mint-bench/evaluation/`

Typical generation outputs used by evaluation are:

```text
gen_outputs/<model>_<preset>_outputs_<lang>/
gen_outputs_timbre/<model>_timbre_candidates_<lang>/
```

Please follow the instructions in the generation directory first, then come back to the evaluation stage once the audio is ready.

## Test Set Selection

The standard evaluation pipeline uses an instruction-text pair set + a generated audio directory. By default, MINT-Bench evaluates against the **official** test set:

```text
data_construction/instruction_text_generator/instruction_text_pairs_official/
```

If you build your own DIY benchmark via the data construction pipeline, you can evaluate it by passing `--pairs_path` to `run_evaluation.sh`.

Generated audio is **not** assumed at a fixed location: you must pass `--generation_path` explicitly. The script accepts either:
- a specific run directory: `gen_outputs/<model>_<preset>_outputs_<lang>/`
- the parent directory: `gen_outputs/`  (auto-resolved to the per-run subdir)

## Recommended Workflow

### Default workflow: WER + instruction following on the official test set

```bash
bash evaluation/run_evaluation.sh \
  --model gemini_reference \
  --language en \
  --preset mini \
  --generation_path gen_outputs/gemini_reference_mini_outputs_en
```

This will:
- prepare WER payloads
- run WER evaluation
- prepare instruction-following payloads
- run easy / hard / special instruction-following evaluation

### Evaluate a DIY benchmark you built with data construction

```bash
bash evaluation/run_evaluation.sh \
  --model mymodel \
  --language en \
  --preset mini \
  --pairs_path data_construction/instruction_text_generator/instruction_text_pairs/my_diy_pairs \
  --generation_path gen_outputs/mymodel_mini_outputs_en
```

### Advanced workflow: add timbre diversity

```bash
bash evaluation/run_evaluation.sh \
  --model gemini_reference \
  --language en \
  --preset mini \
  --generation_path gen_outputs/gemini_reference_mini_outputs_en \
  --run_timbre_advanced \
  --timbre_input_json data_construction/instruction_text_generator/instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json \
  --timbre_wav_dir gen_outputs_timbre/gemini_reference_timbre_candidates_en
```

The timbre stage requires both `--timbre_input_json` (the timbre subset JSON) and `--timbre_wav_dir` (the multi-candidate audio directory). Timbre subsets default to the official set under `instruction_text_pairs_timbre/`, but you can pass any DIY timbre subset JSON.

This adds:
- timbre candidate payload preparation
- instruction-following filtering on those candidates
- timbre-diversity summary JSON
- final timbre-diversity scoring

## Run a Single Stage

### WER only

```bash
bash evaluation/run_evaluation.sh \
  --model gemini_reference \
  --language en \
  --stage wer \
  --generation_path gen_outputs/gemini_reference_mini_outputs_en
```

### Instruction-following only

```bash
bash evaluation/run_evaluation.sh \
  --model gemini_reference \
  --language en \
  --stage instruction \
  --generation_path gen_outputs/gemini_reference_mini_outputs_en
```

## Key Directories

### Payloads

- [payloads_wer/](payloads_wer/)
- [payloads_instruction_following/](payloads_instruction_following/)
- [payloads_instruction_following_timbre/](payloads_instruction_following_timbre/)
- [payloads_timbre_diversity/](payloads_timbre_diversity/)

### Results

- [results/](results/)
- [results_timbre/](results_timbre/)
- [results_timbre_diversity/](results_timbre_diversity/)

## Environment / Requirements

### API key for instruction-following evaluation

Instruction-following stages require one of:

```bash
export API_KEY=...
# or
export MINT_BENCH_API_KEY=...
# or
export GOOGLE_API_KEY=...
```

### Model caches

WER and timbre diversity use cached models under:

```text
evaluation/seed-tts-eval/pretrained/
```

## Inspecting Results

Once `results/` has at least one model-language run, there are two recommended ways to inspect the scores.

### 1. Visualization (recommended)

We strongly recommend using the Streamlit visualization for interactive exploration: leaderboard, per-bucket bar charts, and a per-node taxonomy heatmap, all combined with WER pass-rate penalty.

```bash
bash visualization/run_visualization.sh
# or with a custom results dir
bash visualization/run_visualization.sh /path/to/results
```

See [visualization/README.md](../visualization/README.md) for full details.

### 2. CLI summary

For a quick non-interactive summary, use:

```bash
python evaluation/show_results.py
# or with a custom results dir
python evaluation/show_results.py --results_dir /path/to/results
```

This prints a compact per-model table of Overall / Timbre / Style / Easy / Hard / Special scores, and writes a structured JSON summary to disk.

## Per-Stage Documentation

For full details of each stage, see:

- [WER_EVALUATION.md](WER_EVALUATION.md)
- [INSTRUCTION_FOLLOWING_EVALUATION.md](INSTRUCTION_FOLLOWING_EVALUATION.md)
- [TIMBRE_DIVERSITY_EVALUATION.md](TIMBRE_DIVERSITY_EVALUATION.md)

Those documents cover:
- exact input / output formats
- stage-specific payload preparation
- wrapper scripts and options
- troubleshooting
- advanced usage details
