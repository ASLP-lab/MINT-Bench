# Timbre Diversity Evaluation

This document focuses on the advanced timbre-diversity stage of MINT-Bench evaluation. For the overall evaluation flow, see [EVALUATION.md](EVALUATION.md). For the project homepage overview, see [README.md](../README.md).

Unlike WER or standard instruction-following evaluation, timbre diversity depends on a candidate-filtering step: before computing speaker-similarity diversity, we first need to verify that the generated candidates actually satisfy the instruction.

So there are **two different timbre preparation steps with different purposes**:

1. **prepare_instruction_following_payload_timbre.py**
   - prepares payload JSONL for the LALM instruction-following evaluator
   - output goes to `evaluation/payloads_instruction_following_timbre/`
2. **prepare_timbre_diversity_payload.py**
   - reads the LALM-scored timbre result JSONL
   - filters valid candidates and writes the summary JSON used by the final timbre diversity scorer
   - output goes to `evaluation/payloads_timbre_diversity/`

The full timbre diversity flow is:

```text
timbre subset JSON + timbre candidate wav dir
  -> prepare_instruction_following_payload_timbre.py
    -> payloads_instruction_following_timbre/<run>.jsonl
      -> lalm_eval_async_timbre.sh
        -> results_timbre/<run>.jsonl
          -> prepare_timbre_diversity_payload.py
            -> payloads_timbre_diversity/<run>.summary.json
              -> cal_timbre_diversity.sh
                -> results_timbre_diversity/<run>_timbre_diversity_score.{txt,json}
```

`prepare_timbre_payload.py` is still useful when you want a plain candidate payload JSONL under `payloads_timbre_diversity/`, but the score computation itself depends on the LALM-filtered results and the summary generated from them.

## Why instruction-following comes first

Timbre diversity should only be computed over candidates that are actually valid realizations of the requested timbre instruction. Otherwise, the diversity metric would be inflated by off-target or failed generations.

So the required logic is:
- first run LALM scoring on **all candidates**
- then keep only candidates with acceptable `instruction_following_level`
- then compute pairwise speaker similarity inside each surviving candidate group

## Quick Start

By default, the timbre stage evaluates against the official MINT-Bench timbre subsets under `data_construction/instruction_text_generator/instruction_text_pairs_timbre/`. To evaluate a DIY timbre subset, pass your own subset JSON to `-i`. Audio is always loaded from the directory you pass to `-w`, so DIY candidate generations work too.

### Step 1: Prepare LALM timbre payload

```bash
# Official timbre subset
python evaluation/prepare_instruction_following_payload_timbre.py \
  -i data_construction/instruction_text_generator/instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json \
  -w gen_outputs_timbre/gemini_reference_timbre_candidates_en

# DIY timbre subset
python evaluation/prepare_instruction_following_payload_timbre.py \
  -i data_construction/instruction_text_generator/instruction_text_pairs_timbre/my_diy_timbre_subset.json \
  -w gen_outputs_timbre/mymodel_timbre_candidates_en
```

Default output:

```text
evaluation/payloads_instruction_following_timbre/gemini_reference_timbre_candidates_en.jsonl
```

### Step 2: Run instruction-following evaluation on all candidates

```bash
INPUT_PATH=evaluation/payloads_instruction_following_timbre/gemini_reference_timbre_candidates_en.jsonl \
API_KEY=your-key \
bash evaluation/lalm_eval_async_timbre.sh
```

Default result output:

```text
evaluation/results_timbre/gemini_reference_timbre_candidates_en.jsonl
```

### Step 3: Prepare timbre-diversity summary from LALM results

This step filters candidates by `lalm_eval.instruction_following_level` and groups valid candidates by `plan_id`.

```bash
python evaluation/prepare_timbre_diversity_payload.py \
  -i evaluation/results_timbre/gemini_reference_timbre_candidates_en.jsonl
```

Default summary output:

```text
evaluation/payloads_timbre_diversity/gemini_reference_timbre_candidates_en.summary.json
```

### Step 4: Compute timbre diversity score

```bash
bash evaluation/cal_timbre_diversity.sh \
  -i evaluation/payloads_timbre_diversity/gemini_reference_timbre_candidates_en.summary.json
```

Default score output:

```text
evaluation/results_timbre_diversity/
```

Files produced:

```text
gemini_reference_timbre_candidates_en_timbre_diversity_score.txt
gemini_reference_timbre_candidates_en_timbre_diversity_score.json
```

## Directory Layout

```text
evaluation/
  payloads_instruction_following_timbre/
    <model>_timbre_candidates_<lang>.jsonl

  results_timbre/
    <model>_timbre_candidates_<lang>.jsonl
    logs/

  payloads_timbre_diversity/
    <model>_timbre_candidates_<lang>.jsonl          # optional helper payload
    <model>_timbre_candidates_<lang>.summary.json   # required by cal_timbre_diversity.sh

  results_timbre_diversity/
    <model>_timbre_candidates_<lang>_timbre_diversity_score.txt
    <model>_timbre_candidates_<lang>_timbre_diversity_score.json
```

## Stage Details

### Stage A: `prepare_instruction_following_payload_timbre.py`

Inputs:
- one timbre subset JSON
- one wav directory containing multi-candidate outputs

Behavior:
- scans candidates `cand01` ... `cand10` by default
- supports `elevenlabs` text normalization via `text_repeat_count`
- supports `hume` `.mp3` audio instead of `.wav`
- writes one JSONL for the LALM evaluator

### Stage B: `lalm_eval_async_timbre.sh`

This is the LALM evaluator pass over **all timbre candidates**.

Important behavior:
- timbre payload filenames do not use `easy__...jsonl` naming
- the wrapper runs `lalm_eval_async.py --difficulty timbre`
- prompt routing still uses `taxonomy_node`, so timbre items with `easy/timbre/...` use `prompts/easy.txt`

### Stage C: `prepare_timbre_diversity_payload.py`

This step filters out bad candidates and creates the summary consumed by the diversity metric.

It:
- reads the timbre instruction-following result JSONL
- keeps only records whose `lalm_eval.instruction_following_level == valid_level`
- groups records by `plan_id`
- drops groups with too few valid candidates
- writes one summary JSON

Default filtering:
- `valid_level = 3`
- `min_candidates_per_plan = 2`

### Stage D: `cal_timbre_diversity.sh`

This calls [seed-tts-eval/cal_timbre_diversity.py](seed-tts-eval/cal_timbre_diversity.py), which:
- resolves wav paths from the summary JSON
- loads WavLM-large based speaker embeddings
- computes mean pairwise cosine similarity per group
- converts similarity to a diversity metric using:
  - `metric = (1 - mean_similarity) * (items / 10.0)`
- emits both text and JSON summaries

## Configuration

### `prepare_instruction_following_payload_timbre.py`

Key options:
- `-i/--input_json`
- `-w/--wav_dir`
- `-o/--output_dir`
- `-m/--model`
- `-n/--num_candidates`

### `prepare_timbre_diversity_payload.py`

Key options:
- `-i/--input_jsonl`
- `-o/--output_dir`
- `--valid_level`
- `--min_candidates_per_plan`
- `--max_candidates_per_plan`
- `--require_existing_wav`

### `cal_timbre_diversity.sh`

Key options:
- `-i/--input_json`
- `-o/--output_dir`
- `-d/--device`

Environment variables:
- `PYTHON_BIN`

## Required Pretrained Model

Timbre diversity requires the **WavLM-large fine-tuned speaker verification checkpoint** from the official Seed-TTS-Eval project.

Project reference:
- https://github.com/BytedanceSpeech/seed-tts-eval#:~:text=For%20SIM%2C%20we%20use%20WavLM%2Dlarge%20fine%2Dtuned%20on%20the%20speaker%20verification%20task%20(model%20link)%20to%20obtain%20speaker%20embeddings%20used%20to%20calculate%20the%20cosine%20similarity%20of%20speech%20samples%20of%20each%20test%20utterance%20against%20reference%20clips.

Checkpoint link:
- https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view

Before running timbre diversity, download the checkpoint and place it at:

```text
evaluation/seed-tts-eval/pretrained/wavlm_large_finetune.pth
```

This directory may also contain HuggingFace / Torch cache files used by the upstream embedding model. If `wavlm_large_finetune.pth` is missing, the timbre diversity stage will fail before scoring starts.

## Troubleshooting

### `LibsndfileError` when opening wav files

Cause:
- summary JSON contains relative wav paths and the consumer script resolves them from the wrong working directory

Current behavior:
- `cal_timbre_diversity.py` resolves relative wav paths against the summary file location / evaluation directory before reading audio

### No valid candidate groups remain

Possible causes:
- `valid_level=3` is too strict for the current model output quality
- too few candidates survived instruction-following filtering

Try:
- lowering `--valid_level`
- lowering `--min_candidates_per_plan`
- regenerating more candidate audio

### Checkpoint not found

Cause:
- `wavlm_large_finetune.pth` is missing from `evaluation/seed-tts-eval/pretrained/`

The timbre diversity Python script also supports an explicit `--checkpoint` argument if needed.

## Related Files

- [prepare_instruction_following_payload_timbre.py](prepare_instruction_following_payload_timbre.py)
- [lalm_eval_async_timbre.sh](lalm_eval_async_timbre.sh)
- [prepare_timbre_diversity_payload.py](prepare_timbre_diversity_payload.py)
- [cal_timbre_diversity.sh](cal_timbre_diversity.sh)
- [seed-tts-eval/cal_timbre_diversity.py](seed-tts-eval/cal_timbre_diversity.py)
