# MINT-Bench Generation

This document focuses on audio generation and the output conventions required by downstream evaluation. For the project overview and full pipeline, start from the root [README.md](../README.md).

The current open-source reference implementations are:

- Gemini TTS
- Gemini TTS timbre-candidate generation
- MiMo Pro TTS Voice Design

These scripts are useful in two ways:

1. They provide runnable examples for generating benchmark audio from instruction-text pair JSON files.
2. They define the recommended output naming and output directory conventions that downstream evaluation expects.

## Directory layout

```text
.
├── gemini/
│   ├── gemini_tts_infer_from_pairs_concurrent.py
│   ├── gemini_tts_infer_from_pairs_concurrent.sh
│   ├── gemini_tts_infer_from_pairs_timbre.py
│   └── gemini_tts_infer_from_pairs_timbre.sh
└── mimopro/
    ├── mimo_pro_tts_voicedesign_infer.py
    └── mimo_pro_tts_voicedesign_infer.sh
```

## Input format

The reference generators consume outputs from `data_construction`.

### Standard generation input

Standard generation uses the Stage C instruction-text pair JSON files with a `results` list.

Typical input sources:

- a single per-node JSON file
- a directory of per-node JSON files
- a recursively scanned directory tree of pair JSON files

For example:

```text
data_construction/instruction_text_generator/instruction_text_pairs/
  plan_to_instruction_text_mini_outputs_en/
  plan_to_instruction_text_mini_outputs_zh/
```

### Timbre candidate generation input

Timbre candidate generation uses the timbre subset JSON files.

For example:

```text
data_construction/instruction_text_generator/instruction_text_pairs_timbre/
  en_timbre_subset_balanced_100.json
  zh_timbre_subset_balanced_100.json
```

These files define one timbre-focused subset and are used to generate multiple candidate audios per plan.

## Recommended output convention

### Standard generation

For compatibility with downstream evaluation, the recommended generation convention is:

- use **one output directory per model/run family**
- use `plan_id` as the audio basename
- keep the sidecar metadata JSON next to the audio file with the same basename

That means the recommended mode is:

```bash
--use_plan_id_as_name
```

Example output layout:

```text
gen_outputs/
  gemini_reference_mini_outputs_en/
    PLAN__en__easy_style_tag_atomic_emotion__0001.wav
    PLAN__en__easy_style_tag_atomic_emotion__0001.json
    manifest.json
    summary.json
    errors.jsonl
```

### Timbre candidate generation

For timbre candidate generation, the recommended convention is:

- use **one output directory per model-language timbre run**
- use `plan_id` as the base name
- append `__candXX` for each candidate audio
- keep sidecar metadata JSON next to each candidate audio when applicable

Example output layout:

```text
gen_outputs_timbre/
  gemini_reference_timbre_candidates_en/
    PLAN__en__easy_timbre_direct_atomic_age__0001__cand01.wav
    PLAN__en__easy_timbre_direct_atomic_age__0001__cand02.wav
    PLAN__en__easy_timbre_direct_atomic_age__0001__cand03.wav
    ...
    summary.json
    errors.jsonl
```

## Why the naming convention matters

Downstream evaluation currently assumes that generated audio can be located predictably from:

- the model-specific output directory
- the `plan_id`
- for timbre diversity, the candidate suffix `__candXX`

So if you want the reference evaluation scripts to work with minimal adaptation, keep:

- model-oriented output directories
- `plan_id`-based audio filenames
- timbre candidates named with `__candXX`

## Per-audio metadata

Each generated audio file may be paired with a metadata JSON file.

The metadata JSON is not required by the evaluator itself, but it is useful for:

- tracing the instruction and text used to synthesize the audio
- checking target values and taxonomy node information
- debugging failed or partial runs
- inspecting provider-specific response identifiers

Path-like fields in metadata are stored relative to the `evaluation` directory where applicable, so the metadata is more portable across machines.

## Manifest, summary, and error log

Each run may produce:

- `manifest*.json`: the discovered input items included in the run
- `summary*.json`: run-level counts and configuration summary
- `errors.jsonl`: per-item generation failures

These files are intended to help users debug generation and audit partial runs.

## Resume behavior

Generation may be incomplete for reasons such as:

- provider safety or moderation filters
- transient API failures
- unstable provider behavior
- interrupted runs

For this reason, the reference generators support resumable generation through:

```bash
--skip_existing
```

When enabled, the script checks the current output directory first and skips items whose output audio already looks complete.

This allows users to rerun the same command repeatedly and fill in missing items without regenerating completed audio.

## Recommended workflow

### Standard generation

1. generate audio into a model-specific output directory
2. inspect the generated files and metadata
3. inspect `errors.jsonl`
4. rerun with `--skip_existing` to fill gaps
5. continue iterating until generation coverage is acceptable

### Timbre candidate generation

1. prepare the timbre subset JSON in `instruction_text_pairs_timbre/`
2. generate multiple candidate audios per plan into `gen_outputs_timbre/<model>_timbre_candidates_<lang>/`
3. inspect `errors.jsonl` and coverage of `cand01` ... `cand10`
4. rerun with `--skip_existing` to fill gaps
5. pass those candidates into the timbre evaluation pipeline

This is especially important for benchmark evaluation, because missing audio files will later reduce evaluation coverage.

## Node-level subsetting

Generation supports optional node-level filtering through:

```bash
--node_whitelist_json
```

This should point to a JSON list of taxonomy node strings.

Example:

```json
[
  "easy/style/tag/atomic/emotion",
  "hard/composite/complex_comp/dynamic",
  "special/extra_vocal/explicit_nv"
]
```

This is useful for:

- dry-run testing
- debugging a subset of benchmark cases
- reducing generation cost
- model-specific experiments on selected control types

## Gemini reference generator

Main script:

- [gemini_tts_infer_from_pairs_concurrent.py](gemini/gemini_tts_infer_from_pairs_concurrent.py)

### Main arguments

- `--input_path`: pair JSON file or directory
- `--output_dir`: output directory for audio and metadata
- `--model_name`: Gemini TTS model name
- `--sample_rate`: output WAV sample rate
- `--skip_existing`: skip completed audio files
- `--node_whitelist_json`: optional taxonomy node subset
- `--error_log`: optional error log path
- `--recursive`: recursively scan input directories
- `--use_plan_id_as_name`: recommended for evaluation compatibility
- `--api_key`: explicit key, otherwise `GOOGLE_API_KEY`
- `--api_url`: Gemini API base URL
- `--seed`: deterministic voice selection seed
- `--timeout`: request timeout in seconds
- `--num_workers`: maximum request concurrency

### Example

```bash
python generation/gemini/gemini_tts_infer_from_pairs_concurrent.py \
  --input_path data_construction/instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_mini_dryrun_outputs_en \
  --output_dir gen_outputs/gemini_reference_mini_outputs_en \
  --model_name gemini-2.5-flash-preview-tts \
  --use_plan_id_as_name \
  --skip_existing \
  --recursive
```

If you do not pass `--api_key`, set:

```bash
export GOOGLE_API_KEY=...
```

### Example wrapper

A thin wrapper is also provided:

- [gemini_tts_infer_from_pairs_concurrent.sh](gemini/gemini_tts_infer_from_pairs_concurrent.sh)

It is an example launcher, not a multi-language batch script.

## Gemini timbre-candidate generator

Main script:

- [gemini_tts_infer_from_pairs_timbre.py](gemini/gemini_tts_infer_from_pairs_timbre.py)

This generator is used for the advanced timbre-diversity pipeline. It synthesizes multiple candidates for each timbre subset item.

### Main arguments

- `--input_path`: timbre subset JSON file or directory
- `--output_dir`: output directory for candidate WAV and metadata files
- `--model_name`: Gemini TTS model name
- `--sample_rate`: output WAV sample rate
- `--skip_existing`: skip completed candidate files
- `--node_whitelist_json`: optional taxonomy node subset
- `--error_log`: optional error log path
- `--recursive`: recursively scan input directories
- `--use_plan_id_as_name`: recommended for evaluation compatibility
- `--api_key`: explicit key, otherwise `GOOGLE_API_KEY`
- `--api_url`: Gemini API base URL
- `--seed`: deterministic voice selection seed
- `--timeout`: request timeout in seconds
- `--num_workers`: maximum request concurrency
- `--timbre_taxonomy_node`: node name identifying timbre subset
- `--num_timbre_candidates`: number of candidates to generate per timbre item
- `--num_default_candidates`: number of candidates to generate for non-timbre items

### Example

```bash
python generation/gemini/gemini_tts_infer_from_pairs_timbre.py \
  --input_path data_construction/instruction_text_generator/instruction_text_pairs_timbre/en_timbre_subset_balanced_100.json \
  --output_dir gen_outputs_timbre/gemini_reference_timbre_candidates_en \
  --model_name gemini-2.5-flash-preview-tts \
  --use_plan_id_as_name \
  --skip_existing \
  --seed 42 \
  --num_timbre_candidates 10
```

If you do not pass `--api_key`, set:

```bash
export GOOGLE_API_KEY=...
```

### Example wrapper

A thin wrapper is also provided:

- [gemini_tts_infer_from_pairs_timbre.sh](gemini/gemini_tts_infer_from_pairs_timbre.sh)

It is an example launcher, not a batch script. Configure it through environment variables such as `INPUT_PATH`, `OUTPUT_DIR`, `NUM_TIMBRE_CANDIDATES`, and `GOOGLE_API_KEY`.

## MiMo reference generator

Main script:

- [mimo_pro_tts_voicedesign_infer.py](mimopro/mimo_pro_tts_voicedesign_infer.py)

### Main arguments

- `--input_path`: pair JSON file or directory
- `--output_dir`: output directory for audio and metadata
- `--model`: MiMo model name
- `--base_url`: MiMo API base URL
- `--api_key`: explicit key, otherwise `MIMO_API_KEY`
- `--audio_format`: output format, e.g. `wav`
- `--skip_existing`: skip completed audio files
- `--recursive`: recursively scan input directories
- `--node_whitelist_json`: optional taxonomy node subset
- `--use_plan_id_as_name`: recommended for evaluation compatibility
- `--error_log`: optional error log path

### Example

```bash
python generation/mimopro/mimo_pro_tts_voicedesign_infer.py \
  --input_path data_construction/instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_mini_dryrun_outputs_en \
  --output_dir gen_outputs/mimopro_reference_mini_outputs_en \
  --model mimo-v2.5-tts-voicedesign \
  --use_plan_id_as_name \
  --skip_existing \
  --recursive
```

If you do not pass `--api_key`, set:

```bash
export MIMO_API_KEY=...
```

### Example wrapper

A thin wrapper is also provided:

- [mimo_pro_tts_voicedesign_infer.sh](mimopro/mimo_pro_tts_voicedesign_infer.sh)

It is an example launcher, not a multi-language batch script.

## Dry-run recommendation

For a quick end-to-end smoke test of generation and evaluation integration, use the existing dry-run subsets from `data_construction`.

For example:

```text
data_construction/instruction_text_generator/instruction_text_pairs/
  plan_to_instruction_text_mini_dryrun_outputs_en/
  plan_to_instruction_text_mini_dryrun_outputs_zh/
```

These are good starting points for:

- verifying provider access
- checking output naming
- validating metadata and summary artifacts
- testing evaluation payload preparation later

For timbre-diversity dry runs, use the timbre subset JSON files and generate a small set of candidate audios before running the advanced evaluation chain.

## Practical notes

- Review generated outputs before evaluation.
- Expect occasional incomplete runs.
- Keep rerunning with `--skip_existing` until coverage is acceptable.
- Prefer `--use_plan_id_as_name` unless you have a strong reason to use a custom basename mode.
- Keep output directories organized by model so evaluation can find them easily.
- For timbre generation, keep candidate counts and directory naming stable so the advanced timbre evaluation scripts can discover them directly.
