# WER Evaluation Pipeline

This document focuses on the WER stage of MINT-Bench evaluation. For the overall evaluation flow, see [EVALUATION.md](EVALUATION.md). For the project homepage overview, see [README.md](../README.md).

## Overview

The WER evaluation pipeline consists of two main steps:

1. **Prepare WER input lists** — Extract reference texts and audio paths from generated TTS outputs, filtered to exclude accent and special nodes.
2. **Compute WER** — Run ASR models on the audio files and compute error rates by language.

## Quick Start

### Step 1: Prepare WER Input Lists

Generate WER input lists for a specific model. By default, the script reads pair JSONs from the official test set at:

```text
data_construction/instruction_text_generator/instruction_text_pairs_official/
```

If you want to evaluate a DIY benchmark, pass `--pairs_path` explicitly. Generation outputs are looked up under `gen_outputs/` by default, or you can pass `--generation_path`.

```bash
# Default: official pair set + gen_outputs/
python evaluation/prepare_wer_payload.py -m <model_name> -l en

# Multiple languages
python evaluation/prepare_wer_payload.py -m <model_name> -l zh -l en

# Custom output directory
python evaluation/prepare_wer_payload.py -m <model_name> -l en -o /path/to/output

# Custom preset (default: "mini")
python evaluation/prepare_wer_payload.py -m <model_name> -l en -p mini

# DIY benchmark
python evaluation/prepare_wer_payload.py \
  -m <model_name> -l en \
  --pairs_path data_construction/instruction_text_generator/instruction_text_pairs/my_diy_pairs \
  --generation_path gen_outputs/my_diy_run
```

Output structure:
```text
evaluation/payloads_wer/<model_name>_<preset>_outputs_<lang>/wer_input.lst
```

Each line in the `.lst` file has format: `<wav_path>|<reference_text>`

### Step 2: Compute WER

Run WER evaluation for one model-language pair:

```bash
bash evaluation/cal_wer_multilingual.sh -m <model_name> -l <language> -p <preset>
```

Example:

```bash
bash evaluation/cal_wer_multilingual.sh -m gemini_reference -l en -p mini
```

The script will:
- read the prepared `.lst` file for that model / language
- parallelize WER computation across available GPUs (configured via `ARNOLD_WORKER_GPU` or `-w`)
- save final results to `evaluation/results/<model_name>_<preset>_outputs_<lang>/wer.out`

## Configuration

### Environment Variables

- `PYTHON_BIN` — Python interpreter (default: `python`)
- `ARNOLD_WORKER_GPU` — Number of parallel GPU workers (default: 8)

### Model Cache

Pre-trained ASR models are cached at:
```text
evaluation/seed-tts-eval/pretrained/
```

The following models are automatically downloaded and cached on first use:
- **English and other non-Chinese languages**: OpenAI Whisper large v3
- **Chinese (zh)**: FunASR Paraformer

Ensure this directory exists and has sufficient space (~10GB) before running WER evaluation.

## Supported Languages

| Code | Language |
|------|----------|
| `zh` | Chinese (Simplified) |
| `en` | English |
| `de` | German |
| `es` | Spanish |
| `fr` | French |
| `it` | Italian |
| `ja` | Japanese |
| `ko` | Korean |
| `pt` | Portuguese |
| `ru` | Russian |

## Input Data Requirements

### Generated Audio

Generated TTS audio must be located in:
```text
gen_outputs/<model_name>_<preset>_outputs_<lang>/<plan_id>.wav
```

Audio files are matched by `plan_id` from the instruction-text pairs.

Model-specific exceptions:
- **hume**: Audio files use `.mp3` extension instead of `.wav`
- **elevenlabs**: Metadata includes `text_repeat_count` for text normalization

### Instruction-Text Pairs

Instruction-text pair JSON files must be located in:
```text
data_construction/instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_<preset>_outputs_<lang>/
```

Each JSON file contains:
- `taxonomy_node`: Node identifier (filtered to exclude `special` and `accent`)
- `results`: Array of items with `plan_id`, `text`, `target_values`

Items with `accent` in `target_values` are automatically filtered out.

## Output Format

### WER Input List (`.lst`)

Plain text file with one item per line:
```text
/path/to/audio1.wav|Reference text for audio 1
/path/to/audio2.wav|Reference text for audio 2
...
```

### WER Results (`.wer`)

Tab-separated values with columns:
```text
utt	wav_res	res_wer	text_ref	text_res	res_wer_ins	res_wer_del	res_wer_sub
...
WER: <value>%
```

Summary statistics printed to stdout:
- Average WER (%)
- Median WER (%)
- Ratio of samples with WER > 50%

## Advanced Usage

### Process Single Language Only

```bash
python evaluation/prepare_wer_payload.py -m mymodel -l zh
bash evaluation/cal_wer_multilingual.sh -m mymodel -l zh -p mini
```

### Specify Multiple Presets

To evaluate different preset versions of the same model:

```bash
python evaluation/prepare_wer_payload.py -m mymodel -l en -p full
python evaluation/prepare_wer_payload.py -m mymodel -l en -p mini
```

Results will be saved under separate directories:
- `evaluation/payloads_wer/mymodel_full_outputs_en/wer_input.lst`
- `evaluation/payloads_wer/mymodel_mini_outputs_en/wer_input.lst`

### Custom Model Text Processing

For models requiring special text handling:

- **elevenlabs**: The script automatically reads `<audio_path>.json` metadata and applies `text_repeat_count` multiplication.
- Other models: Text is used as-is from the instruction-text pairs.

## Troubleshooting

### FileNotFoundError: Cannot find instruction-text pairs dir

**Cause**: Data construction has not been run, or output directory naming does not match expected format.

**Solution:**
1. Ensure data construction stage C has completed
2. Verify output directory names match pattern: `plan_to_instruction_text_<preset>_outputs_<lang>`

### FileNotFoundError: Cannot find generation dir

**Cause**: TTS generation has not been run for the specified model and language.

**Solution:**
1. Run generation for the target model / language
2. Verify output directory is under `gen_outputs/` with naming format: `<model>_<preset>_outputs_<lang>`

### CUDA out of memory

**Cause**: GPU worker count is too high for available memory.

**Solution:**
```bash
ARNOLD_WORKER_GPU=2 bash evaluation/cal_wer_multilingual.sh -m mymodel -l en -p mini
```

### Models fail to download

**Cause**: Network issues or insufficient space in cache directory.

**Solution:**
1. Ensure `evaluation/seed-tts-eval/pretrained/` exists and has write permissions
2. Pre-download models manually if needed

## Implementation Details

### WER Computation

- **Tokenization:**
  - Character-level for CJK languages (zh, ja, ko)
  - Word-level for space-separated languages (en, pt, fr, de, it, ru, es)
- **Text normalization:** punctuation removal, whitespace normalization, language-specific conversions
- **Metric:** standard WER formula `(S + D + I) / N`

### Parallelization

The WER runner splits each language's input list into chunks, distributes them across GPUs, and aggregates results. GPU assignment is controlled by `CUDA_VISIBLE_DEVICES`.

### Resume Behavior

Running the pipeline multiple times processes all items again. No resume / skip logic is implemented in the WER stage.

## Related Files

- [prepare_wer_payload.py](prepare_wer_payload.py)
- [cal_wer_multilingual.sh](cal_wer_multilingual.sh)
- [seed-tts-eval/run_wer_multilingual.py](seed-tts-eval/run_wer_multilingual.py)
- [seed-tts-eval/average_wer.py](seed-tts-eval/average_wer.py)
