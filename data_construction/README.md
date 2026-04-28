# MINT-Bench Data Construction

This document focuses on how to build benchmark instances for MINT-Bench. For the project overview and end-to-end usage flow, start from the root [README.md](../README.md).

The data construction pipeline builds benchmark instances in three stages:

1. **Node specification**: export the benchmark node configuration for a language and preset.
2. **Structured label planning**: generate structured label plans from the node configuration.
3. **Instruction-text pair construction**: convert label plans into final instruction-text pairs.

The recommended workflow is to run **one stage at a time**, inspect the logs and intermediate outputs, and then continue to the next stage. This is especially important for the LLM-backed stages because you may want to manually spot-check outputs for errors, drift, or hallucinated content.

## Directory layout

```text
.
├── nodes_generator/
├── label_plan_generator/
├── instruction_text_generator/
├── run_data_construction.py
└── run_data_construction.sh
```

## What each stage does

### Stage A: Node specification

`nodes_generator/export_label_plan_nodes_multilingual.py` exports the internal benchmark specification for a given language and preset.

This stage defines the base benchmark space, including:

- atomic acoustic attributes and value inventories
- supported languages
- benchmark presets (`full` and `mini`)
- taxonomy nodes and their sample counts
- the structural coverage of the benchmark

This is the main place to customize or extend the benchmark design.

### Stage B: Structured label planning

`label_plan_generator/label_plan_generator_multilingual.py` converts node specifications into structured label plans.

Each plan item describes:

- the taxonomy path
- the target control values
- the intended instruction form
- the target text semantic type and length
- downstream realization constraints

### Stage C: Instruction-text pair construction

`instruction_text_generator/instruction_text_generator_multilingual.py` converts structured label plans into final benchmark items.

Each final item contains:

- the instruction
- the text to be synthesized
- the target values
- candidate selection metadata
- quality-check metadata

## Presets and language support

MINT-Bench currently exposes two public presets:

- `full`
- `mini`

`smoke` is still accepted by the scripts as a compatibility alias, but `mini` is the public name and should be used in documentation and new commands.

Language support in the current configuration:

- `zh`, `en`: support `full` and `mini`
- `ja`, `ko`, `pt`, `fr`, `de`, `it`, `ru`, `es`: support `mini` only

## Unified runner

The simplest entry point is `run_data_construction.py`.

### Show help

```bash
python data_construction/run_data_construction.py --help
```

### Main arguments

- `--stage`: `node_gen`, `label_plan`, `instruction_text`, or `all`
- `--preset`: `full` or `mini`
- `--language`: one language, or repeat / comma-separate multiple languages
- `--base-dir`: base directory for data construction artifacts
- `--api_key`: API key for the LLM-backed stages
- `--api_url`: Gemini-compatible API base URL
- `--max_concurrent`: concurrency for the LLM-backed stages
- `--batch_size`: Stage C batch size
- `--num_candidates_per_plan`: number of Stage C candidates per plan item
- `--selection_seed`: deterministic candidate selection seed for Stage C
- `--max_chunk_attempts`: validation retries per Stage C chunk

## Recommended usage

### Run Stage A only

```bash
python data_construction/run_data_construction.py \
  --stage node_gen \
  --preset mini \
  --language en
```

### Run Stage B only

```bash
python data_construction/run_data_construction.py \
  --stage label_plan \
  --preset mini \
  --language en \
  --api_key "$MINT_BENCH_API_KEY" \
  --api_url "$MINT_BENCH_API_URL"
```

### Run Stage C only

```bash
python data_construction/run_data_construction.py \
  --stage instruction_text \
  --preset mini \
  --language en \
  --api_key "$MINT_BENCH_API_KEY" \
  --api_url "$MINT_BENCH_API_URL"
```

### Run the full pipeline

```bash
python data_construction/run_data_construction.py \
  --stage all \
  --preset mini \
  --language en \
  --api_key "$MINT_BENCH_API_KEY" \
  --api_url "$MINT_BENCH_API_URL"
```

Running all stages in one command is supported, but the recommended workflow is still to run them stage by stage.

## Shell wrapper

A thin shell wrapper is also provided:

```bash
bash data_construction/run_data_construction.sh --help
```

It forwards environment variables to `run_data_construction.py`.

Common environment variables:

- `STAGE`
- `PRESET`
- `LANGUAGE`
- `BASE_DIR`
- `API_KEY`
- `API_URL`
- `MAX_CONCURRENT`
- `BATCH_SIZE`
- `NUM_CANDIDATES_PER_PLAN`
- `SELECTION_SEED`
- `MAX_CHUNK_ATTEMPTS`

Example:

```bash
STAGE=label_plan \
PRESET=mini \
LANGUAGE=en \
API_KEY="$MINT_BENCH_API_KEY" \
API_URL="$MINT_BENCH_API_URL" \
bash data_construction/run_data_construction.sh
```

## Default outputs

By default, outputs are written under this `data_construction` directory.

### Stage A outputs

```text
nodes_generator/nodes/label_plan_nodes_{preset}_{language}.json
```

### Stage B outputs

```text
label_plan_generator/label_plans/label_plan_nodes_{preset}_outputs_{language}/
label_plan_generator/logs/label_plan_nodes_{preset}_{language}.log
```

### Stage C outputs

```text
instruction_text_generator/instruction_text_pairs/plan_to_instruction_text_{preset}_outputs_{language}/
instruction_text_generator/logs/plan_to_instruction_text_{preset}_outputs_{language}.log
```

## Notes on customization

The data construction entrypoint intentionally exposes only:

- language selection
- preset selection
- stage selection

Node definitions, node counts, attribute inventories, and coverage design are pre-defined in the node specification layer. To customize the benchmark itself, edit the configuration logic in:

- `nodes_generator/export_label_plan_nodes_multilingual.py`

That file is the main extension point for:

- adding or removing attributes
- changing value inventories
- changing benchmark coverage
- adjusting node budgets
- extending language support

## Dependencies

Stage A uses the Python standard library.

Stages B and C require the dependencies used by the LLM-backed generators, including packages such as:

- `aiohttp`
- `aiofiles`
- `tqdm`

Make sure the required Python environment is activated before running the pipeline.

## Practical recommendation

For most users, the safest workflow is:

1. run `node_gen`
2. inspect the exported node config
3. run `label_plan`
4. inspect logs and sample outputs
5. run `instruction_text`
6. inspect logs and sample outputs again

This keeps the benchmark construction process controllable and makes it easier to catch issues early.