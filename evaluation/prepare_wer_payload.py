#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SUPPORTED_LANGUAGES = ["zh", "en", "de", "es", "fr", "it", "ja", "ko", "pt", "ru"]
DEFAULT_PRESET = "mini"

# Default pairs root points to the official MINT-Bench instruction-text pair set.
# Users evaluating their own DIY benchmark should pass --pairs_path explicitly.
DEFAULT_PAIRS_ROOT = (
    REPO_ROOT / "data_construction" / "instruction_text_generator" / "instruction_text_pairs_official"
)
DEFAULT_GEN_ROOT = REPO_ROOT / "gen_outputs"


def resolve_pairs_dir(pairs_path: Path, language: str, preset: str) -> Path:
    """Resolve a pairs directory.

    `pairs_path` may be:
    - a directory already containing per-node JSON files for one (preset, language)
    - a parent root containing `plan_to_instruction_text_<preset>_outputs_<lang>/`

    For backward compatibility with the legacy `smoke` naming, when preset is
    `mini` we also accept `plan_to_instruction_text_smoke_outputs_<lang>/`.
    """
    if pairs_path.is_dir():
        # If the directory directly contains JSON pair files, treat it as the run dir.
        if any(p.suffix == ".json" for p in pairs_path.iterdir()):
            return pairs_path

        candidates = [pairs_path / f"plan_to_instruction_text_{preset}_outputs_{language}"]
        if preset == "mini":
            # Legacy `smoke` naming, and the official set uses `full` for zh/en.
            candidates.append(pairs_path / f"plan_to_instruction_text_smoke_outputs_{language}")
            candidates.append(pairs_path / f"plan_to_instruction_text_full_outputs_{language}")
        for cand in candidates:
            if cand.is_dir():
                return cand

    raise FileNotFoundError(
        f"Cannot find instruction-text pairs dir under {pairs_path} for "
        f"language={language}, preset={preset}."
    )


def resolve_generation_dir(generation_path: Path, model: str, language: str, preset: str) -> Path:
    """Resolve a generation output directory.

    `generation_path` may be:
    - a directory already named `<model>_<preset>_outputs_<lang>/` (used as-is)
    - a parent root that contains such subdirectories

    For backward compatibility with the legacy `smoke` naming, when preset is
    `mini` we also accept `<model>_smoke_outputs_<lang>/`.
    """
    if generation_path.is_dir():
        expected = f"{model}_{preset}_outputs_{language}"
        if generation_path.name == expected:
            return generation_path
        if preset == "mini" and generation_path.name == f"{model}_smoke_outputs_{language}":
            return generation_path

        candidates = [generation_path / expected]
        if preset == "mini":
            candidates.append(generation_path / f"{model}_smoke_outputs_{language}")
        for cand in candidates:
            if cand.is_dir():
                return cand

    raise FileNotFoundError(
        f"Cannot find generation dir under {generation_path} for "
        f"model={model}, language={language}, preset={preset}."
    )


def iter_pairs_json_files(pairs_dir: Path):
    for p in sorted(pairs_dir.iterdir()):
        if p.is_file() and p.name.endswith(".json"):
            yield p


def prepare_wer_input(
    model: str,
    language: str,
    preset: str,
    output_dir: Path,
    pairs_path: Path,
    generation_path: Path,
) -> int:
    try:
        pairs_dir = resolve_pairs_dir(pairs_path, language, preset)
        wav_dir = resolve_generation_dir(generation_path, model, language, preset)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    lines = []
    for pth in iter_pairs_json_files(pairs_dir):
        try:
            with open(pth, "r", encoding="utf-8") as f:
                info = json.load(f)

            node_name = info["taxonomy_node"]
            if "special" in node_name or "accent" in node_name:
                continue

            for result in info["results"]:
                target_values = result["target_values"]
                if "accent" in target_values:
                    continue

                text = result["text"]
                plan_id = result["plan_id"]

                if model == "elevenlabs":
                    try:
                        gen_meta_file = wav_dir / f"{plan_id}.json"
                        with open(gen_meta_file, "r", encoding="utf-8") as gf:
                            gen_meta = json.load(gf)
                        text_repeat_count = gen_meta["text_repeat_count"]
                        text = text * text_repeat_count
                    except Exception as e:
                        print(f"Warning: {e}", file=sys.stderr)
                        continue

                if model == "hume":
                    wav_path = wav_dir / f"{plan_id}.mp3"
                else:
                    wav_path = wav_dir / f"{plan_id}.wav"

                if wav_path.exists():
                    lines.append(f"{wav_path}|{text}")
        except Exception as e:
            print(f"Warning: Error processing {pth}: {e}", file=sys.stderr)

    run_dir = output_dir / f"{model}_{preset}_outputs_{language}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "wer_input.lst"

    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(
        f"lang={language} preset={preset} "
        f"pairs_dir={pairs_dir.name} wav_dir={wav_dir.name} "
        f"items={len(lines)} out={output_file}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WER evaluation input lists for MINT-Bench TTS evaluation. "
        "Filters out accent and special nodes, handles per-model text transformations "
        "(e.g., elevenlabs text repetition), and generates lists in wav_path|reference_text format."
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name (e.g., 'gemini_reference', 'elevenlabs', 'hume')",
    )
    parser.add_argument(
        "-l", "--language",
        type=lambda s: [lang.strip() for lang in s.split(',') if lang.strip()],
        action="extend",
        dest="languages",
        default=[],
        help=f"Language code(s) to process. Can be specified multiple times (-l zh -l en) "
        f"or as a comma-separated list (-l zh,en). "
        f"Default: process all supported languages. Supported: {', '.join(SUPPORTED_LANGUAGES)}",
    )
    parser.add_argument(
        "-p", "--preset",
        default=DEFAULT_PRESET,
        help=f"Output preset name (default: {DEFAULT_PRESET}). "
        "Used to locate instruction-text pairs and generation outputs.",
    )
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=DEFAULT_PAIRS_ROOT,
        help=(
            "Instruction-text pairs root or run directory. "
            f"Default points at the official MINT-Bench test set: {DEFAULT_PAIRS_ROOT}. "
            "Pass this explicitly if you want to evaluate a DIY benchmark you built "
            "via the data construction pipeline."
        ),
    )
    parser.add_argument(
        "--generation_path",
        type=Path,
        default=DEFAULT_GEN_ROOT,
        help=(
            "Generation outputs root or specific run directory containing "
            "<model>_<preset>_outputs_<lang>/ subfolders. "
            f"Default: {DEFAULT_GEN_ROOT}."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for WER input lists (default: {SCRIPT_DIR}/payloads_wer). "
        "Lists are saved as <output_dir>/<model>_<preset>_outputs_<lang>/wer_input.lst",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "payloads_wer"

    if args.languages:
        for lang in args.languages:
            if lang not in SUPPORTED_LANGUAGES:
                print(
                    f"Error: Language '{lang}' not supported. "
                    f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}",
                    file=sys.stderr,
                )
                return 1
        languages = args.languages
    else:
        languages = SUPPORTED_LANGUAGES

    exit_code = 0
    for language in languages:
        ret = prepare_wer_input(
            args.model,
            language,
            args.preset,
            args.output_dir,
            args.pairs_path,
            args.generation_path,
        )
        if ret != 0:
            exit_code = ret

    return exit_code


if __name__ == "__main__":
    sys.exit(main())