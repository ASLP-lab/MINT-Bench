import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3"]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_PAYLOAD_DIR = EVALUATION_DIR / "payloads_instruction_following"


def sanitize_filename(name: str) -> str:
    return name.strip().replace("/", "__")


def to_evaluation_relative_path(path: str) -> str:
    return os.path.relpath(path, EVALUATION_DIR).replace(os.sep, "/")


def load_node_whitelist(path: str) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("The node whitelist file must contain a JSON list.")


def infer_model_triplet_from_generation_dir(generation_dir: str) -> Optional[Tuple[str, str, str]]:
    base_name = os.path.basename(os.path.normpath(generation_dir))
    parts = base_name.split("_")
    if len(parts) < 4:
        return None
    if parts[-2] != "outputs":
        return None

    language = parts[-1]
    preset = parts[-3]
    if preset == "smoke":
        preset = "mini"
    if preset not in {"full", "mini"}:
        return None

    model_parts = parts[:-3]
    if not model_parts:
        return None

    model_name = "_".join(model_parts)
    return model_name, preset, language


def collect_pair_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")
    return [
        os.path.join(input_path, fname)
        for fname in sorted(os.listdir(input_path))
        if fname.endswith(".json")
    ]


def resolve_audio_path(generation_dir: str, plan_id: str, audio_ext: str) -> str:
    return os.path.join(generation_dir, f"{plan_id}{audio_ext}")


def resolve_text_for_model(model_name: str, generation_dir: str, plan_id: str, text: str) -> Optional[str]:
    if model_name != "elevenlabs":
        return text

    meta_path = os.path.join(generation_dir, f"{plan_id}.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            gen_meta = json.load(f)
        text_repeat_count = gen_meta["text_repeat_count"]
        return text * text_repeat_count
    except Exception:
        return None


def build_payloads_for_pair_file(
    pair_file: str,
    generation_dir: str,
    model_name: str,
    audio_ext: str,
    node_whitelist: Optional[List[str]] = None,
    skip_missing_audio: bool = True,
) -> List[Dict]:
    node_whitelist = node_whitelist or []

    with open(pair_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    taxonomy_node = info.get("taxonomy_node", "")
    if node_whitelist and taxonomy_node not in node_whitelist:
        return []

    language = info["language"]
    results = info["results"]

    payloads = []
    for res in results:
        plan_id = res["plan_id"]
        meta_path = res["meta_path"]
        instruction = res["instruction"]
        text = res["text"]

        resolved_text = resolve_text_for_model(model_name, generation_dir, plan_id, text)
        if resolved_text is None:
            continue

        wav_path = resolve_audio_path(generation_dir, plan_id, audio_ext)
        if skip_missing_audio and not os.path.exists(wav_path):
            continue

        payloads.append(
            {
                "taxonomy_node": meta_path,
                "plan_id": plan_id,
                "language": language,
                "instruction": instruction,
                "text": resolved_text,
                "target_values": res["target_values"],
                "wav_path": to_evaluation_relative_path(wav_path),
            }
        )

    return payloads


def collect_generation_targets(generation_path: str, model_name_override: str) -> List[Tuple[str, str, str, str]]:
    if os.path.isfile(generation_path):
        raise ValueError("generation_path must be a directory, not a file.")
    if not os.path.isdir(generation_path):
        raise ValueError(f"Generation path does not exist: {generation_path}")

    inferred = infer_model_triplet_from_generation_dir(generation_path)
    if inferred is not None:
        model_name, preset, language = inferred
        if model_name_override:
            model_name = model_name_override
        return [(generation_path, model_name, preset, language)]

    targets: List[Tuple[str, str, str, str]] = []
    for entry in sorted(os.listdir(generation_path)):
        full_path = os.path.join(generation_path, entry)
        if not os.path.isdir(full_path):
            continue
        inferred = infer_model_triplet_from_generation_dir(full_path)
        if inferred is None:
            continue
        model_name, preset, language = inferred
        if model_name_override:
            model_name = model_name_override
        targets.append((full_path, model_name, preset, language))

    if not targets:
        raise ValueError(
            "Could not find any generation output directories. Use a directory name like <model_name>_<preset>_outputs_<language>, or pass a parent directory containing such directories."
        )
    return targets


def resolve_pairs_path_for_target(pairs_path: str, preset: str, language: str) -> str:
    if os.path.isfile(pairs_path):
        return pairs_path

    candidate = os.path.join(pairs_path, f"plan_to_instruction_text_{preset}_outputs_{language}")
    if os.path.isdir(candidate):
        return candidate

    fallback_candidate = os.path.join(pairs_path, f"plan_to_instruction_text_smoke_outputs_{language}")
    if preset == "mini" and os.path.isdir(fallback_candidate):
        return fallback_candidate

    raise ValueError(
        f"Could not resolve pair directory for preset={preset}, language={language} under {pairs_path}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare evaluation payload JSONL files from instruction-text pair JSONs and generated audio outputs.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to a pair JSON file, a directory of pair JSON files, or the instruction_text_pairs root directory.")
    parser.add_argument("--generation_path", type=str, required=True, help="Either one generation output directory or a parent directory containing multiple model output directories.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PAYLOAD_DIR), help="Directory for generated payload JSONL files. Defaults to evaluation/payloads.")
    parser.add_argument("--model_name", type=str, default="", help="Optional model name override. Normally inferred from the generation directory basename.")
    parser.add_argument("--audio_ext", type=str, default=".wav", help="Generated audio extension, including the leading dot. Default: .wav")
    parser.add_argument("--node_whitelist_json", type=str, default="", help="Optional JSON list of taxonomy_node values to include.")
    parser.add_argument("--skip_missing_audio", action="store_true", help="Skip payload entries whose generated audio file is missing.")
    parser.add_argument("--summary_path", type=str, default="", help="Optional summary JSON path. Defaults to <output_dir>/summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    node_whitelist = load_node_whitelist(args.node_whitelist_json) if args.node_whitelist_json else []
    generation_targets = collect_generation_targets(args.generation_path, args.model_name)

    total_payloads = 0
    written_files = 0
    skipped_empty = 0
    total_pair_files = 0
    processed_targets = []

    for generation_dir, model_name, preset, language in generation_targets:
        resolved_pairs_path = resolve_pairs_path_for_target(args.pairs_path, preset, language)
        pair_files = collect_pair_files(resolved_pairs_path)
        total_pair_files += len(pair_files)

        output_subdir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(generation_dir)))
        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        target_written_files = 0
        target_payloads = 0
        target_skipped_empty = 0

        for pair_file in pair_files:
            payloads = build_payloads_for_pair_file(
                pair_file=pair_file,
                generation_dir=generation_dir,
                model_name=model_name,
                audio_ext=args.audio_ext,
                node_whitelist=node_whitelist,
                skip_missing_audio=args.skip_missing_audio,
            )

            if not payloads:
                skipped_empty += 1
                target_skipped_empty += 1
                continue

            output_file = os.path.join(
                output_subdir,
                os.path.basename(pair_file).replace(".json", ".jsonl"),
            )
            with open(output_file, "w", encoding="utf-8") as f:
                for payload in payloads:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            total_payloads += len(payloads)
            written_files += 1
            target_payloads += len(payloads)
            target_written_files += 1

        processed_targets.append(
            {
                "generation_dir": to_evaluation_relative_path(generation_dir),
                "output_subdir": to_evaluation_relative_path(output_subdir),
                "model_name": model_name,
                "preset": preset,
                "language": language,
                "input_pair_files": len(pair_files),
                "written_payload_files": target_written_files,
                "skipped_empty_files": target_skipped_empty,
                "total_payload_items": target_payloads,
            }
        )

    summary = {
        "pairs_path": to_evaluation_relative_path(args.pairs_path),
        "generation_path": to_evaluation_relative_path(args.generation_path),
        "output_dir": to_evaluation_relative_path(args.output_dir),
        "audio_ext": args.audio_ext,
        "node_whitelist_json": to_evaluation_relative_path(args.node_whitelist_json) if args.node_whitelist_json else "",
        "skip_missing_audio": args.skip_missing_audio,
        "generation_targets": processed_targets,
        "input_pair_files": total_pair_files,
        "written_payload_files": written_files,
        "skipped_empty_files": skipped_empty,
        "total_payload_items": total_payloads,
        "note": "This payload layout assumes generated audio basenames match plan_id, which is the recommended evaluation-compatible generation mode.",
    }

    summary_path = args.summary_path or os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Prepared {written_files} payload files with {total_payloads} payload items.")
    print(f"Payload directory: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
