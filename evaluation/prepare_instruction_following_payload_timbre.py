#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_NUM_CANDIDATES = 10


def detect_model_from_wav_dir(wav_dir: Path) -> str:
    """Infer model name from wav_dir basename, e.g. '<model>_timbre_candidates_<lang>'."""
    name = wav_dir.name
    suffix_pattern = "_timbre_candidates_"
    if suffix_pattern in name:
        return name.split(suffix_pattern)[0]
    return name


def prepare_timbre_payload(
    input_json: Path,
    wav_dir: Path,
    output_dir: Path,
    model: str,
    num_candidates: int,
) -> int:
    if not input_json.is_file():
        print(f"Error: Input JSON not found: {input_json}", file=sys.stderr)
        return 1

    if not wav_dir.is_dir():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        return 1

    with open(input_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    language = info.get("language", "unknown")
    results = info.get("results", [])

    payloads = []
    for res in results:
        plan_id = res["plan_id"]
        meta_path = res["meta_path"]
        instruction = res["instruction"]
        text = res["text"]
        target_values = res["target_values"]

        for cand_idx in range(1, num_candidates + 1):
            cand_text = text

            if model == "elevenlabs":
                try:
                    gen_meta_file = wav_dir / f"{plan_id}__cand{cand_idx:02d}.json"
                    with open(gen_meta_file, "r", encoding="utf-8") as gf:
                        gen_meta = json.load(gf)
                    text_repeat_count = gen_meta["text_repeat_count"]
                    cand_text = text * text_repeat_count
                except Exception as e:
                    print(f"Warning: {e}", file=sys.stderr)
                    continue

            if model == "hume":
                wav_path = wav_dir / f"{plan_id}__cand{cand_idx:02d}.mp3"
            else:
                wav_path = wav_dir / f"{plan_id}__cand{cand_idx:02d}.wav"

            if wav_path.exists():
                payloads.append({
                    "taxonomy_node": meta_path,
                    "plan_id": plan_id,
                    "language": language,
                    "instruction": instruction,
                    "text": cand_text,
                    "target_values": target_values,
                    "wav_path": str(wav_path),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{wav_dir.name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        f"language={language} model={model} "
        f"input_json={input_json.name} wav_dir={wav_dir.name} "
        f"items={len(payloads)} out={output_file}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare timbre diversity evaluation payload from an input JSON "
        "(timbre subset definition) and a wav directory containing multi-candidate "
        "generation outputs. Produces a JSONL payload at "
        "<output_dir>/<wav_dir_basename>.jsonl"
    )
    parser.add_argument(
        "-i", "--input_json",
        type=Path,
        required=True,
        help="Path to timbre subset JSON file (e.g., en_timbre_subset_balanced_100.json)",
    )
    parser.add_argument(
        "-w", "--wav_dir",
        type=Path,
        required=True,
        help="Path to directory containing generated multi-candidate WAV files "
        "(e.g., gen_outputs_timbre/<model>_timbre_candidates_<lang>/)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help=f"Output directory for payload JSONL (default: {SCRIPT_DIR}/payloads_timbre_diversity)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name. If not specified, inferred from wav_dir basename "
        "(everything before '_timbre_candidates_'). Used for model-specific text handling "
        "(elevenlabs text_repeat_count, hume mp3 audio).",
    )
    parser.add_argument(
        "-n", "--num_candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates per item (default: {DEFAULT_NUM_CANDIDATES})",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "payloads_timbre_diversity"

    model = args.model if args.model else detect_model_from_wav_dir(args.wav_dir)

    return prepare_timbre_payload(
        input_json=args.input_json,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        model=model,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    sys.exit(main())
