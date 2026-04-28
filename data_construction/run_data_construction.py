import argparse
import asyncio
from pathlib import Path
from typing import Optional

from nodes_generator.export_label_plan_nodes_multilingual import (
    default_output_filename,
    export_label_plan_config,
    normalize_languages,
    normalize_preset,
)
from label_plan_generator.label_plan_generator_multilingual import (
    process_all_nodes,
    resolve_api_config as resolve_label_plan_api_config,
)
from instruction_text_generator.instruction_text_generator_multilingual import (
    process_all_plan_files,
    resolve_api_config as resolve_instruction_api_config,
)


def build_default_paths(base_dir: Path, preset: str, language: str) -> dict:
    return {
        "node_config": base_dir / "nodes_generator" / "nodes" / default_output_filename(preset, language),
        "label_plan_output": base_dir / "label_plan_generator" / "label_plans" / f"label_plan_nodes_{preset}_outputs_{language}",
        "label_plan_log": base_dir / "label_plan_generator" / "logs" / f"label_plan_nodes_{preset}_{language}.log",
        "instruction_output": base_dir / "instruction_text_generator" / "instruction_text_pairs" / f"plan_to_instruction_text_{preset}_outputs_{language}",
        "instruction_log": base_dir / "instruction_text_generator" / "logs" / f"plan_to_instruction_text_{preset}_outputs_{language}.log",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MINT-Bench data construction pipeline stage by stage or end to end. "
            "Recommended workflow: run one stage at a time, inspect logs and intermediate outputs, then continue."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["node_gen", "label_plan", "instruction_text", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--preset",
        "--mode",
        dest="preset",
        default="mini",
        choices=["full", "mini", "smoke"],
        help="Benchmark preset. 'mini' is the public name; 'smoke' is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--language",
        action="append",
        help="Language to run. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="Base directory for data construction artifacts. Defaults to the current data_construction directory.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the LLM-backed stages. Falls back to MINT_BENCH_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="",
        help="Gemini-compatible API base URL. Falls back to MINT_BENCH_API_URL.",
    )
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent requests for LLM-backed stages")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of plan items per Stage C API request")
    parser.add_argument("--num_candidates_per_plan", type=int, default=3, help="Number of candidates to generate per plan item in Stage C")
    parser.add_argument("--selection_seed", type=int, default=None, help="Optional deterministic candidate selection seed for Stage C")
    parser.add_argument("--max_chunk_attempts", type=int, default=3, help="Maximum validation retries per chunk in Stage C")
    return parser.parse_args()


def print_stage_banner(stage: str) -> None:
    print(f"\n=== Running stage: {stage} ===")
    print("[INFO] Recommended workflow: inspect logs and intermediate outputs before moving to the next stage.")


async def run_for_language(
    base_dir: Path,
    preset: str,
    language: str,
    stage: str,
    api_key: str,
    api_url: str,
    max_concurrent: int,
    batch_size: int,
    num_candidates_per_plan: int,
    selection_seed: Optional[int],
    max_chunk_attempts: int,
) -> None:
    paths = build_default_paths(base_dir, preset, language)

    if stage in {"node_gen", "all"}:
        print_stage_banner(f"node_gen ({language})")
        export_label_plan_config(
            preset=preset,
            output_path=str(paths["node_config"]),
            language=language,
        )

    if stage in {"label_plan", "all"}:
        print_stage_banner(f"label_plan ({language})")
        resolved_key, resolved_url = resolve_label_plan_api_config(api_key, api_url)
        await process_all_nodes(
            config_path=str(paths["node_config"]),
            output_dir=str(paths["label_plan_output"]),
            error_log_path=str(paths["label_plan_log"]),
            url=resolved_url,
            key=resolved_key,
            max_concurrent=max_concurrent,
            languages=[language],
        )

    if stage in {"instruction_text", "all"}:
        print_stage_banner(f"instruction_text ({language})")
        resolved_key, resolved_url = resolve_instruction_api_config(api_key, api_url)
        await process_all_plan_files(
            input_path=str(paths["label_plan_output"]),
            output_dir=str(paths["instruction_output"]),
            error_log_path=str(paths["instruction_log"]),
            url=resolved_url,
            key=resolved_key,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            num_candidates_per_plan=num_candidates_per_plan,
            selection_seed=selection_seed,
            max_chunk_attempts=max_chunk_attempts,
            languages=[language],
        )


async def main_async() -> None:
    args = parse_args()
    preset = normalize_preset(args.preset)
    languages = normalize_languages(args.language) or ["en"]
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    for language in languages:
        await run_for_language(
            base_dir=base_dir,
            preset=preset,
            language=language,
            stage=args.stage,
            api_key=args.api_key,
            api_url=args.api_url,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            num_candidates_per_plan=args.num_candidates_per_plan,
            selection_seed=args.selection_seed,
            max_chunk_attempts=args.max_chunk_attempts,
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
