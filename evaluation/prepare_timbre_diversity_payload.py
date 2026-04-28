#!/usr/bin/env python3
import os
import json
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "payloads_timbre_diversity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare speaker-similarity pair list from LALM evaluation results for timbre diversity analysis. "
        "Reads an input JSONL with per-candidate LALM scores, filters valid candidates, and writes a summary JSON "
        "describing valid pairs per plan_id."
    )
    parser.add_argument("-i", "--input_jsonl", type=Path, required=True,
                        help="Path to input JSONL file with one LALM eval record per line "
                        "(produced by lalm_eval_async.py for timbre payload)")
    parser.add_argument("-o", "--output_dir", type=Path, default=None,
                        help=f"Output directory for summary JSON (default: {DEFAULT_OUTPUT_DIR}). "
                        "Summary is saved as <output_dir>/<input_basename>.summary.json")
    parser.add_argument("--valid_level", type=int, default=3,
                        help="Keep candidates whose lalm_eval.instruction_following_level equals this value (default: 3)")
    parser.add_argument("--min_candidates_per_plan", type=int, default=2,
                        help="Minimum valid candidates needed for a plan_id to form similarity pairs (default: 2)")
    parser.add_argument("--max_candidates_per_plan", type=int, default=0,
                        help="Optional max valid candidates kept per plan_id; 0 means no limit (default: 0)")
    parser.add_argument("--require_existing_wav", action="store_true",
                        help="Only keep entries whose wav_path exists on disk")
    return parser.parse_args()


def safe_int_from_candidate_id(candidate_id: Any) -> Optional[int]:
    if candidate_id is None:
        return None
    text = str(candidate_id).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return None


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj.setdefault("_line_idx", line_idx)
                records.append(obj)
    return records


def is_valid_record(record: Dict[str, Any], valid_level: int, require_existing_wav: bool) -> bool:
    wav_path = record.get("wav_path")
    if not isinstance(wav_path, str) or not wav_path.strip():
        return False
    if require_existing_wav and not os.path.exists(wav_path):
        return False

    lalm_eval = record.get("lalm_eval")
    if not isinstance(lalm_eval, dict):
        return False
    if lalm_eval.get("instruction_following_level") != valid_level:
        return False
    return True


def record_sort_key(record: Dict[str, Any]):
    cand_idx = record.get("candidate_index")
    if isinstance(cand_idx, int):
        return (cand_idx, str(record.get("wav_path", "")))
    cand_id_int = safe_int_from_candidate_id(record.get("candidate_id"))
    if cand_id_int is not None:
        return (cand_id_int, str(record.get("wav_path", "")))
    return (10**9, str(record.get("wav_path", "")))


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_summary = output_dir / f"{args.input_jsonl.stem}.summary.json"

    records = load_jsonl_records(str(args.input_jsonl))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total_records = 0
    total_valid_records = 0

    for record in records:
        total_records += 1
        if not is_valid_record(record, args.valid_level, args.require_existing_wav):
            continue
        plan_id = str(record.get("plan_id", "")).strip()
        if not plan_id:
            continue
        grouped[plan_id].append(record)
        total_valid_records += 1

    kept_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for plan_id, items in grouped.items():
        items = sorted(items, key=record_sort_key)
        if args.max_candidates_per_plan and args.max_candidates_per_plan > 0:
            items = items[: args.max_candidates_per_plan]
        if len(items) >= args.min_candidates_per_plan:
            kept_grouped[plan_id] = items

    total_pairs = 0
    per_plan_stats: List[Dict[str, Any]] = []

    for plan_id in sorted(kept_grouped.keys()):
        items = kept_grouped[plan_id]
        pair_count = 0
        pairs: List[Dict[str, Any]] = []
        for a, b in itertools.combinations(items, 2):
            pairs.append({
                "wav1": a["wav_path"],
                "wav2": b["wav_path"],
                "uid1": str(a.get("uid", a["wav_path"])),
                "uid2": str(b.get("uid", b["wav_path"])),
                "candidate_id_1": str(a.get("candidate_id", "")),
                "candidate_id_2": str(b.get("candidate_id", "")),
            })
            pair_count += 1

        per_plan_stats.append(
            {
                "plan_id": plan_id,
                "taxonomy_node": items[0].get("taxonomy_node", ""),
                "num_valid_candidates": len(items),
                "wav_paths": [x.get("wav_path") for x in items],
                "num_pairs": pair_count,
            }
        )
        total_pairs += pair_count

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "valid_level": args.valid_level,
        "require_existing_wav": args.require_existing_wav,
        "min_candidates_per_plan": args.min_candidates_per_plan,
        "max_candidates_per_plan": args.max_candidates_per_plan,
        "total_records": total_records,
        "total_valid_records": total_valid_records,
        "num_plan_ids_with_enough_valid_candidates": len(kept_grouped),
        "total_pairs": total_pairs,
        "per_plan": per_plan_stats,
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"input_jsonl={args.input_jsonl.name} "
        f"total_records={total_records} valid_records={total_valid_records} "
        f"plans_with_pairs={len(kept_grouped)} total_pairs={total_pairs} "
        f"out={output_summary}"
    )


if __name__ == "__main__":
    main()
