#!/usr/bin/env python3
"""Summarize MINT-Bench evaluation results into a JSON file and a CLI table.

Reads per-model evaluation results from `mint-bench/evaluation/results/`,
applies the same scoring as the visualization app (raw average × WER pass
rate, with frozen per-language WER thresholds), and prints a compact
per-model summary of:

    Overall · Timbre · Style · Easy · Hard · Special

Also writes a structured JSON summary to disk.

Usage:
    python evaluation/show_results.py
    python evaluation/show_results.py --results_dir /path/to/results
    python evaluation/show_results.py --output /tmp/summary.json --metric pen_total
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EXCLUDED_TOP_DIRS = {"logs"}

RUN_DIR_RE = re.compile(
    r"^(?P<model>.+)_(?P<preset>full|mini|smoke)_outputs_(?P<language>[A-Za-z0-9_-]+)$"
)
FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


# Frozen per-language WER thresholds (from the paper).
WER_THRESHOLDS: Dict[str, float] = {
    "de": 0.1657,
    "en": 0.0225,
    "es": 0.0917,
    "fr": 0.1840,
    "it": 0.1090,
    "ja": 0.3764,
    "ko": 0.4820,
    "pt": 0.1587,
    "ru": 0.3718,
    "zh": 0.0150,
}


# Bucket taxonomy aligned with the visualization app.
TIMBRE_ATTRS = {"age", "gender", "pitch", "texture"}
STYLE_ATTRS = {"speed", "volume", "emotion", "accent", "personality", "tone"}

SUMMARY_BUCKETS: List[Tuple[str, str]] = [
    ("overall", "Overall"),
    ("timbre", "Timbre"),
    ("style", "Style"),
    ("easy", "Easy"),
    ("hard", "Hard"),
    ("special", "Special"),
]


def safe_get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def try_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def safe_mean(vals: Iterable[Any]) -> Optional[float]:
    clean: List[float] = []
    for v in vals:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:  # NaN
            continue
        clean.append(fv)
    if not clean:
        return None
    return sum(clean) / len(clean)


def has_any_attr(target_values: Any, attr_set: Set[str]) -> bool:
    return isinstance(target_values, dict) and any(k in attr_set for k in target_values.keys())


def bucket_keys_for_item(taxonomy_node: str, target_values: Any) -> Set[str]:
    """Map an evaluation item to one or more summary buckets.

    Buckets are simpler than the visualization app: top-level only.
    """
    keys: Set[str] = {"overall"}

    if has_any_attr(target_values, TIMBRE_ATTRS):
        keys.add("timbre")
    if has_any_attr(target_values, STYLE_ATTRS):
        keys.add("style")

    node = str(taxonomy_node or "")
    if node.startswith("easy/"):
        keys.add("easy")
    elif node.startswith("hard/"):
        keys.add("hard")
    elif node.startswith("special/"):
        keys.add("special")

    return keys


def discover_eval_runs(results_base_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not results_base_dir or not os.path.isdir(results_base_dir):
        return rows
    for entry in sorted(os.listdir(results_base_dir)):
        full = os.path.join(results_base_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry in EXCLUDED_TOP_DIRS:
            continue
        m = RUN_DIR_RE.match(entry)
        if not m:
            continue
        info = m.groupdict()
        rows.append(
            {
                "model": info["model"],
                "preset": info["preset"],
                "language": info["language"],
                "dir_name": entry,
                "dir_path": full,
            }
        )
    return rows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def load_run_items(run_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for f in sorted(os.listdir(run_dir)):
        if f.endswith(".jsonl"):
            items.extend(load_jsonl(os.path.join(run_dir, f)))
    return items


def load_run_wer(run_dir: str) -> Tuple[Dict[str, float], Dict[str, float], Optional[float]]:
    """Parse `wer.out` inside a run directory.

    Returns (wer_by_path, wer_by_basename, run_level_wer_percent_or_None).
    """
    path = os.path.join(run_dir, "wer.out")
    by_path: Dict[str, float] = {}
    by_base: Dict[str, float] = {}
    overall: Optional[float] = None
    if not os.path.isfile(path):
        return by_path, by_base, overall

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lower().startswith("wer:"):
                value = line.split(":", 1)[1].strip().rstrip("%").strip()
                overall = try_float(value, default=float("nan"))
                continue
            parts = line.split("\t")
            if idx == 0:
                joined = " ".join(parts).lower()
                if "wer" in joined and ("text_ref" in joined or "wav" in joined):
                    continue

            wav_path = ""
            wer_val: float = float("nan")
            if len(parts) >= 7:
                if FLOAT_RE.match(str(parts[1]).strip()) and parts[0].endswith(".wav"):
                    wav_path, wer_str = parts[0], parts[1]
                else:
                    wav_path = parts[1]
                    wer_str = parts[2] if len(parts) > 2 else ""
                wer_val = try_float(wer_str)
            elif len(parts) == 2:
                wav_path, wer_str = parts
                wer_val = try_float(wer_str)
            else:
                continue

            wav_path = str(wav_path).strip()
            if not wav_path or wer_val != wer_val:
                continue
            by_path[wav_path] = wer_val
            by_base[os.path.basename(wav_path)] = wer_val
    return by_path, by_base, overall


def compute_run_summary(
    run: Dict[str, str],
) -> Dict[str, Any]:
    """Compute per-bucket scores for a single (model, preset, language) run."""
    items = load_run_items(run["dir_path"])
    wer_by_path, wer_by_base, run_wer = load_run_wer(run["dir_path"])
    threshold = WER_THRESHOLDS.get(run["language"].lower(), 1.0)

    bucket_collect: Dict[str, Dict[str, List[Any]]] = {
        bucket: {"raw_if": [], "raw_total": [], "wer": []}
        for bucket, _ in SUMMARY_BUCKETS
    }

    valid = 0
    wer_matched = 0
    for item in items:
        ev = safe_get(item, "lalm_eval", {})
        if not isinstance(ev, dict):
            continue
        raw_if = safe_get(ev, "instruction_following_level")
        raw_total = safe_get(ev, "final_score")
        wav_path = str(safe_get(item, "wav_path", "")).strip()
        if raw_if is None or raw_total is None or not wav_path:
            continue
        valid += 1
        wer = wer_by_path.get(wav_path)
        if wer is None:
            wer = wer_by_base.get(os.path.basename(wav_path))
        if wer is not None:
            wer_matched += 1

        keys = bucket_keys_for_item(
            safe_get(item, "taxonomy_node", ""),
            safe_get(item, "target_values", {}),
        )
        for key in keys:
            slot = bucket_collect[key]
            slot["raw_if"].append(float(raw_if))
            slot["raw_total"].append(float(raw_total))
            slot["wer"].append(wer)

    bucket_stats: Dict[str, Dict[str, Any]] = {}
    for bucket, _ in SUMMARY_BUCKETS:
        slot = bucket_collect[bucket]
        n = len(slot["raw_if"])
        wer_vals = slot["wer"]
        if n == 0:
            pass_rate = 1.0
        else:
            pass_count = sum(1 for w in wer_vals if w is None or w <= threshold)
            pass_rate = pass_count / n

        raw_if_avg = safe_mean(slot["raw_if"])
        raw_total_avg = safe_mean(slot["raw_total"])
        wer_avg = safe_mean(slot["wer"])
        pen_if = raw_if_avg * pass_rate if raw_if_avg is not None else None
        pen_total = raw_total_avg * pass_rate if raw_total_avg is not None else None
        bucket_stats[bucket] = {
            "count": n,
            "raw_if_avg": raw_if_avg,
            "raw_total_avg": raw_total_avg,
            "pen_if_avg": pen_if,
            "pen_total_avg": pen_total,
            "wer_avg": wer_avg,
            "wer_pass_rate": pass_rate,
            "wer_coverage": sum(1 for w in wer_vals if w is not None),
        }

    return {
        "model": run["model"],
        "preset": run["preset"],
        "language": run["language"],
        "dir_name": run["dir_name"],
        "language_threshold": threshold,
        "valid_items": valid,
        "wer_matched_items": wer_matched,
        "run_wer_percent": run_wer,
        "buckets": bucket_stats,
    }


def fmt_score(stat: Dict[str, Any], metric: str) -> str:
    val = stat.get(metric)
    n = stat.get("count", 0)
    if val is None or n == 0:
        return "  --  "
    return f"{val:5.2f}"


def fmt_pass_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "  -- "
    return f"{rate:5.1%}"


def print_table(summaries: List[Dict[str, Any]], metric: str) -> None:
    if not summaries:
        print("(no runs found)")
        return

    headers = ["Model", "Preset", "Lang"] + [label for _, label in SUMMARY_BUCKETS] + ["Items", "WER%", "Pass"]

    rows: List[List[str]] = []
    for s in summaries:
        row = [
            s["model"],
            s["preset"],
            s["language"],
        ]
        for bucket, _ in SUMMARY_BUCKETS:
            row.append(fmt_score(s["buckets"][bucket], metric))
        row.append(str(s["valid_items"]))
        run_wer = s.get("run_wer_percent")
        row.append(f"{run_wer:.2f}" if isinstance(run_wer, (int, float)) and run_wer == run_wer else "  -- ")
        row.append(fmt_pass_rate(s["buckets"]["overall"].get("wer_pass_rate")))
        rows.append(row)

    widths = [max(len(str(c)) for c in [headers[i]] + [r[i] for r in rows]) for i in range(len(headers))]
    line_sep = "+".join("-" * (w + 2) for w in widths)
    line_sep = f"+{line_sep}+"

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) if i < 3 else c.rjust(widths[i]) for i, c in enumerate(cells)) + " |"

    print(line_sep)
    print(fmt_row(headers))
    print(line_sep)
    for row in rows:
        print(fmt_row(row))
    print(line_sep)
    print(f"Metric: {metric}   (cells show raw average × WER pass rate; '--' = no items)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MINT-Bench evaluation results into a JSON file "
        "and print a compact per-model table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Path to evaluation results directory containing per-run subdirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the structured JSON summary. "
             "Defaults to <results_dir>/summary.json.",
    )
    parser.add_argument(
        "--metric",
        choices=["pen_total_avg", "pen_if_avg", "raw_total_avg", "raw_if_avg"],
        default="pen_total_avg",
        help="Which metric to print in the per-bucket table.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Filter to one preset (e.g. mini). Default: include all presets.",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        default=None,
        help="Filter to one or more languages. Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the table (still writes the JSON summary).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results_dir does not exist: {args.results_dir}", file=sys.stderr)
        return 1

    runs = discover_eval_runs(args.results_dir)
    if args.preset:
        runs = [r for r in runs if r["preset"] == args.preset]
    if args.languages:
        runs = [r for r in runs if r["language"] in set(args.languages)]

    if not runs:
        print(f"No matching evaluation runs found under {args.results_dir}", file=sys.stderr)
        return 1

    summaries = [compute_run_summary(run) for run in runs]
    summaries.sort(
        key=lambda s: (
            s["preset"],
            s["language"],
            -(s["buckets"]["overall"].get("pen_total_avg") or float("-inf")),
            s["model"],
        )
    )

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.results_dir, "summary.json")

    payload = {
        "results_dir": os.path.abspath(args.results_dir),
        "metric_used_in_table": args.metric,
        "wer_thresholds": WER_THRESHOLDS,
        "buckets": [{"key": k, "label": l} for k, l in SUMMARY_BUCKETS],
        "summaries": summaries,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print_table(summaries, args.metric)
        print()
    print(f"Summary JSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
