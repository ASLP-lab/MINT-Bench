import os
import json
import argparse
from typing import Any, Dict, List

TIMBRE_ATTRS = {"age", "gender", "pitch", "texture"}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_timbre_sample(sample: Dict[str, Any], file_name: str = "", taxonomy_node: str = "") -> bool:
    target_values = sample.get("target_values", {}) or {}
    attribute_set = sample.get("attribute_set", []) or []

    # 1) target_values directly includes a timbre attribute
    if any(k in TIMBRE_ATTRS for k in target_values.keys()):
        return True

    # 2) attribute_set includes a timbre attribute
    if any(x in TIMBRE_ATTRS for x in attribute_set if isinstance(x, str)):
        return True

    # 3) taxonomy node or file name directly points to a timbre node
    lowered_taxonomy = str(taxonomy_node).lower()
    lowered_name = str(file_name).lower()
    if "__timbre__" in lowered_name or "/timbre/" in lowered_taxonomy:
        return True

    return False


def collect_timbre_items(input_dir: str) -> Dict[str, Any]:
    all_items: List[Dict[str, Any]] = []
    source_files: List[str] = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        try:
            data = load_json(fpath)
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        taxonomy_node = data.get("taxonomy_node", "")
        language = data.get("language", "en")
        results = data.get("results", [])

        if not isinstance(results, list):
            continue

        kept = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if is_timbre_sample(item, file_name=fname, taxonomy_node=taxonomy_node):
                copied = dict(item)
                copied.setdefault("source_file", fname)
                copied.setdefault("source_taxonomy_node", taxonomy_node)
                copied.setdefault("language", language)
                kept.append(copied)

        if kept:
            all_items.extend(kept)
            source_files.append(fname)

    return {
        "taxonomy_node": "aggregate/timbre",
        "language": language,
        "group_name": "timbre",
        "num_items": len(all_items),
        "timbre_attributes": sorted(TIMBRE_ATTRS),
        "source_files": source_files,
        "results": all_items,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect all timbre-related instruction-text pairs into en_timbre.json")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing plan_to_instruction_text output JSON files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="en_timbre.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    output = collect_timbre_items(args.input_dir)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {output['num_items']} timbre-related items to {args.output_path}")


if __name__ == "__main__":
    main()
