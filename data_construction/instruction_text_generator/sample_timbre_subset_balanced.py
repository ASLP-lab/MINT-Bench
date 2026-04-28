import os
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

TIMBRE_ATTRS = ["age", "gender", "pitch", "texture"]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_primary_timbre_attr(item: Dict[str, Any]) -> Optional[str]:
    """
    Infer the primary timbre attribute for this sample as reliably as possible.
    Priority:
    1. exactly one timbre key appears in target_values
    2. source_taxonomy_node / meta_path explicitly mentions age/gender/pitch/texture
    3. pick the first timbre attribute found in target_values / attribute_set using a fixed order
    """
    target_values = item.get("target_values", {}) or {}
    attribute_set = item.get("attribute_set", []) or []
    meta_path = str(item.get("meta_path", "") or "")
    source_taxonomy_node = str(item.get("source_taxonomy_node", "") or "")
    source_file = str(item.get("source_file", "") or "")

    tv_keys = [k for k in target_values.keys() if k in TIMBRE_ATTRS]
    if len(tv_keys) == 1:
        return tv_keys[0]

    search_text = " || ".join([meta_path, source_taxonomy_node, source_file]).lower()
    for attr in TIMBRE_ATTRS:
        if f"/{attr}" in search_text or f"__{attr}" in search_text or search_text.endswith(attr):
            return attr

    for attr in TIMBRE_ATTRS:
        if attr in tv_keys:
            return attr

    for attr in TIMBRE_ATTRS:
        if attr in attribute_set:
            return attr

    return None


def group_by_timbre_attr(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {attr: [] for attr in TIMBRE_ATTRS}
    unknown = []

    for item in results:
        if not isinstance(item, dict):
            continue
        attr = infer_primary_timbre_attr(item)
        if attr is None:
            unknown.append(item)
        else:
            groups[attr].append(item)

    return groups, unknown


def balanced_sample(groups: Dict[str, List[Dict[str, Any]]], n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Keep age/gender/pitch/texture as balanced as possible.
    If one group runs short, redistribute the remaining quota to other groups.
    """
    rng = random.Random(seed)

    # Shuffle each group so we do not always pick the first few items.
    groups = {k: list(v) for k, v in groups.items()}
    for k in groups:
        rng.shuffle(groups[k])

    available_attrs = [k for k, v in groups.items() if len(v) > 0]
    if not available_attrs:
        return [], {k: 0 for k in TIMBRE_ATTRS}

    n = min(n, sum(len(v) for v in groups.values()))

    # Initial even split.
    base = n // len(available_attrs)
    remainder = n % len(available_attrs)

    quotas = {k: 0 for k in TIMBRE_ATTRS}
    for attr in available_attrs:
        quotas[attr] = base

    # Distribute the remainder randomly across available groups.
    shuffled_attrs = available_attrs[:]
    rng.shuffle(shuffled_attrs)
    for attr in shuffled_attrs[:remainder]:
        quotas[attr] += 1

    selected = []
    taken = {k: 0 for k in TIMBRE_ATTRS}

    # First pass: sample according to the group quotas.
    for attr in available_attrs:
        take_n = min(quotas[attr], len(groups[attr]))
        selected.extend(groups[attr][:take_n])
        taken[attr] += take_n
        groups[attr] = groups[attr][take_n:]

    # Second pass: if a group runs short, fill the remaining quota from the others.
    remaining_needed = n - len(selected)
    if remaining_needed > 0:
        refill_pool = []
        for attr in available_attrs:
            refill_pool.extend([(attr, x) for x in groups[attr]])
        rng.shuffle(refill_pool)

        for attr, item in refill_pool[:remaining_needed]:
            selected.append(item)
            taken[attr] += 1

    rng.shuffle(selected)
    return selected, taken


def main():
    parser = argparse.ArgumentParser(description="Balanced sampling for timbre subset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to en_timbre.json")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save subset json")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data = load_json(args.input_path)
    results = data.get("results", [])
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError("Input json does not contain valid 'results'")

    groups, unknown = group_by_timbre_attr(results)
    sampled, taken = balanced_sample(groups, args.num_samples, args.seed)

    output = {
        "taxonomy_node": data.get("taxonomy_node", "aggregate/timbre"),
        "language": data.get("language", "en"),
        "group_name": "timbre_balanced_subset",
        "num_items": len(sampled),
        "sampling_summary": {
            "requested_num_samples": args.num_samples,
            "actual_num_samples": len(sampled),
            "seed": args.seed,
            "taken_per_timbre_attr": taken,
            "available_per_timbre_attr": {k: len(v) for k, v in groups.items()},
            "num_unknown_items_ignored": len(unknown),
        },
        "results": sampled,
    }

    save_json(output, args.output_path)

    print(f"[OK] Saved {len(sampled)} items to {args.output_path}")
    print("[INFO] Taken per timbre attr:")
    for attr in TIMBRE_ATTRS:
        print(f"  - {attr}: {taken[attr]}")


if __name__ == "__main__":
    main()