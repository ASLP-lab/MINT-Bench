"""MINT-Bench Evaluation Leaderboard.

Reads per-model evaluation results from `mint-bench/evaluation/results/`
and renders a structured leaderboard.

Each model directory looks like:
    <model>_<preset>_outputs_<lang>/
        easy__....jsonl
        hard__....jsonl
        special__....jsonl
        wer.out                    # Step-aggregated WER summary

Score per bucket:
    main = raw_score_avg * wer_pass_rate

WER pass rate per bucket = fraction of items in that bucket whose per-item
WER is <= the language-specific threshold from the paper. Items missing WER
data are treated as passed.

Usage:
    streamlit run app_view_evaluation_results.py -- <results_dir>

If <results_dir> is omitted, the script resolves results from (in order):
    1) the CLI argument
    2) the MINT_BENCH_RESULTS_DIR env var
    3) ../evaluation/results (relative to this script)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This file lives at mint-bench/visualization/, so evaluation results are at
# mint-bench/evaluation/results/.
DEFAULT_RESULTS_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "evaluation", "results")
)

# Resolve results dir from CLI arg (for `streamlit run ... -- <path>`)
# or env var, falling back to the repo-relative default above.
if len(sys.argv) > 1 and sys.argv[1]:
    RESULTS_BASE_DIR = sys.argv[1]
elif os.environ.get("MINT_BENCH_RESULTS_DIR"):
    RESULTS_BASE_DIR = os.environ["MINT_BENCH_RESULTS_DIR"]
else:
    RESULTS_BASE_DIR = DEFAULT_RESULTS_DIR

# Folders inside results/ that are not model run dirs.
EXCLUDED_TOP_DIRS = {"logs"}

RUN_DIR_RE = re.compile(
    r"^(?P<model>.+)_(?P<preset>full|mini|smoke)_outputs_(?P<language>[A-Za-z0-9_-]+)$"
)
FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# WER thresholds (frozen, from the paper's reference models)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bucket taxonomy
# ---------------------------------------------------------------------------

TIMBRE_ATTRS = {"age", "gender", "pitch", "texture"}
STYLE_ATTRS = {"speed", "volume", "emotion", "accent", "personality", "tone"}

BUCKET_ORDER: List[str] = [
    "overall",
    "timbre",
    "style",
    "easy_tag",
    "easy_direct",
    "easy_simple_comp",
    "hard_multi_comp",
    "hard_dynamic",
    "hard_layered",
    "hard_conflict",
    "persona_scenario",
    "persona_character",
    "special_disfluency",
    "special_dysphonia",
    "special_implicit",
    "special_explicit",
]

BUCKET_LABEL = {
    "overall": "Overall",
    "timbre": "Timbre",
    "style": "Style",
    "easy_tag": "Easy · Tag",
    "easy_direct": "Easy · Direct",
    "easy_simple_comp": "Easy · Simple-comp",
    "hard_multi_comp": "Hard · Multi-comp",
    "hard_dynamic": "Hard · Dynamic",
    "hard_layered": "Hard · Layered",
    "hard_conflict": "Hard · Conflict",
    "persona_scenario": "Persona · Scenario",
    "persona_character": "Persona · Character",
    "special_disfluency": "Special · Disfluency",
    "special_dysphonia": "Special · Dysphonia",
    "special_implicit": "Special · Implicit NV",
    "special_explicit": "Special · Explicit NV",
}


# ---------------------------------------------------------------------------
# Streamlit page config + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MINT-Bench Evaluation Leaderboard",
    page_icon="🎯",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mint-bg: #f6f8fc;
            --mint-bg-2: #eef3fb;
            --mint-card: #ffffff;
            --mint-text: #0f172a;
            --mint-muted: #475569;
            --mint-line: #d9e4f5;
            --mint-accent: #2563eb;
            --mint-accent-soft: #eef4ff;
            --mint-good: #16a34a;
            --mint-warn: #f59e0b;
            --mint-bad: #dc2626;
        }
        .stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
            background:linear-gradient(180deg,var(--mint-bg) 0%,var(--mint-bg-2) 60%,var(--mint-bg) 100%);
            color:var(--mint-text);
        }
        html,body,[class*="css"]{color:var(--mint-text);}
        .block-container{max-width:1760px;padding-top:1rem;padding-bottom:2rem;}
        [data-testid="stSidebar"]{
            background:linear-gradient(180deg,#edf3fb 0%,#f5f8fd 100%);
            border-right:1px solid var(--mint-line);
        }
        [data-testid="stSidebar"] *{color:var(--mint-text) !important;}

        .hero{
            background:linear-gradient(135deg,#ffffff 0%,#eef4ff 100%);
            border:1px solid var(--mint-line);border-radius:22px;
            padding:1.1rem 1.3rem;box-shadow:0 12px 28px rgba(15,23,42,0.06);
            margin-bottom:0.9rem;
        }
        .hero-title{font-size:1.85rem;font-weight:800;margin-bottom:0.25rem;}
        .hero-sub{font-size:0.95rem;color:var(--mint-muted);margin-bottom:0.5rem;}
        .hero-pill{
            display:inline-block;padding:0.32rem 0.7rem;border-radius:999px;
            background:var(--mint-accent-soft);border:1px solid #dbe7fb;color:#1e3a8a;
            font-size:0.85rem;margin-right:0.4rem;margin-bottom:0.25rem;
        }

        .legend-row{margin:0.3rem 0 0.6rem 0;}
        .legend-pill{
            display:inline-block;padding:0.28rem 0.65rem;border-radius:999px;
            background:var(--mint-accent-soft);border:1px solid #dbe7fb;color:#1e3a8a;
            font-size:0.85rem;margin-right:0.4rem;margin-bottom:0.4rem;
        }

        .table-wrap{
            overflow-x:auto;background:rgba(255,255,255,0.94);
            border:1px solid var(--mint-line);border-radius:18px;padding:12px;
            box-shadow:0 8px 22px rgba(15,23,42,0.05);
        }
        table.lb{
            border-collapse:separate;border-spacing:0;width:100%;min-width:1700px;
            font-size:14.5px;color:var(--mint-text);
        }
        table.lb th, table.lb td{
            border-bottom:1px solid var(--mint-line);
            border-right:1px solid var(--mint-line);
            text-align:center;padding:9px 8px;vertical-align:middle;
        }
        table.lb th:first-child, table.lb td:first-child{border-left:1px solid var(--mint-line);}
        table.lb thead th{background:#eef4ff;font-weight:700;}
        table.lb thead tr.top th{background:#e2ecff;font-size:15px;}
        table.lb thead tr.mid th{background:#eaf1ff;font-size:14.5px;}
        table.lb thead tr.low th{background:#f3f7ff;font-size:14px;}
        table.lb tbody tr:nth-child(odd) td{background:#fbfdff;}
        table.lb tbody tr:hover td{background:#f0f6ff;}
        table.lb td.model{
            text-align:left;font-weight:700;min-width:160px;background:#ffffff;
        }
        table.lb td.rank{
            font-weight:800;color:var(--mint-accent);min-width:42px;background:#ffffff;
        }
        .score-main{font-weight:800;color:var(--mint-text);white-space:nowrap;font-size:15px;}
        .score-sub{font-size:11.5px;color:var(--mint-muted);margin-top:3px;line-height:1.4;}
        .empty-cell{color:#94a3b8;font-weight:600;}

        .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:700;}
        .pill-good{background:#dcfce7;color:#166534;}
        .pill-warn{background:#fef3c7;color:#92400e;}
        .pill-bad{background:#fee2e2;color:#991b1b;}
        .pill-na{background:#e5e7eb;color:#374151;}

        .summary-card{
            background:var(--mint-card);border:1px solid var(--mint-line);
            border-radius:14px;padding:14px 16px;
            box-shadow:0 4px 14px rgba(15,23,42,0.04);
        }

        table.lb.hm td.hm-cell{
            min-width:78px;
            padding:6px 6px;
            transition:transform 0.08s ease;
        }
        table.lb.hm td.hm-cell:hover{
            transform:scale(1.02);
            box-shadow:0 4px 12px rgba(15,23,42,0.12);
            position:relative;z-index:1;
        }
        table.lb.hm td.hm-empty{background:#f1f5f9;color:#94a3b8;}
        table.lb.hm .hm-main{font-weight:800;font-size:14.5px;line-height:1.1;}
        table.lb.hm .hm-sub{font-size:10.5px;line-height:1.3;color:#1f2937;opacity:0.75;margin-top:2px;}
        table.lb.hm thead th{font-size:12.5px;}
        table.lb.hm thead tr.low th{font-size:11.5px;font-weight:600;}

        .hm-legend{
            display:flex;gap:10px;align-items:center;
            margin:0.4rem 0 0.8rem 0;color:var(--mint-muted);
            font-size:0.9rem;
        }
        .hm-legend .grad{
            display:inline-block;width:160px;height:14px;border-radius:8px;
            background:linear-gradient(to right,#fee2e2,#fef3c7,#bbf7d0);
            border:1px solid var(--mint-line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    threshold_summary = ", ".join(
        f"{lang}={thr:.2%}" for lang, thr in sorted(WER_THRESHOLDS.items())
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">🎯 MINT-Bench Evaluation Leaderboard</div>
          <div class="hero-sub">
            Per-model, per-language scores combining LALM instruction-following with WER pass rate.
            Main score per bucket = raw average × WER pass rate.
          </div>
          <div>
            <span class="hero-pill">Results dir: {RESULTS_BASE_DIR}</span>
            <span class="hero-pill">Excluded: {", ".join(sorted(EXCLUDED_TOP_DIRS))}</span>
          </div>
          <div style="margin-top:0.4rem;">
            <span class="hero-pill">WER thresholds (from paper): {threshold_summary}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def safe_get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def try_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def safe_mean(vals: List[Any]) -> Optional[float]:
    clean: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if pd.notna(fv):
            clean.append(fv)
    if not clean:
        return None
    return sum(clean) / len(clean)


@st.cache_data
def discover_eval_runs(results_base_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not results_base_dir or not os.path.isdir(results_base_dir):
        return pd.DataFrame(columns=["model", "preset", "language", "dir_name", "dir_path"])

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
    return pd.DataFrame(rows)


@st.cache_data
def list_jsonl_files(run_dir: str) -> List[str]:
    if not os.path.isdir(run_dir):
        return []
    return [
        os.path.join(run_dir, f)
        for f in sorted(os.listdir(run_dir))
        if f.endswith(".jsonl")
    ]


@st.cache_data
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_index"] = idx
                items.append(obj)
            except Exception:
                continue
    return items


@st.cache_data
def load_eval_run_items(run_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for fp in list_jsonl_files(run_dir):
        items.extend(load_jsonl(fp))
    return items


@st.cache_data
def load_run_wer(run_dir: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """Parse the wer.out file inside a run directory.

    Returns (per-utterance dataframe, run-level WER percent).
    """
    wer_path = os.path.join(run_dir, "wer.out")
    if not os.path.isfile(wer_path):
        return pd.DataFrame(columns=["wav_path", "wav_basename", "wer"]), None

    rows: List[Dict[str, Any]] = []
    overall_wer: Optional[float] = None
    with open(wer_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            if line.lower().startswith("wer:"):
                value = line.split(":", 1)[1].strip().rstrip("%").strip()
                overall_wer = try_float(value, default=float("nan"))
                continue

            parts = line.split("\t")

            # Header line tolerance
            if idx == 0:
                joined = " ".join(parts).lower()
                if "wer" in joined and ("text_ref" in joined or "wav" in joined):
                    continue

            wav_path: str = ""
            wer_val: float = float("nan")

            if len(parts) >= 7:
                # Standard average_wer format: utt? wav res_wer text_ref text_res ins del sub
                if FLOAT_RE.match(str(parts[1]).strip()) and parts[0].endswith(".wav"):
                    wav_path, wer_val_str = parts[0], parts[1]
                else:
                    # Some variants prepend a row id
                    wav_path = parts[1]
                    wer_val_str = parts[2] if len(parts) > 2 else ""
                wer_val = try_float(wer_val_str)
            elif len(parts) == 2:
                wav_path, wer_val_str = parts
                wer_val = try_float(wer_val_str)
            else:
                continue

            wav_path = str(wav_path).strip()
            if not wav_path:
                continue

            rows.append(
                {
                    "wav_path": wav_path,
                    "wav_basename": os.path.basename(wav_path),
                    "wer": wer_val,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["wer"]).reset_index(drop=True)
    return df, overall_wer


# ---------------------------------------------------------------------------
# Bucketing & metrics
# ---------------------------------------------------------------------------


def has_any_attr(target_values: Any, attr_set: Set[str]) -> bool:
    return isinstance(target_values, dict) and any(k in attr_set for k in target_values.keys())


def bucket_keys_for_item(taxonomy_node: str, target_values: Any) -> Set[str]:
    keys: Set[str] = {"overall"}

    if has_any_attr(target_values, TIMBRE_ATTRS):
        keys.add("timbre")
    if has_any_attr(target_values, STYLE_ATTRS):
        keys.add("style")

    node = str(taxonomy_node or "")
    if node.startswith("easy/"):
        if "/tag/" in node:
            keys.add("easy_tag")
        elif "/direct/" in node:
            keys.add("easy_direct")
        elif node.startswith("easy/composite/simple_comp/static"):
            keys.add("easy_simple_comp")

    if node.startswith("hard/composite/complex_comp/static"):
        keys.add("hard_multi_comp")
    elif node.startswith("hard/composite/complex_comp/dynamic"):
        keys.add("hard_dynamic")
    elif node.startswith("hard/composite/complex_comp/layered"):
        keys.add("hard_layered")
    elif node.startswith("hard/composite/complex_comp/conflict"):
        keys.add("hard_conflict")

    if node.startswith("hard/composite/persona/scenario"):
        keys.add("persona_scenario")
    elif node.startswith("hard/composite/persona/"):
        keys.add("persona_character")

    if node.startswith("special/extra_vocal/abnormal/disfluency"):
        keys.add("special_disfluency")
    elif node.startswith("special/extra_vocal/abnormal/dysphonia"):
        keys.add("special_dysphonia")
    elif node.startswith("special/extra_vocal/implicit_nv"):
        keys.add("special_implicit")
    elif node.startswith("special/extra_vocal/explicit_nv"):
        keys.add("special_explicit")

    return keys


def build_eval_df_with_wer(
    eval_items: List[Dict[str, Any]],
    wer_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in eval_items:
        ev = safe_get(item, "lalm_eval", {})
        if not isinstance(ev, dict):
            continue
        raw_if = safe_get(ev, "instruction_following_level")
        raw_total = safe_get(ev, "final_score")
        wav_path = str(safe_get(item, "wav_path", "")).strip()
        if raw_if is None or raw_total is None or not wav_path:
            continue
        rows.append(
            {
                "taxonomy_node": safe_get(item, "taxonomy_node", ""),
                "target_values": safe_get(item, "target_values", {}),
                "wav_path": wav_path,
                "wav_basename": os.path.basename(wav_path),
                "raw_if": float(raw_if),
                "raw_total": float(raw_total),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if not wer_df.empty:
        sub_wer = wer_df[["wav_path", "wav_basename", "wer"]].drop_duplicates(
            subset=["wav_path"], keep="first"
        )

        df = df.merge(
            sub_wer[["wav_path", "wer"]].rename(columns={"wer": "wer_exact"}),
            on="wav_path",
            how="left",
        )
        base_wer = (
            sub_wer[["wav_basename", "wer"]]
            .drop_duplicates(subset=["wav_basename"], keep="first")
            .rename(columns={"wer": "wer_base"})
        )
        df = df.merge(base_wer, on="wav_basename", how="left")
        df["wer"] = df["wer_exact"]
        df.loc[df["wer"].isna(), "wer"] = df.loc[df["wer"].isna(), "wer_base"]
        df = df.drop(columns=[c for c in ["wer_exact", "wer_base"] if c in df.columns])
    else:
        df["wer"] = float("nan")
    return df


def compute_bucket_metrics(df: pd.DataFrame, language: str) -> Dict[str, Dict[str, Any]]:
    threshold = WER_THRESHOLDS.get(language.lower(), 1.0)

    stats: Dict[str, Dict[str, Any]] = {
        name: {
            "raw_if_avg": None,
            "raw_total_avg": None,
            "pen_if_avg": None,
            "pen_total_avg": None,
            "wer_avg": None,
            "wer_pass_rate": 1.0,
            "wer_coverage": 0,
            "count": 0,
        }
        for name in BUCKET_ORDER
    }

    if df.empty:
        return stats

    temp: Dict[str, Dict[str, List[Any]]] = {
        name: {"raw_if": [], "raw_total": [], "wer": []} for name in BUCKET_ORDER
    }

    for _, row in df.iterrows():
        keys = bucket_keys_for_item(row["taxonomy_node"], row["target_values"])
        for key in keys:
            temp[key]["raw_if"].append(row["raw_if"])
            temp[key]["raw_total"].append(row["raw_total"])
            temp[key]["wer"].append(row["wer"] if pd.notna(row["wer"]) else None)
            stats[key]["count"] += 1
            if pd.notna(row["wer"]):
                stats[key]["wer_coverage"] += 1

    for key in BUCKET_ORDER:
        stats[key]["raw_if_avg"] = safe_mean(temp[key]["raw_if"])
        stats[key]["raw_total_avg"] = safe_mean(temp[key]["raw_total"])
        stats[key]["wer_avg"] = safe_mean(temp[key]["wer"])

        wer_list = temp[key]["wer"]
        total = len(wer_list)
        if total == 0:
            stats[key]["wer_pass_rate"] = 1.0
        else:
            pass_count = sum(1 for w in wer_list if w is None or w <= threshold)
            stats[key]["wer_pass_rate"] = pass_count / total

        if stats[key]["raw_if_avg"] is not None:
            stats[key]["pen_if_avg"] = stats[key]["raw_if_avg"] * stats[key]["wer_pass_rate"]
        if stats[key]["raw_total_avg"] is not None:
            stats[key]["pen_total_avg"] = stats[key]["raw_total_avg"] * stats[key]["wer_pass_rate"]

    return stats


# ---------------------------------------------------------------------------
# Per-node matrix (for heatmap)
# ---------------------------------------------------------------------------


def _node_metrics_for_run(merged: pd.DataFrame, language: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics keyed by exact taxonomy_node string (one row per node)."""
    threshold = WER_THRESHOLDS.get(language.lower(), 1.0)
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    if merged.empty:
        return {}
    for _, row in merged.iterrows():
        node = str(row.get("taxonomy_node") or "")
        if not node:
            continue
        slot = grouped.setdefault(node, {"raw_if": [], "raw_total": [], "wer": []})
        slot["raw_if"].append(row["raw_if"])
        slot["raw_total"].append(row["raw_total"])
        slot["wer"].append(row["wer"] if pd.notna(row["wer"]) else None)

    out: Dict[str, Dict[str, Any]] = {}
    for node, slot in grouped.items():
        wer_list = slot["wer"]
        total = len(wer_list)
        pass_rate = 1.0
        if total:
            pass_count = sum(1 for w in wer_list if w is None or w <= threshold)
            pass_rate = pass_count / total
        raw_if_avg = safe_mean(slot["raw_if"])
        raw_total_avg = safe_mean(slot["raw_total"])
        wer_avg = safe_mean(slot["wer"])
        out[node] = {
            "raw_if_avg": raw_if_avg,
            "raw_total_avg": raw_total_avg,
            "pen_if_avg": (raw_if_avg * pass_rate) if raw_if_avg is not None else None,
            "pen_total_avg": (raw_total_avg * pass_rate) if raw_total_avg is not None else None,
            "wer_avg": wer_avg,
            "wer_pass_rate": pass_rate,
            "wer_coverage": sum(1 for w in wer_list if w is not None),
            "count": total,
        }
    return out


@st.cache_data
def compute_node_matrix(results_base_dir: str, preset: str, language: str) -> pd.DataFrame:
    """Return a long dataframe of (model, taxonomy_node, metric...)."""
    eval_runs = discover_eval_runs(results_base_dir)
    if eval_runs.empty:
        return pd.DataFrame()
    eval_runs = eval_runs[
        (eval_runs["preset"] == preset) & (eval_runs["language"] == language)
    ]
    if eval_runs.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, run in eval_runs.iterrows():
        items = load_eval_run_items(run["dir_path"])
        wer_df, _ = load_run_wer(run["dir_path"])
        merged = build_eval_df_with_wer(items, wer_df)
        node_stats = _node_metrics_for_run(merged, language)
        for node, stat in node_stats.items():
            row: Dict[str, Any] = {
                "model": run["model"],
                "preset": preset,
                "language": language,
                "taxonomy_node": node,
            }
            row.update(stat)
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def compute_leaderboard(results_base_dir: str, preset: str, language: str) -> pd.DataFrame:
    eval_runs = discover_eval_runs(results_base_dir)
    if eval_runs.empty:
        return pd.DataFrame()

    eval_runs = eval_runs[
        (eval_runs["preset"] == preset) & (eval_runs["language"] == language)
    ]
    if eval_runs.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, run in eval_runs.iterrows():
        items = load_eval_run_items(run["dir_path"])
        wer_df, overall_wer = load_run_wer(run["dir_path"])
        merged = build_eval_df_with_wer(items, wer_df)
        bucket_stats = compute_bucket_metrics(merged, language)

        out: Dict[str, Any] = {
            "model": run["model"],
            "preset": preset,
            "language": language,
            "dir_name": run["dir_name"],
            "valid_items": int(len(merged)),
            "wer_matched_items": int(merged["wer"].notna().sum()) if not merged.empty else 0,
            "avg_wer": float(merged["wer"].mean()) if not merged.empty and merged["wer"].notna().any() else None,
            "overall_wer_run": overall_wer,
        }
        for bucket, stat in bucket_stats.items():
            for k, v in stat.items():
                out[f"{bucket}_{k}"] = v
        rows.append(out)

    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb

    return lb.sort_values(
        by=["overall_pen_total_avg", "overall_pen_if_avg", "overall_raw_total_avg"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def fmt_pass_pill(pass_rate: Optional[float]) -> str:
    if pass_rate is None:
        return '<span class="pill pill-na">N/A</span>'
    if pass_rate >= 0.9:
        return f'<span class="pill pill-good">{pass_rate:.0%}</span>'
    if pass_rate >= 0.6:
        return f'<span class="pill pill-warn">{pass_rate:.0%}</span>'
    return f'<span class="pill pill-bad">{pass_rate:.0%}</span>'


def fmt_cell(stat: Dict[str, Any]) -> str:
    raw_if = stat.get("raw_if_avg")
    raw_total = stat.get("raw_total_avg")
    p_if = stat.get("pen_if_avg")
    p_total = stat.get("pen_total_avg")
    count = int(stat.get("count", 0) or 0)
    wer_avg = stat.get("wer_avg")
    pass_rate = stat.get("wer_pass_rate")

    if raw_if is None or raw_total is None or count == 0:
        return '<div class="empty-cell">—</div>'

    wer_txt = f"{wer_avg:.1%}" if wer_avg is not None else "NA"
    pill = fmt_pass_pill(pass_rate)

    return (
        f'<div class="score-main">{p_if:.2f} / {p_total:.2f}</div>'
        f'<div class="score-sub">raw {raw_if:.2f} / {raw_total:.2f} · '
        f'WER {wer_txt} · pass {pill}</div>'
    )


def stat_from_row(row: pd.Series, bucket: str) -> Dict[str, Any]:
    return {
        "raw_if_avg": row.get(f"{bucket}_raw_if_avg"),
        "raw_total_avg": row.get(f"{bucket}_raw_total_avg"),
        "pen_if_avg": row.get(f"{bucket}_pen_if_avg"),
        "pen_total_avg": row.get(f"{bucket}_pen_total_avg"),
        "wer_avg": row.get(f"{bucket}_wer_avg"),
        "wer_pass_rate": row.get(f"{bucket}_wer_pass_rate"),
        "wer_coverage": int(row.get(f"{bucket}_wer_coverage", 0) or 0),
        "count": int(row.get(f"{bucket}_count", 0) or 0),
    }


def render_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="table-wrap"><div class="empty-cell">No models in current slice.</div></div>'

    body_rows: List[str] = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        cells = [
            f'<td class="rank">{rank}</td>',
            f'<td class="model">{row["model"]}</td>',
        ]
        for bucket in BUCKET_ORDER:
            cells.append("<td>" + fmt_cell(stat_from_row(row, bucket)) + "</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    head = (
        '<thead>'
        '<tr class="top">'
        '<th rowspan="3">#</th>'
        '<th rowspan="3">Model</th>'
        '<th rowspan="3">Overall</th>'
        '<th rowspan="3">Timbre</th>'
        '<th rowspan="3">Style</th>'
        '<th colspan="3">Easy</th>'
        '<th colspan="6">Hard</th>'
        '<th colspan="4">Special</th>'
        '</tr>'
        '<tr class="mid">'
        '<th rowspan="2">Tag</th>'
        '<th rowspan="2">Direct</th>'
        '<th rowspan="2">Simple-comp</th>'
        '<th colspan="4">Complex comp</th>'
        '<th colspan="2">Persona</th>'
        '<th colspan="2">Abnormal</th>'
        '<th colspan="2">Nonverbal</th>'
        '</tr>'
        '<tr class="low">'
        '<th>multi-comp</th>'
        '<th>dynamic</th>'
        '<th>layered</th>'
        '<th>conflict</th>'
        '<th>scenario</th>'
        '<th>character</th>'
        '<th>disfluency</th>'
        '<th>dysphonia</th>'
        '<th>implicit</th>'
        '<th>explicit</th>'
        '</tr>'
        '</thead>'
    )

    return (
        '<div class="table-wrap"><table class="lb">'
        + head
        + '<tbody>'
        + "".join(body_rows)
        + '</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

# Hierarchical node groups for heatmap.
# Each top-level entry: (difficulty_label, [(family_label, [(branch_label, [(leaf_label, taxonomy_node)])])])
NODE_HIERARCHY: List[Tuple[str, List[Tuple[str, List[Tuple[str, List[Tuple[str, str]]]]]]]] = [
    ("Easy", [
        ("Style", [
            ("Direct · Atomic", [
                ("Accent", "easy/style/direct/atomic/accent"),
                ("Emotion", "easy/style/direct/atomic/emotion"),
                ("Personality", "easy/style/direct/atomic/personality"),
                ("Speed", "easy/style/direct/atomic/speed"),
                ("Tone", "easy/style/direct/atomic/tone"),
                ("Volume", "easy/style/direct/atomic/volume"),
            ]),
            ("Tag · Atomic", [
                ("Accent", "easy/style/tag/atomic/accent"),
                ("Emotion", "easy/style/tag/atomic/emotion"),
                ("Personality", "easy/style/tag/atomic/personality"),
                ("Speed", "easy/style/tag/atomic/speed"),
                ("Tone", "easy/style/tag/atomic/tone"),
                ("Volume", "easy/style/tag/atomic/volume"),
            ]),
        ]),
        ("Timbre", [
            ("Direct · Atomic", [
                ("Age", "easy/timbre/direct/atomic/age"),
                ("Gender", "easy/timbre/direct/atomic/gender"),
                ("Pitch", "easy/timbre/direct/atomic/pitch"),
                ("Texture", "easy/timbre/direct/atomic/texture"),
            ]),
            ("Tag · Atomic", [
                ("Age", "easy/timbre/tag/atomic/age"),
                ("Gender", "easy/timbre/tag/atomic/gender"),
                ("Pitch", "easy/timbre/tag/atomic/pitch"),
                ("Texture", "easy/timbre/tag/atomic/texture"),
            ]),
        ]),
        ("Composite", [
            ("Simple-comp", [
                ("Static", "easy/composite/simple_comp/static"),
            ]),
        ]),
    ]),
    ("Hard", [
        ("Composite", [
            ("Complex-comp", [
                ("Static", "hard/composite/complex_comp/static"),
                ("Dynamic", "hard/composite/complex_comp/dynamic"),
                ("Layered", "hard/composite/complex_comp/layered"),
                ("Conflict", "hard/composite/complex_comp/conflict"),
            ]),
            ("Persona", [
                ("Scenario", "hard/composite/persona/scenario"),
                ("Social role", "hard/composite/persona/social_role"),
                ("Stylized persona", "hard/composite/persona/stylized_persona"),
                ("IP portrait", "hard/composite/persona/ip_portrait"),
                ("Non-IP archetype", "hard/composite/persona/non_ip_archetype"),
            ]),
        ]),
    ]),
    ("Special", [
        ("Extra-vocal", [
            ("Abnormal", [
                ("Disfluency", "special/extra_vocal/abnormal/disfluency"),
                ("Dysphonia", "special/extra_vocal/abnormal/dysphonia"),
            ]),
            ("Nonverbal", [
                ("Implicit NV", "special/extra_vocal/implicit_nv"),
                ("Explicit NV", "special/extra_vocal/explicit_nv"),
            ]),
        ]),
    ]),
]


def _flatten_node_columns(
    hierarchy: List[Tuple[str, List[Tuple[str, List[Tuple[str, List[Tuple[str, str]]]]]]]],
) -> Tuple[
    List[Tuple[str, str, str, str, str]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
]:
    """Flatten the nested hierarchy into:
        leaves: list of (difficulty, family, branch, leaf_label, taxonomy_node)
        diff_spans / fam_spans / branch_spans / leaf_spans:
            [(label, colspan_or_1)] in left-to-right order at each header level.
    """
    leaves: List[Tuple[str, str, str, str, str]] = []
    diff_spans: List[Tuple[str, int]] = []
    fam_spans: List[Tuple[str, int]] = []
    branch_spans: List[Tuple[str, int]] = []
    leaf_spans: List[Tuple[str, int]] = []

    for diff_label, families in hierarchy:
        diff_count = 0
        for fam_label, branches in families:
            fam_count = 0
            for branch_label, items in branches:
                branch_count = 0
                for leaf_label, node in items:
                    leaves.append((diff_label, fam_label, branch_label, leaf_label, node))
                    leaf_spans.append((leaf_label, 1))
                    branch_count += 1
                branch_spans.append((branch_label, branch_count))
                fam_count += branch_count
            fam_spans.append((fam_label, fam_count))
            diff_count += fam_count
        diff_spans.append((diff_label, diff_count))

    return leaves, diff_spans, fam_spans, branch_spans, leaf_spans


def _value_to_color(value: Optional[float], vmin: float = 1.0, vmax: float = 5.0) -> str:
    """Map a score in [vmin, vmax] to a red→amber→green background."""
    if value is None:
        return "#f1f5f9"  # neutral gray for missing
    try:
        v = float(value)
    except Exception:
        return "#f1f5f9"
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # Two-stop gradient: 0 -> #fee2e2 (red-50), 0.5 -> #fef3c7 (amber-100), 1.0 -> #bbf7d0 (green-200)
    if ratio < 0.5:
        t = ratio / 0.5
        # red(254,226,226) -> amber(254,243,199)
        r = int(254)
        g = int(226 + (243 - 226) * t)
        b = int(226 + (199 - 226) * t)
    else:
        t = (ratio - 0.5) / 0.5
        # amber(254,243,199) -> green(187,247,208)
        r = int(254 + (187 - 254) * t)
        g = int(243 + (247 - 243) * t)
        b = int(199 + (208 - 199) * t)
    return f"rgb({r},{g},{b})"


def _text_color_for_bg(value: Optional[float]) -> str:
    """Pick a readable text color for the heatmap cell."""
    if value is None:
        return "#94a3b8"
    return "#0f172a"


def render_node_heatmap_html(
    node_df: pd.DataFrame,
    selected_models: List[str],
    metric: str,
    only_present_columns: bool,
) -> str:
    if node_df.empty:
        return '<div class="table-wrap"><div class="empty-cell">No node-level data.</div></div>'

    metric_col = metric  # one of: pen_total_avg, pen_if_avg, raw_total_avg, raw_if_avg

    # Build (model, node) -> stat lookup.
    cell_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in node_df.iterrows():
        cell_lookup[(row["model"], row["taxonomy_node"])] = row.to_dict()

    leaves, diff_spans, fam_spans, branch_spans, leaf_spans = _flatten_node_columns(NODE_HIERARCHY)

    if only_present_columns:
        present = {n for (_, n) in cell_lookup.keys()}
        leaves = [
            (d, f, b, l, n) for (d, f, b, l, n) in leaves if n in present
        ]

        # rebuild spans by walking filtered leaves in order.
        from collections import OrderedDict

        def collapse(seq: List[str]) -> List[Tuple[str, int]]:
            out: List[Tuple[str, int]] = []
            cur: Optional[str] = None
            cnt = 0
            for s in seq:
                if s == cur:
                    cnt += 1
                else:
                    if cur is not None:
                        out.append((cur, cnt))
                    cur = s
                    cnt = 1
            if cur is not None:
                out.append((cur, cnt))
            return out

        diff_spans = collapse([d for (d, _, _, _, _) in leaves])
        fam_spans = collapse([f for (_, f, _, _, _) in leaves])
        branch_spans = collapse([b for (_, _, b, _, _) in leaves])
        leaf_spans = [(l, 1) for (_, _, _, l, _) in leaves]

    if not leaves:
        return '<div class="table-wrap"><div class="empty-cell">No matching nodes.</div></div>'

    # Sort models: keep input ordering if specified, else sort by overall score.
    models = list(selected_models)
    if not models:
        models = sorted({m for (m, _) in cell_lookup.keys()})

    # Build header rows.
    head_top = "".join(
        f'<th colspan="{c}">{label}</th>' for label, c in diff_spans
    )
    head_fam = "".join(
        f'<th colspan="{c}">{label}</th>' for label, c in fam_spans
    )
    head_branch = "".join(
        f'<th colspan="{c}">{label}</th>' for label, c in branch_spans
    )
    head_leaf = "".join(
        f'<th>{label}</th>' for label, _ in leaf_spans
    )

    head = (
        '<thead>'
        f'<tr class="top"><th rowspan="4">Model</th>{head_top}</tr>'
        f'<tr class="mid">{head_fam}</tr>'
        f'<tr class="mid">{head_branch}</tr>'
        f'<tr class="low">{head_leaf}</tr>'
        '</thead>'
    )

    body_rows: List[str] = []
    for model in models:
        cells = [f'<td class="model">{model}</td>']
        for (_, _, _, _, node) in leaves:
            stat = cell_lookup.get((model, node))
            if stat is None:
                cells.append(
                    '<td class="hm-cell hm-empty">'
                    '<div class="hm-main">—</div>'
                    '</td>'
                )
                continue
            v = stat.get(metric_col)
            raw_if = stat.get("raw_if_avg")
            raw_total = stat.get("raw_total_avg")
            wer_avg = stat.get("wer_avg")
            pass_rate = stat.get("wer_pass_rate")
            count = int(stat.get("count", 0) or 0)
            bg = _value_to_color(v)
            fg = _text_color_for_bg(v)
            if v is None or count == 0:
                main = "—"
                sub = ""
            else:
                main = f"{v:.2f}"
                wer_txt = f"{wer_avg:.0%}" if wer_avg is not None else "NA"
                pr_txt = f"{pass_rate:.0%}" if pass_rate is not None else "NA"
                sub = (
                    f"raw {raw_if:.1f}/{raw_total:.1f}<br/>"
                    f"WER {wer_txt} · pass {pr_txt}"
                ) if raw_if is not None and raw_total is not None else ""
            cells.append(
                f'<td class="hm-cell" style="background:{bg};color:{fg};">'
                f'<div class="hm-main">{main}</div>'
                f'<div class="hm-sub">{sub}</div>'
                f'</td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-wrap"><table class="lb hm">'
        + head
        + "<tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def make_download_df(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "model",
        "preset",
        "language",
        "dir_name",
        "valid_items",
        "wer_matched_items",
        "avg_wer",
        "overall_wer_run",
    ]
    suffixes = [
        "raw_if_avg",
        "raw_total_avg",
        "pen_if_avg",
        "pen_total_avg",
        "wer_avg",
        "wer_pass_rate",
        "wer_coverage",
        "count",
    ]
    cols = list(base_cols)
    for b in BUCKET_ORDER:
        for s in suffixes:
            cols.append(f"{b}_{s}")
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


inject_css()
render_header()

eval_runs_df = discover_eval_runs(RESULTS_BASE_DIR)
if eval_runs_df.empty:
    st.error(
        f"No evaluation runs found under: {RESULTS_BASE_DIR}\n\n"
        "Expected subdirectories like `<model>_<preset>_outputs_<lang>/` with `.jsonl` files and `wer.out`."
    )
    st.stop()

available_presets = sorted(eval_runs_df["preset"].dropna().unique().tolist())

with st.sidebar:
    st.header("Slice")
    selected_preset = st.selectbox("Preset", available_presets, index=0)

    lang_candidates = sorted(
        eval_runs_df[eval_runs_df["preset"] == selected_preset]["language"]
        .dropna()
        .unique()
        .tolist()
    )
    selected_language = st.selectbox("Language", lang_candidates, index=0)

    model_candidates = sorted(
        eval_runs_df[
            (eval_runs_df["preset"] == selected_preset)
            & (eval_runs_df["language"] == selected_language)
        ]["model"]
        .dropna()
        .unique()
        .tolist()
    )
    selected_models = st.multiselect(
        "Models",
        options=model_candidates,
        default=model_candidates,
    )

    st.divider()
    st.caption("WER threshold for current language")
    thr = WER_THRESHOLDS.get(selected_language.lower())
    if thr is not None:
        st.metric("Threshold", f"{thr:.2%}")
    else:
        st.metric("Threshold", "1.00 (no penalty)")

st.markdown(
    '<div class="legend-row">'
    '<span class="legend-pill">Main = raw avg × WER pass rate</span>'
    '<span class="legend-pill">Pass rate = items with WER ≤ language threshold</span>'
    '<span class="legend-pill">Items missing WER count as passed</span>'
    '<span class="legend-pill">WER from wer.out inside each run dir</span>'
    '</div>',
    unsafe_allow_html=True,
)

leaderboard_df = compute_leaderboard(RESULTS_BASE_DIR, selected_preset, selected_language)
if selected_models:
    leaderboard_df = leaderboard_df[leaderboard_df["model"].isin(selected_models)].reset_index(drop=True)

# Top summary cards
cols = st.columns(4)
with cols[0]:
    st.metric("Preset", selected_preset)
with cols[1]:
    st.metric("Language", selected_language)
with cols[2]:
    st.metric("Models", len(leaderboard_df))
with cols[3]:
    if not leaderboard_df.empty:
        cov = (
            f"{int(leaderboard_df['wer_matched_items'].sum())}"
            f"/{int(leaderboard_df['valid_items'].sum())}"
        )
    else:
        cov = "0/0"
    st.metric("WER coverage", cov)

if leaderboard_df.empty:
    st.warning("No results for current slice.")
else:
    st.markdown("### Structured Leaderboard")
    st.markdown(render_table_html(leaderboard_df), unsafe_allow_html=True)

    # Quick comparison chart for overall scores.
    chart_df = (
        leaderboard_df[["model", "overall_pen_total_avg", "overall_pen_if_avg"]]
        .dropna(subset=["overall_pen_total_avg"])
        .sort_values("overall_pen_total_avg", ascending=True)
        .set_index("model")
    )
    if not chart_df.empty:
        st.markdown("### Overall comparison")
        st.bar_chart(chart_df, height=320)

    # Bucket comparison chart for the picked metric.
    st.markdown("### Per-bucket comparison")
    metric_choice = st.radio(
        "Metric",
        options=["pen_total_avg", "pen_if_avg", "raw_total_avg", "raw_if_avg"],
        format_func=lambda x: {
            "pen_total_avg": "Penalized Total",
            "pen_if_avg": "Penalized IF",
            "raw_total_avg": "Raw Total",
            "raw_if_avg": "Raw IF",
        }[x],
        horizontal=True,
    )
    bucket_cols = [f"{b}_{metric_choice}" for b in BUCKET_ORDER]
    available_cols = [c for c in bucket_cols if c in leaderboard_df.columns]
    bucket_df = leaderboard_df[["model"] + available_cols].copy()
    bucket_df.columns = ["model"] + [BUCKET_LABEL[c.rsplit("_" + metric_choice, 1)[0]] for c in available_cols]
    bucket_df = bucket_df.set_index("model")
    st.bar_chart(bucket_df, height=380)

    # Per-node heatmap (taxonomy hierarchy).
    st.markdown("### Per-node Heatmap")
    node_df = compute_node_matrix(RESULTS_BASE_DIR, selected_preset, selected_language)
    if selected_models:
        node_df = node_df[node_df["model"].isin(selected_models)].reset_index(drop=True)

    hm_cols = st.columns([2, 2, 3])
    with hm_cols[0]:
        only_present = st.checkbox(
            "Only show nodes with data",
            value=True,
            help="Hide taxonomy nodes that have no items in the current slice.",
        )
    with hm_cols[1]:
        sort_choice = st.selectbox(
            "Row order",
            options=["leaderboard", "alphabetical"],
            format_func=lambda x: "Leaderboard order" if x == "leaderboard" else "Alphabetical",
            index=0,
        )

    if sort_choice == "leaderboard":
        models_in_order = leaderboard_df["model"].tolist()
    else:
        models_in_order = sorted(leaderboard_df["model"].tolist())

    st.markdown(
        f'<div class="hm-legend">Color: '
        f'<span class="grad"></span> '
        f'<span>raw 1.0</span><span style="margin-left:auto;">raw 5.0</span> '
        f'<span style="margin-left:1rem;">Metric: <b>{metric_choice}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        render_node_heatmap_html(
            node_df,
            selected_models=models_in_order,
            metric=metric_choice,
            only_present_columns=only_present,
        ),
        unsafe_allow_html=True,
    )

    # Raw / detailed table.
    st.markdown("### Raw Table")
    download_df = make_download_df(leaderboard_df)
    st.dataframe(download_df, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mint_bench_leaderboard__{selected_preset}__{selected_language}.csv",
        mime="text/csv",
    )

    with st.expander("Debug summary"):
        st.write("Eval runs in current slice:")
        st.dataframe(
            eval_runs_df[
                (eval_runs_df["preset"] == selected_preset)
                & (eval_runs_df["language"] == selected_language)
            ].reset_index(drop=True),
            use_container_width=True,
        )
        st.write(
            "WER thresholds (frozen):",
            WER_THRESHOLDS,
        )
