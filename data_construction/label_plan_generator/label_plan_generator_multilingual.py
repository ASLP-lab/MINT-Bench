import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from tqdm import tqdm


LANGUAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "zh": {
        "display_name": "Chinese",
        "instruction_language": "zh",
        "text_language": "zh",
        "tag_template": "属性：值",
        "needs_english_instruction": False,
        "instruction_prefix": "",
    },
    "en": {
        "display_name": "English",
        "instruction_language": "en",
        "text_language": "en",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": False,
        "instruction_prefix": "",
    },
    "ja": {
        "display_name": "Japanese",
        "instruction_language": "en",
        "text_language": "ja",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Japanese.",
    },
    "ko": {
        "display_name": "Korean",
        "instruction_language": "en",
        "text_language": "ko",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Korean.",
    },
    "pt": {
        "display_name": "Portuguese",
        "instruction_language": "en",
        "text_language": "pt",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Portuguese.",
    },
    "fr": {
        "display_name": "French",
        "instruction_language": "en",
        "text_language": "fr",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in French.",
    },
    "de": {
        "display_name": "German",
        "instruction_language": "en",
        "text_language": "de",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in German.",
    },
    "it": {
        "display_name": "Italian",
        "instruction_language": "en",
        "text_language": "it",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Italian.",
    },
    "ru": {
        "display_name": "Russian",
        "instruction_language": "en",
        "text_language": "ru",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Russian.",
    },
    "es": {
        "display_name": "Spanish",
        "instruction_language": "en",
        "text_language": "es",
        "tag_template": "Attribute: Value",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Spanish.",
    },
}

MIXED_TAG_NODE_RATIOS = {
    "easy/composite/simple_comp/static": 0.40,
    "hard/composite/complex_comp/static": 0.30,
}

DEFAULT_API_URL = "https://generativelanguage.googleapis.com"


def normalize_filter_values(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []

    normalized: List[str] = []
    for value in values:
        if value is None:
            continue
        for chunk in str(value).split(","):
            item = chunk.strip().strip("/")
            if not item:
                continue
            if item not in normalized:
                normalized.append(item)
    return normalized


def resolve_api_config(api_key: str, api_url: str) -> Tuple[str, str]:
    resolved_key = api_key or os.environ.get("MINT_BENCH_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    resolved_url = api_url or os.environ.get("MINT_BENCH_API_URL") or DEFAULT_API_URL

    if not resolved_key:
        raise ValueError(
            "Missing API key. Pass --api_key or set MINT_BENCH_API_KEY / GOOGLE_API_KEY."
        )

    return resolved_key, resolved_url.rstrip("/")


def filter_node_configs(
    node_configs: List[Dict[str, Any]],
    languages: Optional[Sequence[str]] = None,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    language_filter = {value.lower() for value in normalize_filter_values(languages)}
    taxonomy_node_filter = set(normalize_filter_values(taxonomy_nodes))
    taxonomy_prefix_filter = normalize_filter_values(taxonomy_prefixes)

    if not language_filter and not taxonomy_node_filter and not taxonomy_prefix_filter:
        return list(node_configs)

    filtered: List[Dict[str, Any]] = []
    for cfg in node_configs:
        language = str(cfg.get("language", "")).strip().lower()
        taxonomy_node = str(cfg.get("taxonomy_node", "")).strip().strip("/")

        if language_filter and language not in language_filter:
            continue
        if taxonomy_node_filter and taxonomy_node not in taxonomy_node_filter:
            continue
        if taxonomy_prefix_filter and not any(taxonomy_node.startswith(prefix) for prefix in taxonomy_prefix_filter):
            continue

        filtered.append(cfg)

    return filtered


BASE_PROMPT = r"""
You are helping construct a benchmark for instruction-following text-to-speech (TTS).

Your task in this stage is NOT to generate final instruction-text pairs.
Instead, generate a structured label plan for a given taxonomy node.

Each plan item should specify:
1. the taxonomy path,
2. the control target(s),
3. the intended target value(s),
4. the realization style needed later,
5. the intended text semantic type and text length,
6. any constraints needed later for generating the final instruction and text.

The goal of this stage is to create a clean, diverse, balanced, non-leaking, benchmark-ready scratch plan that can later be converted into final instruction-text pairs.

--------------------------------
GENERAL RULES
--------------------------------
1. Do NOT generate final polished instructions.
   - Only plan the later instruction style and constraints.

2. Do NOT generate final text sentences.
   - Only specify text_semantic, text_length, and text_constraints.

3. Text semantics must not trivially leak the target control attributes.
   Examples:
   - if the target is age=child, do not force child-themed lexical content
   - if the target is emotion=angry, do not force obviously angry words into the text
   - if the target is accent=sichuan, do not force dialect words unless explicitly required

4. Keep the difficulty determined mainly by the taxonomy node, not by overly complicated text content.

5. Prefer diversity across:
   - target values
   - semantic content types
   - attribute combinations
   - instruction realization styles when the node allows more than one realization style

6. Use the provided allowed_labels and allowed_values strictly.
   - Do not invent out-of-scope labels or values.

7. Output ONLY valid JSON.
   - No explanations outside JSON.

--------------------------------
LANGUAGE REALIZATION POLICY
--------------------------------
A. Primary languages
- If language is zh, the later final instruction and final text must both be Chinese.
- If language is en, the later final instruction and final text must both be English.

B. Non-primary languages
- If language is ja / ko / pt / fr / de / it / ru / es:
  - the later final instruction must be written in English
  - the later final instruction should explicitly cue the target language with a phrase like:
    "Speak in Japanese." / "Speak in Korean." / "Speak in French."
  - the later final text must be written in the target language itself
- Plan items must respect this downstream realization rule.

--------------------------------
OUTPUT FORMAT
--------------------------------
Return a JSON object with the following structure:

{
  "taxonomy_node": "<input taxonomy node>",
  "language": "<input language>",
  "generation_summary": {
    "num_items": <int>,
    "domain": "<timbre|style|composite|persona|extra_vocal>",
    "difficulty": "<easy|hard|special>",
    "notes": "<short note>"
  },
  "items": [
    {
      "plan_id": "<unique short id>",
      "meta_path": "<taxonomy path>",
      "attribute_set": ["<attr1>", "<attr2>"],
      "instruction_form": "<tag|direct>",
      "target_values": {
        "<attr1>": "<value1>",
        "<attr2>": "<value2>"
      },
      "aux_values": {
        "<aux_key>": "<aux_value>"
      },
      "text_semantic": "<semantic type>",
      "text_length": "<short|medium|long>",
      "text_constraints": [
        "<constraint 1>",
        "<constraint 2>"
      ],
      "instruction_constraints": [
        "<constraint 1>",
        "<constraint 2>"
      ],
      "flags": {
        "static_or_dynamic": "<static|dynamic>",
        "requires_roleplay": <true|false>,
        "requires_nonverbal_tag": <true|false>,
        "requires_abnormal_state": <true|false>,
        "contains_conflict": <true|false>,
        "contains_layered_affect": <true|false>
      }
    }
  ]
}

--------------------------------
FIELD DEFINITIONS
--------------------------------
- instruction_form:
  - tag: later final instruction should use fixed dictionary-like key-value tags
  - direct: later final instruction should use natural voice-direction phrasing

- tag format rule for later realization:
  - Chinese: use fixed "属性：值" style
  - all non-Chinese instruction languages: use fixed "Attribute: Value" style
  - for multi-tag items, each attribute must follow the same fixed pattern

- text_length:
  - short: one brief but complete utterance; avoid bare fragments
  - medium: enough material for one or two full sentences or comparable content
  - long: enough material for staged delivery, dynamic progression, or richer persona realization

--------------------------------
TEXT LENGTH PLANNING RULES
--------------------------------
- Atomic easy nodes may use short or medium.
- easy/composite/simple_comp/static should be mostly medium.
  - avoid planning overly tiny text for a two-attribute control item.
- hard/composite/complex_comp/static should be medium or long.
- dynamic nodes should be long.
- layered, conflict, persona, abnormal, implicit_nv, and explicit_nv should usually be medium or long.
- In general, do not plan very short text for hard or special nodes unless the item is unusually simple.

--------------------------------
NODE-SPECIFIC RULES
--------------------------------
A. easy / timbre / tag / atomic
- exactly one timbre attribute
- instruction_form must be tag

B. easy / timbre / direct / atomic
- exactly one timbre attribute
- instruction_form must be direct

C. easy / style / tag / atomic
- exactly one style attribute
- instruction_form must be tag

D. easy / style / direct / atomic
- exactly one style attribute
- instruction_form must be direct

E. easy / composite / simple_comp / static
- exactly two attributes only
- no conflict
- no layered affect
- no dynamic progression
- allowed combinations:
  - timbre+timbre
  - style+style
  - timbre+style
- Use a balanced mix of instruction_form values:
  - a substantial portion should be tag
  - the rest should be direct
- Tag-form composite items are important here.

F. hard / composite / complex_comp / static
- three or more attributes
- no time progression
- may involve timbre, style, or both
- Use mostly direct items, but include a meaningful minority of tag-form items.
- Tag-form complex static items are also needed here.

G. hard / composite / complex_comp / dynamic
- one or more attributes must change over time
- instruction_form should be direct
- prefer multi-stage progression rather than only binary change

H. hard / composite / complex_comp / layered
- multi-layered emotion, subtext, or hidden affect
- instruction_form should be direct
- text should allow dual interpretation

I. hard / composite / complex_comp / conflict
- conflicting or counter-stereotypical cues
- instruction_form should be direct
- text itself should remain relatively simple

J. persona nodes
- instruction_form should be direct
- scenario / role / stylized persona / archetype / ip portrait should all remain roleplay-oriented
- For ip_portrait, aux_values may store internal anchors such as franchise_or_title, character_name, and portrait_hint.
- These are internal anchors only and must not be copied directly into the final user-facing instruction.

K. special / extra_vocal / abnormal
- instruction_form should be direct
- no explicit nonverbal tags in the future text
- treat these as continuous speaking states

L. special / extra_vocal / implicit_nv
- instruction_form should be direct
- the future instruction may describe nonverbal events
- the future text must not contain explicit bracket tags
- plan event type, frequency, and rough placement if useful

M. special / extra_vocal / explicit_nv
- instruction_form should be direct
- the future text must contain explicit bracket tags
- the future instruction should control only the global speaking manner, pacing, emotional attitude, or delivery style
- the future instruction should NOT separately describe the inserted nonverbal events themselves
- because the text already contains explicit tags, do not duplicate those nonverbal cues in the future instruction

--------------------------------
TEXT SEMANTIC EXAMPLES
--------------------------------
Possible text_semantic values include:
- daily_notice
- casual_statement
- simple_request
- factual_statement
- short_reflection
- mild_dialogue
- imperative_command
- emotionally_ambiguous_statement
- narratable_sentence
- explanatory_sentence
- brief_announcement

--------------------------------
INPUT VARIABLES
--------------------------------
You will be given:
- taxonomy_node
- num_items
- allowed_labels
- allowed_values
- optional aux_value_lookup_key
- optional aux_value_lookup
- language
- language_profile

Use them strictly.

Now generate the label plan.
"""


def get_language_profile(language: str) -> Dict[str, Any]:
    return LANGUAGE_PROFILES.get(language, LANGUAGE_PROFILES["en"])


def build_label_plan_prompt(
    taxonomy_node: str,
    num_items: int,
    allowed_labels: Optional[List[str]] = None,
    allowed_values: Optional[Dict[str, List[str]]] = None,
    language: str = "zh",
    aux_value_lookup_key: Optional[str] = None,
    aux_value_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "taxonomy_node": taxonomy_node,
        "num_items": num_items,
        "allowed_labels": allowed_labels if allowed_labels is not None else [],
        "allowed_values": allowed_values if allowed_values is not None else {},
        "language": language,
        "language_profile": get_language_profile(language),
    }

    if aux_value_lookup_key is not None:
        payload["aux_value_lookup_key"] = aux_value_lookup_key
    if aux_value_lookup is not None:
        payload["aux_value_lookup"] = aux_value_lookup

    return f"{BASE_PROMPT}\n\nINPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"


def validate_label_plan_schema(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise ValueError("Returned result is not a dict")
    if "taxonomy_node" not in result:
        raise ValueError("Returned result is missing taxonomy_node")
    if "generation_summary" not in result:
        raise ValueError("Returned result is missing generation_summary")
    if "items" not in result:
        raise ValueError("Returned result is missing items")
    if not isinstance(result["items"], list):
        raise ValueError("items must be a list")


def enrich_ip_portrait_aux_values(
    result: Dict[str, Any],
    aux_value_lookup_key: Optional[str],
    aux_value_lookup: Optional[Dict[str, Dict[str, str]]],
) -> Dict[str, Any]:
    if not aux_value_lookup_key or not aux_value_lookup:
        return result

    items = result.get("items", [])
    if not isinstance(items, list):
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        target_values = item.get("target_values", {})
        if not isinstance(target_values, dict):
            continue
        portrait_label = target_values.get(aux_value_lookup_key)
        if not portrait_label:
            continue
        anchor = aux_value_lookup.get(portrait_label)
        if not anchor:
            continue
        aux_values = item.get("aux_values")
        if not isinstance(aux_values, dict):
            aux_values = {}
        for k, v in anchor.items():
            aux_values.setdefault(k, v)
        item["aux_values"] = aux_values

    return result


def extract_json_from_text(text: str) -> Any:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except Exception:
            pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr:end_arr + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from the model response")


async def generate_label_plan(
    session: aiohttp.ClientSession,
    url: str,
    key: str,
    prompt_text: str,
    timeout: int = 360,
) -> Tuple[Optional[Any], Optional[str]]:
    api_url_base = f"{url}/v1beta/models/gemini-2.5-pro:generateContent"
    full_api_url = f"{api_url_base}?key={key}"

    request_data = {
        "contents": [{"parts": [{"text": prompt_text}]}]
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with session.post(
            full_api_url,
            headers=headers,
            json=request_data,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return None, f"HTTP error: {resp.status}, body: {text}"

            data = await resp.json()

            try:
                inner_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                response_json = extract_json_from_text(inner_text)
            except Exception as e:
                return None, f"Failed to parse JSON from Gemini response: {e}; raw response: {json.dumps(data, ensure_ascii=False)[:1000]}"

            return response_json, None

    except asyncio.TimeoutError:
        return None, "Request timed out"
    except aiohttp.ClientError as e:
        return None, f"Network error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "__", name.strip("/"))


def make_plan_id(language: str, taxonomy_node: str, index: int) -> str:
    node_slug = re.sub(r"[^a-zA-Z0-9]+", "_", taxonomy_node.strip("/")).strip("_").lower()
    lang_slug = re.sub(r"[^a-zA-Z0-9]+", "", str(language).strip().lower()) or "xx"
    return f"PLAN__{lang_slug}__{node_slug}__{index:04d}"


def _append_unique(target: List[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def infer_instruction_form(item: Dict[str, Any], taxonomy_node: str) -> str:
    existing = item.get("instruction_form")
    if isinstance(existing, str) and existing in {"tag", "direct"}:
        return existing

    meta_path = str(item.get("meta_path") or taxonomy_node)
    if "/tag/" in meta_path:
        return "tag"
    if "/direct/" in meta_path:
        return "direct"

    instruction_constraints = item.get("instruction_constraints")
    if isinstance(instruction_constraints, list):
        merged = " ".join(str(x).lower() for x in instruction_constraints)
        if "dictionary" in merged or "tag" in merged or "属性：值" in merged or "attribute: value" in merged:
            return "tag"

    return "direct"


def _pick_evenly_spaced_indices(total: int, picks: int) -> List[int]:
    if picks <= 0 or total <= 0:
        return []
    if picks >= total:
        return list(range(total))
    indices = set()
    for i in range(picks):
        idx = round(i * (total - 1) / max(1, picks - 1)) if picks > 1 else 0
        indices.add(idx)
    return sorted(indices)


def normalize_label_plan_result(
    result: Dict[str, Any],
    taxonomy_node: str,
    language: str,
) -> Dict[str, Any]:
    result["taxonomy_node"] = taxonomy_node
    result["language"] = language

    items = result.get("items")
    if not isinstance(items, list):
        return result

    profile = get_language_profile(language)

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        item["plan_id"] = make_plan_id(language, taxonomy_node, idx)
        item.setdefault("meta_path", taxonomy_node)
        item.setdefault("attribute_set", [])
        item.setdefault("target_values", {})
        item.setdefault("aux_values", {})
        item.setdefault("text_constraints", [])
        item.setdefault("instruction_constraints", [])
        item.setdefault("flags", {})

        if not isinstance(item["text_constraints"], list):
            item["text_constraints"] = [str(item["text_constraints"])]
        if not isinstance(item["instruction_constraints"], list):
            item["instruction_constraints"] = [str(item["instruction_constraints"])]
        if not isinstance(item["flags"], dict):
            item["flags"] = {}
        if not isinstance(item["aux_values"], dict):
            item["aux_values"] = {}
        if not isinstance(item["target_values"], dict):
            item["target_values"] = {}
        if not isinstance(item["attribute_set"], list):
            item["attribute_set"] = list(item["target_values"].keys())

        item["instruction_form"] = infer_instruction_form(item, taxonomy_node)

        if profile["needs_english_instruction"]:
            _append_unique(
                item["instruction_constraints"],
                f"final instruction must be in English and should begin with '{profile['instruction_prefix']}'",
            )
            _append_unique(
                item["text_constraints"],
                f"final text must be written in {profile['display_name']}",
            )
        else:
            _append_unique(
                item["instruction_constraints"],
                f"final instruction should be written in {profile['display_name']}",
            )
            _append_unique(
                item["text_constraints"],
                f"final text should be written in {profile['display_name']}",
            )

        if item["instruction_form"] == "tag":
            _append_unique(
                item["instruction_constraints"],
                f"use fixed tag format '{profile['tag_template']}' in final realization",
            )
        else:
            _append_unique(
                item["instruction_constraints"],
                "use natural voice-direction phrasing in final realization",
            )

        meta_path = item["meta_path"]
        current_length = str(item.get("text_length") or "medium")
        if "hard/composite/complex_comp/dynamic" in meta_path:
            item["text_length"] = "long"
            _append_unique(item["text_constraints"], "provide enough content for staged progression")
        elif "easy/composite/simple_comp/static" in meta_path and current_length == "short":
            item["text_length"] = "medium"
        elif (
            "hard/composite/complex_comp/static" in meta_path
            or "hard/composite/complex_comp/layered" in meta_path
            or "hard/composite/complex_comp/conflict" in meta_path
            or "/persona/" in meta_path
            or "special/extra_vocal" in meta_path
        ) and current_length == "short":
            item["text_length"] = "medium"

        if "special/extra_vocal/explicit_nv" in meta_path:
            item["flags"].setdefault("requires_nonverbal_tag", True)
            _append_unique(
                item["instruction_constraints"],
                "final instruction must not mention specific nonverbal events or explicit tags; let the text tags carry those cues",
            )
            _append_unique(
                item["text_constraints"],
                "insert explicit nonverbal tags naturally into the final text",
            )
        elif "special/extra_vocal/implicit_nv" in meta_path:
            item["flags"].setdefault("requires_nonverbal_tag", False)
            _append_unique(
                item["text_constraints"],
                "final text must not contain bracketed nonverbal tags",
            )

    ratio = MIXED_TAG_NODE_RATIOS.get(taxonomy_node)
    if ratio is not None and items:
        desired_tag_count = max(1, round(len(items) * ratio))
        current_tag_indices = [i for i, x in enumerate(items) if isinstance(x, dict) and x.get("instruction_form") == "tag"]
        if len(current_tag_indices) < desired_tag_count:
            selected = _pick_evenly_spaced_indices(len(items), desired_tag_count)
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                item["instruction_form"] = "tag" if idx in selected else "direct"
                if item["instruction_form"] == "tag":
                    _append_unique(
                        item["instruction_constraints"],
                        f"use fixed tag format '{profile['tag_template']}' in final realization",
                    )
                else:
                    _append_unique(
                        item["instruction_constraints"],
                        "use natural voice-direction phrasing in final realization",
                    )

    return result


async def process_one_node(
    node_cfg: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    key: str,
    output_dir: str,
    error_log: str,
) -> Dict[str, Any]:
    async with semaphore:
        taxonomy_node = node_cfg["taxonomy_node"]
        num_items = int(node_cfg["num_items"])
        allowed_labels = node_cfg.get("allowed_labels")
        allowed_values = node_cfg.get("allowed_values")
        language = node_cfg.get("language", "zh")

        aux_value_lookup_key = node_cfg.get("aux_value_lookup_key")
        aux_value_lookup = node_cfg.get("aux_value_lookup")

        prompt_text = build_label_plan_prompt(
            taxonomy_node=taxonomy_node,
            num_items=num_items,
            allowed_labels=allowed_labels,
            allowed_values=allowed_values,
            language=language,
            aux_value_lookup_key=aux_value_lookup_key,
            aux_value_lookup=aux_value_lookup,
        )

        result, error = await generate_label_plan(
            session=session,
            url=url,
            key=key,
            prompt_text=prompt_text,
        )

        if error:
            async with aiofiles.open(error_log, "a", encoding="utf-8") as ferr:
                await ferr.write(f"ERROR {taxonomy_node}: {error}\n")
            raise RuntimeError(error)

        validate_label_plan_schema(result)

        result = enrich_ip_portrait_aux_values(
            result=result,
            aux_value_lookup_key=aux_value_lookup_key,
            aux_value_lookup=aux_value_lookup,
        )
        result = normalize_label_plan_result(
            result=result,
            taxonomy_node=taxonomy_node,
            language=language,
        )

        out_name = sanitize_filename(taxonomy_node) + ".json"
        output_file = os.path.join(output_dir, out_name)

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=2))

        return {
            "taxonomy_node": taxonomy_node,
            "output_file": output_file,
            "num_items": num_items,
        }


async def process_all_nodes(
    config_path: str,
    output_dir: str,
    error_log_path: str,
    url: str,
    key: str,
    max_concurrent: int = 5,
    languages: Optional[Sequence[str]] = None,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("")

    if config_path.endswith(".jsonl"):
        node_configs: List[Dict[str, Any]] = []
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    node_configs.append(json.loads(line))
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            node_configs = json.load(f)

    if not isinstance(node_configs, list):
        raise ValueError("The config file must contain a list of dicts or JSONL rows.")

    node_configs = filter_node_configs(
        node_configs,
        languages=languages,
        taxonomy_nodes=taxonomy_nodes,
        taxonomy_prefixes=taxonomy_prefixes,
    )

    existing = {os.path.splitext(x)[0] for x in os.listdir(output_dir)}
    pending_configs = []
    for cfg in node_configs:
        taxonomy_node = cfg["taxonomy_node"]
        out_name = sanitize_filename(taxonomy_node)
        if out_name not in existing:
            pending_configs.append(cfg)

    print(f"Total node configs: {len(node_configs)}")
    print(f"Pending node configs: {len(pending_configs)}")
    print("[INFO] Recommended workflow: run one stage at a time and inspect logs/intermediate outputs before proceeding.")

    if not pending_configs:
        print("No pending nodes to process.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                process_one_node(
                    node_cfg=cfg,
                    session=session,
                    semaphore=semaphore,
                    url=url,
                    key=key,
                    output_dir=output_dir,
                    error_log=error_log_path,
                )
            )
            for cfg in pending_configs
        ]

        success_count = 0
        error_count = 0

        pbar = tqdm(total=len(tasks), desc="Generating label plans")
        for task in asyncio.as_completed(tasks):
            try:
                await task
                success_count += 1
            except Exception:
                error_count += 1
            pbar.set_postfix_str(f"success={success_count}, error={error_count}")
            pbar.update(1)
        pbar.close()

    print(f"\nCompleted: success={success_count}, error={error_count}, total={len(pending_configs)}")
    print(f"Output directory: {output_dir}")
    print(f"Error log: {error_log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured label plans from exported node specifications."
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to the exported node config JSON or JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save one label-plan JSON per taxonomy node")
    parser.add_argument("--error_log", type=str, required=True, help="Path to the error log file")
    parser.add_argument(
        "--api_url",
        type=str,
        default="",
        help="Gemini-compatible API base URL. Falls back to MINT_BENCH_API_URL when omitted.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key. Falls back to MINT_BENCH_API_KEY or GOOGLE_API_KEY when omitted.",
    )
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent requests")
    parser.add_argument(
        "--language",
        action="append",
        help="Filter by language. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--taxonomy-node",
        action="append",
        help="Filter by exact taxonomy node. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--taxonomy-prefix",
        action="append",
        help="Filter by taxonomy prefix such as easy/style or hard/composite. Repeat the flag or provide a comma-separated list.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_key, resolved_url = resolve_api_config(args.api_key, args.api_url)

    asyncio.run(
        process_all_nodes(
            config_path=args.config_path,
            output_dir=args.output_dir,
            error_log_path=args.error_log,
            url=resolved_url,
            key=resolved_key,
            max_concurrent=args.max_concurrent,
            languages=args.language,
            taxonomy_nodes=args.taxonomy_node,
            taxonomy_prefixes=args.taxonomy_prefix,
        )
    )
