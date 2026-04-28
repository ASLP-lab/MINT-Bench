import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from tqdm import tqdm


DEFAULT_NUM_CANDIDATES = 3
DEFAULT_MAX_CHUNK_ATTEMPTS = 3
DEFAULT_API_URL = "https://generativelanguage.googleapis.com"
CJK_LANGUAGES = {"zh", "ja", "ko"}
BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


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


def should_keep_plan_file(
    plan_data: Dict[str, Any],
    file_name: str,
    languages: Optional[Sequence[str]] = None,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
) -> bool:
    language_filter = {value.lower() for value in normalize_filter_values(languages)}
    taxonomy_node_filter = set(normalize_filter_values(taxonomy_nodes))
    taxonomy_prefix_filter = normalize_filter_values(taxonomy_prefixes)

    language = str(plan_data.get("language", "")).strip().lower()
    taxonomy_node = str(plan_data.get("taxonomy_node", "")).strip().strip("/")
    file_stem = os.path.splitext(file_name)[0]

    if language_filter and language not in language_filter:
        return False

    if taxonomy_node_filter and taxonomy_node not in taxonomy_node_filter and file_stem not in taxonomy_node_filter:
        return False

    if taxonomy_prefix_filter and not any(taxonomy_node.startswith(prefix) for prefix in taxonomy_prefix_filter):
        return False

    return True


LANGUAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "zh": {
        "name": "Chinese",
        "instruction_language": "zh",
        "text_language": "zh",
        "needs_english_instruction": False,
        "instruction_prefix": "",
        "tag_colon": "：",
    },
    "en": {
        "name": "English",
        "instruction_language": "en",
        "text_language": "en",
        "needs_english_instruction": False,
        "instruction_prefix": "",
        "tag_colon": ":",
    },
    "ja": {
        "name": "Japanese",
        "instruction_language": "en",
        "text_language": "ja",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Japanese.",
        "tag_colon": ":",
    },
    "ko": {
        "name": "Korean",
        "instruction_language": "en",
        "text_language": "ko",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Korean.",
        "tag_colon": ":",
    },
    "pt": {
        "name": "Portuguese",
        "instruction_language": "en",
        "text_language": "pt",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Portuguese.",
        "tag_colon": ":",
    },
    "fr": {
        "name": "French",
        "instruction_language": "en",
        "text_language": "fr",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in French.",
        "tag_colon": ":",
    },
    "de": {
        "name": "German",
        "instruction_language": "en",
        "text_language": "de",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in German.",
        "tag_colon": ":",
    },
    "it": {
        "name": "Italian",
        "instruction_language": "en",
        "text_language": "it",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Italian.",
        "tag_colon": ":",
    },
    "ru": {
        "name": "Russian",
        "instruction_language": "en",
        "text_language": "ru",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Russian.",
        "tag_colon": ":",
    },
    "es": {
        "name": "Spanish",
        "instruction_language": "en",
        "text_language": "es",
        "needs_english_instruction": True,
        "instruction_prefix": "Speak in Spanish.",
        "tag_colon": ":",
    },
}

NONVERBAL_KEYWORDS_EN = {
    "breath", "inhale", "exhale", "quick breath", "sigh", "gasp", "panting", "wheezing", "snore", "yawn",
    "cough", "sneeze", "throat clearing", "throat clear", "hiccup", "sniff", "sniffle", "snort",
    "chuckle", "giggle", "laugh", "laugh harder", "start laughing", "stifled laugh", "burst of laughter",
    "crying", "sobbing", "crying loudly", "wail", "whimper",
    "hum", "humming", "groan", "moan", "grunt", "mumble",
    "lipsmack", "gulp", "swallow", "burp", "tsk", "sss", "clucking", "hissing", "whisper",
}

NONVERBAL_KEYWORDS_ZH = {
    "呼吸", "吸气", "呼气", "快速吸气", "叹气", "喘息", "喘", "打鼾", "打哈欠",
    "咳嗽", "喷嚏", "清嗓", "打嗝", "抽鼻子", "鼻音", "窃笑", "轻笑", "大笑", "憋笑", "爆笑",
    "哭", "哭泣", "抽泣", "哀号", "呜咽", "哼", "呻吟", "咕哝", "感叹", "咂嘴", "吞咽", "嘶", "耳语",
}

BASE_PROMPT = r"""
You are helping construct a benchmark for instruction-following text-to-speech (TTS).

Your task is to convert a structured label plan into final instruction-text pairs.

For each plan item, generate EXACTLY {num_candidates} candidate instruction-text pairs.
The candidates do not need to be maximally different from one another. They may vary naturally in wording, phrasing, or text content.
The goal is to produce faithful, usable candidates and then let the pipeline randomly select one.

--------------------------------
GENERAL RULES
--------------------------------
1. Follow the label plan strictly.
   - Do not change meta_path.
   - Do not add extra control targets.
   - Do not ignore instruction_form, text_semantic, or instruction_constraints.

2. The instruction must match the intended realization mechanism.
   - tag: fixed key-value tags
   - direct: natural voice-direction phrasing

3. The text must be natural and speakable for TTS.
   - Avoid fragments unless the plan clearly calls for a very short utterance.
   - Avoid over-literary or obscure wording.

4. The text must NOT trivially leak the target.
   Examples:
   - child voice target does not mean child-themed lexical content
   - angry target does not mean explicit angry wording
   - accent target does not mean dialect spellings unless explicitly required

5. Difficulty should come mainly from the control design, not from making the text content weird or artificially complicated.

6. Respect aux_values only as interpretation hints.
   - This matters especially for ip_portrait.
   - Do NOT copy franchise names, titles, or character names into the final instruction or text.

7. Output ONLY valid JSON.

--------------------------------
LANGUAGE POLICY
--------------------------------
A. zh
- instruction must be Chinese
- text must be Chinese

B. en
- instruction must be English
- text must be English

C. ja / ko / pt / fr / de / it / ru / es
- instruction must be English
- instruction must explicitly cue the target language with a prefix like:
  "Speak in Japanese." / "Speak in Korean." / "Speak in French."
- text must be written in the target language itself

--------------------------------
TAG INSTRUCTION RULES
--------------------------------
If instruction_form is tag, follow these strictly.

1. Chinese tag instructions must use fixed "属性：值" format.
2. All non-Chinese instruction languages must use fixed "Attribute: Value" format.
3. For multi-attribute tag items, every attribute must appear in the same fixed format.
4. Do not paraphrase tag items into free prose.
5. For non-primary languages, keep the English target-language prefix and then use English key-value tags.
   Example pattern:
   "Speak in Japanese. Attribute: Value, Attribute: Value"

--------------------------------
DIRECT INSTRUCTION RULES
--------------------------------
If instruction_form is direct:
- use natural, operational voice-direction phrasing
- keep easy items concise
- make hard items richer only when the control complexity truly needs it
- for ip_portrait, use a richer and more distinctive direction style than ordinary persona items

--------------------------------
NODE-SPECIFIC RULES
--------------------------------
A. easy / composite / simple_comp / static
- exactly two attributes only
- some items are tag-form and some are direct-form; obey the plan item's instruction_form

B. hard / composite / complex_comp / static
- three or more attributes
- some items are tag-form and some are direct-form; obey the plan item's instruction_form
- for direct-form items, integrate the attributes coherently
- for tag-form items, keep strict key-value format

C. hard / composite / complex_comp / dynamic
- describe temporal change clearly and in order
- preserve multi-stage progression when present

D. hard / composite / complex_comp / layered
- instruction should make the layered reading operationally clear
- text should support dual interpretation

E. hard / composite / complex_comp / conflict
- instruction should make the contradiction vivid and usable
- text itself may stay relatively plain

F. persona
- scenario, social_role, stylized_persona, and non_ip_archetype are direct-form persona instructions
- non_ip_archetype may either directly name the generic archetype or describe it as a vocal portrait
- ip_portrait must not copy copyrighted names or titles from aux_values
- instead, convert them into a distinctive abstract vocal portrait

G. abnormal
- disfluency and dysphonia are continuous speaking states
- treat them as performance directions, not as inserted bracket tags

H. implicit_nv
- instruction may describe nonverbal vocal events
- text must not contain explicit bracket tags

I. explicit_nv
- text must contain explicit bracketed nonverbal tags
- instruction must NOT mention the specific nonverbal events, event names, or bracket tags
- because the text already carries those events explicitly, the instruction should only describe global speaking manner, pacing, emotion, intensity, or delivery context

--------------------------------
OUTPUT FORMAT
--------------------------------
Return a JSON object with the following structure:

{{
  "results": [
    {{
      "plan_id": "<copied from input>",
      "meta_path": "<copied from input>",
      "target_values": {{
        "<attr1>": "<value1>"
      }},
      "candidates": [
        {{
          "candidate_id": "c1",
          "instruction": "<candidate instruction>",
          "text": "<candidate text>",
          "quality_checks": {{
            "matches_plan": true,
            "text_leaks_target": false,
            "contains_explicit_nonverbal_tag": <true|false>,
            "notes": "<short note>"
          }}
        }}
      ]
    }}
  ]
}}

Return JSON only.
"""


def get_language_profile(language: str) -> Dict[str, Any]:
    return LANGUAGE_PROFILES.get(language, LANGUAGE_PROFILES["en"])


def infer_instruction_form(plan_item: Dict[str, Any]) -> str:
    val = str(plan_item.get("instruction_form") or "").strip().lower()
    if val in {"tag", "direct"}:
        return val
    meta_path = str(plan_item.get("meta_path") or "")
    if "/tag/" in meta_path or meta_path.endswith("/tag"):
        return "tag"
    return "direct"


def build_instruction_text_prompt(plan_batch: Dict[str, Any], num_candidates: int) -> str:
    prompt = BASE_PROMPT.format(num_candidates=num_candidates)
    return f"{prompt}\n\nINPUT:\n{json.dumps(plan_batch, ensure_ascii=False, indent=2)}\n"


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


def validate_result_schema(result_json: Dict[str, Any], expected_count: int, num_candidates: int) -> None:
    if not isinstance(result_json, dict):
        raise ValueError("Returned result is not a dict")
    if "results" not in result_json:
        raise ValueError("Returned result is missing the results field")
    if not isinstance(result_json["results"], list):
        raise ValueError("results must be a list")
    if len(result_json["results"]) != expected_count:
        raise ValueError(f"Unexpected number of results: {len(result_json['results'])} != {expected_count}")

    for item in result_json["results"]:
        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"plan_id={item.get('plan_id')} is missing a candidates list")
        if len(candidates) != num_candidates:
            raise ValueError(
                f"plan_id={item.get('plan_id')} has an unexpected candidate count: {len(candidates)} != {num_candidates}"
            )


def enrich_plan_items_if_needed(plan_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in plan_items:
        copied = dict(item)
        meta_path = str(copied.get("meta_path") or "")
        if "ip_portrait" in meta_path and not isinstance(copied.get("aux_values"), dict):
            copied["aux_values"] = {}
        out.append(copied)
    return out


def enrich_results_from_plan(result_json: Dict[str, Any], plan_chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    plan_map = {x["plan_id"]: x for x in plan_chunk if "plan_id" in x}

    for item in result_json.get("results", []):
        plan_id = item.get("plan_id")
        if plan_id not in plan_map:
            continue

        plan_item = plan_map[plan_id]
        item.setdefault("meta_path", plan_item.get("meta_path", ""))
        item.setdefault("target_values", plan_item.get("target_values", {}))
        item.setdefault("instruction_form", infer_instruction_form(plan_item))

        candidates = item.get("candidates") or []
        for idx, cand in enumerate(candidates, start=1):
            if not isinstance(cand, dict):
                continue
            cand.setdefault("candidate_id", f"c{idx}")
            instruction = cand.get("instruction")
            if isinstance(instruction, str):
                cand["instruction"] = _normalize_tag_instruction_format(
                    instruction,
                    plan_item,
                    plan_map[plan_id].get("language", plan_item.get("language", "en")),
                )
            qc = cand.get("quality_checks")
            if not isinstance(qc, dict):
                cand["quality_checks"] = {
                    "matches_plan": True,
                    "text_leaks_target": False,
                    "contains_explicit_nonverbal_tag": False,
                    "notes": "",
                }

    return result_json


def _normalize_for_keyword_search(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _validate_language_policy(instruction: str, language: str) -> None:
    profile = get_language_profile(language)
    if profile["needs_english_instruction"]:
        if not instruction.strip().lower().startswith(profile["instruction_prefix"].lower()):
            raise ValueError(
                f"Non-primary-language instructions must start with '{profile['instruction_prefix']}': {instruction}"
            )


def _normalize_tag_instruction_format(instruction: str, plan_item: Dict[str, Any], language: str) -> str:
    if infer_instruction_form(plan_item) != "tag":
        return instruction

    profile = get_language_profile(language)
    prefix = profile["instruction_prefix"] if profile["needs_english_instruction"] else ""
    body = instruction.strip()

    if prefix and body.startswith(prefix):
        suffix = body[len(prefix):].lstrip()
        if language == "zh":
            suffix = suffix.replace(":", "：")
        else:
            suffix = suffix.replace("：", ":")
        return f"{prefix} {suffix}".strip()

    if language == "zh":
        return body.replace(":", "：")
    return body.replace("：", ":")


def _validate_tag_instruction_format(instruction: str, plan_item: Dict[str, Any], language: str) -> None:
    if infer_instruction_form(plan_item) != "tag":
        return

    profile = get_language_profile(language)
    colon = profile["tag_colon"]
    expected_pairs = max(1, len(plan_item.get("attribute_set") or []))

    body = _normalize_tag_instruction_format(instruction, plan_item, language)
    if profile["needs_english_instruction"]:
        prefix = profile["instruction_prefix"]
        if body.startswith(prefix):
            body = body[len(prefix):].strip()

    pairs = re.findall(rf"[^,;；\n]+{re.escape(colon)}\s*[^,;；\n]+", body)
    if len(pairs) < expected_pairs:
        raise ValueError(f"Tag instruction does not contain enough key-value pairs: {instruction}")


def _validate_nonverbal_constraints(instruction: str, text: str, plan_item: Dict[str, Any], language: str) -> None:
    meta_path = str(plan_item.get("meta_path") or "")

    if "special/extra_vocal/explicit_nv" in meta_path:
        if not BRACKET_TAG_RE.search(text):
            raise ValueError("explicit_nv text must contain an explicit bracketed nonverbal tag")
        if BRACKET_TAG_RE.search(instruction):
            raise ValueError("explicit_nv instruction must not contain an explicit bracket tag")

        normalized_instruction = _normalize_for_keyword_search(instruction)
        for value in (plan_item.get("target_values") or {}).values():
            if isinstance(value, str):
                token = value.lower().replace("_", " ").strip()
                if token and token in normalized_instruction:
                    raise ValueError(f"explicit_nv instruction must not mention the specific nonverbal event directly: {instruction}")



    if "special/extra_vocal/implicit_nv" in meta_path and BRACKET_TAG_RE.search(text):
        raise ValueError("implicit_nv text must not contain explicit bracketed nonverbal tags")


def _validate_ip_leak(instruction: str, text: str, plan_item: Dict[str, Any]) -> None:
    meta_path = str(plan_item.get("meta_path") or "")
    aux_values = plan_item.get("aux_values") or {}
    if "ip_portrait" not in meta_path or not isinstance(aux_values, dict):
        return

    forbidden: List[str] = []
    for key in ["franchise_or_title", "character_name"]:
        val = aux_values.get(key)
        if isinstance(val, str) and val.strip():
            forbidden.append(val.strip())

    for token in forbidden:
        if token and (token in instruction or token in text):
            raise ValueError(
                f"ip_portrait leaked an internal anchor: plan_id={plan_item.get('plan_id')}, leaked={token}"
            )


def _validate_one_candidate(candidate: Dict[str, Any], plan_item: Dict[str, Any], language: str) -> None:
    plan_id = plan_item.get("plan_id")
    instruction = str(candidate.get("instruction", "")).strip()
    text = str(candidate.get("text", "")).strip()

    if not instruction:
        raise ValueError(f"instruction is empty: {plan_id}")
    if not text:
        raise ValueError(f"text is empty: {plan_id}")

    # _validate_language_policy(instruction, language)
    _validate_tag_instruction_format(instruction, plan_item, language)
    _validate_nonverbal_constraints(instruction, text, plan_item, language)
    _validate_ip_leak(instruction, text, plan_item)


def validate_result_against_plan(
    result_json: Dict[str, Any],
    plan_chunk: List[Dict[str, Any]],
    language: str,
    expected_candidates_per_plan: int,
) -> None:
    if not isinstance(result_json, dict):
        raise ValueError("Returned result is not a dict")
    if "results" not in result_json:
        raise ValueError("Returned result is missing the results field")
    if not isinstance(result_json["results"], list):
        raise ValueError("results must be a list")

    plan_map = {x["plan_id"]: x for x in plan_chunk if "plan_id" in x}
    seen_plan_ids = set()

    for item in result_json["results"]:
        plan_id = item.get("plan_id")
        if plan_id not in plan_map:
            raise ValueError(f"Unknown plan_id: {plan_id}")
        if plan_id in seen_plan_ids:
            raise ValueError(f"Duplicate plan_id: {plan_id}")
        seen_plan_ids.add(plan_id)

        plan_item = plan_map[plan_id]
        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"plan_id={plan_id} is missing a candidates list")
        if len(candidates) != expected_candidates_per_plan:
            raise ValueError(
                f"plan_id={plan_id} has an unexpected candidate count: {len(candidates)} != {expected_candidates_per_plan}"
            )

        for idx, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise ValueError(f"plan_id={plan_id} candidate {idx} is not a dict")
            _validate_one_candidate(candidate, plan_item, language)

    missing = set(plan_map.keys()) - seen_plan_ids
    if missing:
        raise ValueError(f"Missing plan_id outputs: {sorted(missing)}")


def _select_candidate_index(plan_id: str, taxonomy_node: str, candidate_count: int, selection_seed: Optional[int]) -> int:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be greater than 0")
    if selection_seed is None:
        return random.randrange(candidate_count)

    payload = f"{selection_seed}::{taxonomy_node}::{plan_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest, 16) % candidate_count


def collapse_candidate_results(
    result_json: Dict[str, Any],
    taxonomy_node: str,
    selection_seed: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    final_results: List[Dict[str, Any]] = []
    candidate_pools: List[Dict[str, Any]] = []

    for item in result_json.get("results", []):
        plan_id = str(item.get("plan_id", ""))
        candidates = item.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"plan_id={plan_id} is missing available candidates")

        selected_index = _select_candidate_index(
            plan_id=plan_id,
            taxonomy_node=taxonomy_node,
            candidate_count=len(candidates),
            selection_seed=selection_seed,
        )
        selected = dict(candidates[selected_index])
        selected["plan_id"] = plan_id
        selected["meta_path"] = item.get("meta_path", "")
        selected["target_values"] = item.get("target_values", {})
        selected["selected_candidate_index"] = selected_index
        selected["selected_candidate_id"] = candidates[selected_index].get("candidate_id", f"c{selected_index + 1}")
        final_results.append(selected)

        candidate_pools.append({
            "plan_id": plan_id,
            "meta_path": item.get("meta_path", ""),
            "target_values": item.get("target_values", {}),
            "selected_candidate_index": selected_index,
            "selected_candidate_id": candidates[selected_index].get("candidate_id", f"c{selected_index + 1}"),
            "candidates": candidates,
        })

    return final_results, candidate_pools


async def generate_instruction_text(
    session: aiohttp.ClientSession,
    url: str,
    key: str,
    prompt_text: str,
    timeout: int = 360,
    max_retries: int = 3,
) -> Tuple[Optional[Any], Optional[str]]:
    api_url_base = f"{url}/v1beta/models/gemini-2.5-pro:generateContent"
    full_api_url = f"{api_url_base}?key={key}"

    request_data = {"contents": [{"parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(
                full_api_url,
                headers=headers,
                json=request_data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    last_error = f"HTTP error: {resp.status}, body: {text}"
                    continue

                data = await resp.json()
                try:
                    inner_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    response_json = extract_json_from_text(inner_text)
                except Exception as e:
                    last_error = (
                        f"Failed to parse JSON from Gemini response: {e}; raw response: "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    )
                    continue

                return response_json, None

        except asyncio.TimeoutError:
            last_error = f"Request timed out (attempt {attempt})"
        except aiohttp.ClientError as e:
            last_error = f"Network error (attempt {attempt}): {e}"
        except Exception as e:
            last_error = f"Unexpected error (attempt {attempt}): {e}"

        if attempt < max_retries:
            await asyncio.sleep(1.5 * attempt)

    return None, last_error


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def process_one_plan_file(
    plan_file: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    key: str,
    output_dir: str,
    error_log: str,
    batch_size: int = 10,
    num_candidates_per_plan: int = DEFAULT_NUM_CANDIDATES,
    selection_seed: Optional[int] = None,
    max_chunk_attempts: int = DEFAULT_MAX_CHUNK_ATTEMPTS,
) -> Dict[str, Any]:
    async with semaphore:
        try:
            async with aiofiles.open(plan_file, "r", encoding="utf-8") as f:
                content = await f.read()
            plan_data = json.loads(content)

            taxonomy_node = plan_data.get("taxonomy_node", "")
            language = plan_data.get("language", "zh")
            items = plan_data.get("items", [])

            if not taxonomy_node:
                raise ValueError(f"Plan file is missing taxonomy_node: {plan_file}")
            if not isinstance(items, list) or not items:
                raise ValueError(f"Plan file is missing items or items is empty: {plan_file}")

            chunks = chunk_list(items, batch_size)
            all_selected_results: List[Dict[str, Any]] = []
            all_candidate_pools: List[Dict[str, Any]] = []

            for chunk_idx, chunk in enumerate(chunks):
                chunk = enrich_plan_items_if_needed(chunk)
                batch_payload = {
                    "taxonomy_node": taxonomy_node,
                    "language": language,
                    "num_candidates_per_plan": num_candidates_per_plan,
                    "items": chunk,
                }
                prompt_text = build_instruction_text_prompt(batch_payload, num_candidates_per_plan)

                validated_result: Optional[Dict[str, Any]] = None
                last_error: Optional[str] = None

                for attempt in range(1, max_chunk_attempts + 1):
                    result_json, error = await generate_instruction_text(
                        session=session,
                        url=url,
                        key=key,
                        prompt_text=prompt_text,
                    )
                    if error:
                        last_error = error
                        continue

                    try:
                        validate_result_schema(
                            result_json,
                            expected_count=len(chunk),
                            num_candidates=num_candidates_per_plan,
                        )
                        result_json = enrich_results_from_plan(result_json, chunk)
                        validate_result_against_plan(
                            result_json,
                            chunk,
                            language=language,
                            expected_candidates_per_plan=num_candidates_per_plan,
                        )
                        validated_result = result_json
                        break
                    except Exception as e:
                        last_error = str(e)

                if validated_result is None:
                    raise RuntimeError(f"Batch {chunk_idx} failed to generate: {last_error}")

                selected_results, candidate_pools = collapse_candidate_results(
                    validated_result,
                    taxonomy_node=taxonomy_node,
                    selection_seed=selection_seed,
                )
                all_selected_results.extend(selected_results)
                all_candidate_pools.extend(candidate_pools)

            output_data = {
                "taxonomy_node": taxonomy_node,
                "language": language,
                "source_plan_file": os.path.basename(plan_file),
                "num_input_items": len(items),
                "num_output_items": len(all_selected_results),
                "num_candidates_per_plan": num_candidates_per_plan,
                "selection_seed": selection_seed,
                "results": all_selected_results,
                "candidate_pools": all_candidate_pools,
            }

            output_file = os.path.join(output_dir, os.path.basename(plan_file))
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(output_data, ensure_ascii=False, indent=2))

            return {
                "taxonomy_node": taxonomy_node,
                "output_file": output_file,
                "num_input_items": len(items),
                "num_output_items": len(all_selected_results),
            }

        except Exception as e:
            async with aiofiles.open(error_log, "a", encoding="utf-8") as ferr:
                await ferr.write(f"ERROR {plan_file}: {str(e)}\n")
            raise


async def process_all_plan_files(
    input_path: str,
    output_dir: str,
    error_log_path: str,
    url: str,
    key: str,
    max_concurrent: int = 5,
    batch_size: int = 10,
    num_candidates_per_plan: int = DEFAULT_NUM_CANDIDATES,
    selection_seed: Optional[int] = None,
    max_chunk_attempts: int = DEFAULT_MAX_CHUNK_ATTEMPTS,
    languages: Optional[Sequence[str]] = None,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("")

    if os.path.isfile(input_path):
        plan_files = [input_path]
    elif os.path.isdir(input_path):
        plan_files = []
        for fname in os.listdir(input_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(input_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            if should_keep_plan_file(
                plan_data,
                file_name=fname,
                languages=languages,
                taxonomy_nodes=taxonomy_nodes,
                taxonomy_prefixes=taxonomy_prefixes,
            ):
                plan_files.append(fpath)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    existing = set(os.listdir(output_dir))
    pending_files = [fp for fp in plan_files if os.path.basename(fp) not in existing]

    print(f"Total plan files: {len(plan_files)}")
    print(f"Pending plan files: {len(pending_files)}")
    print("[INFO] Recommended workflow: run one stage at a time and inspect logs/intermediate outputs before proceeding.")

    if not pending_files:
        print("No pending plan files to process.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                process_one_plan_file(
                    plan_file=plan_file,
                    session=session,
                    semaphore=semaphore,
                    url=url,
                    key=key,
                    output_dir=output_dir,
                    error_log=error_log_path,
                    batch_size=batch_size,
                    num_candidates_per_plan=num_candidates_per_plan,
                    selection_seed=selection_seed,
                    max_chunk_attempts=max_chunk_attempts,
                )
            )
            for plan_file in pending_files
        ]

        success_count = 0
        error_count = 0

        pbar = tqdm(total=len(tasks), desc="Generating instruction-text pairs")
        for task in asyncio.as_completed(tasks):
            try:
                await task
                success_count += 1
            except Exception:
                error_count += 1
            pbar.set_postfix_str(f"success={success_count}, error={error_count}")
            pbar.update(1)
        pbar.close()

    print(f"\nCompleted: success={success_count}, error={error_count}, total={len(pending_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Error log: {error_log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert structured label plans into instruction-text pair JSON files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to an input plan JSON file or a directory of plan files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output result JSON files")
    parser.add_argument("--error_log", type=str, required=True, help="Path to the error log file")
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key. Falls back to MINT_BENCH_API_KEY or GOOGLE_API_KEY when omitted.",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="",
        help="Gemini-compatible API base URL. Falls back to MINT_BENCH_API_URL when omitted.",
    )
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent file-processing tasks")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of plan items per API request")
    parser.add_argument(
        "--num_candidates_per_plan",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help="Number of candidates to generate per plan item",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Optional seed for deterministic candidate selection",
    )
    parser.add_argument(
        "--max_chunk_attempts",
        type=int,
        default=DEFAULT_MAX_CHUNK_ATTEMPTS,
        help="Maximum number of validation retries per chunk",
    )
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
        process_all_plan_files(
            input_path=args.input_path,
            output_dir=args.output_dir,
            error_log_path=args.error_log,
            url=resolved_url,
            key=resolved_key,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            num_candidates_per_plan=args.num_candidates_per_plan,
            selection_seed=args.selection_seed,
            max_chunk_attempts=args.max_chunk_attempts,
            languages=args.language,
            taxonomy_nodes=args.taxonomy_node,
            taxonomy_prefixes=args.taxonomy_prefix,
        )
    )
