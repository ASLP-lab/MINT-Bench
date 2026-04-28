import asyncio
import aiohttp
import aiofiles
import json
import base64
import mimetypes
import os
import argparse
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm.asyncio import tqdm


# =========================
# JSON extraction
# =========================
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output text.
    Supports:
    - pure JSON
    - ```json ... ```
    - surrounding explanation text
    """
    text = text.strip()

    # 1) direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) markdown code block
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass

    # 3) extract the outermost object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass

    return None


# =========================
# Stable unique ID
# =========================
def build_uid(item: Dict[str, Any]) -> str:
    taxonomy_node = item.get("taxonomy_node", "UNK")
    plan_id = item.get("plan_id", "UNK")
    wav_path = item.get("wav_path", "")
    wav_name = os.path.basename(wav_path) if wav_path else "NOWAV"
    return f"{taxonomy_node}::{plan_id}::{wav_name}"


# =========================
# Prompt selection and construction
# =========================
def select_template_by_taxonomy(
    item: Dict[str, Any],
    easy_template: str,
    hard_template: str,
    special_template: str,
) -> str:
    taxonomy_node = item.get("taxonomy_node", "")
    if taxonomy_node.startswith("easy/"):
        return easy_template
    if taxonomy_node.startswith("hard/"):
        return hard_template
    if taxonomy_node.startswith("special/"):
        return special_template
    raise ValueError(f"Unknown taxonomy_node prefix: {taxonomy_node}")


def build_eval_prompt(template: str, item: Dict[str, Any]) -> str:
    """
    Fill one sample into the prompt template.
    The template must contain the placeholder:
      <INPUT_EVAL_JSON>
    """
    payload = {
        "taxonomy_node": item.get("taxonomy_node", ""),
        "plan_id": item.get("plan_id", ""),
        "language": item.get("language", ""),
        "instruction": item.get("instruction", ""),
        "text": item.get("text", ""),
        "target_values": item.get("target_values", {}),
    }
    return template.replace("<INPUT_EVAL_JSON>", json.dumps(payload, ensure_ascii=False, indent=2))


# =========================
# API call
# =========================
async def analyze_audio_with_text(
    session: aiohttp.ClientSession,
    url: str,
    key: str,
    audio_path: str,
    prompt_text: str,
    timeout: int = 360,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Read one audio file asynchronously and call the API.
    Returns:
      response_json, usage, error
    """
    api_url_base = f"{url}/v1beta/models/gemini-3.1-pro-preview:generateContent"
    full_api_url = f"{api_url_base}?key={key}"

    try:
        async with aiofiles.open(audio_path, "rb") as f:
            audio_bytes = await f.read()
    except FileNotFoundError:
        return None, None, f"Audio file not found: {audio_path}"
    except Exception as e:
        return None, None, f"Failed to read audio file: {e}"

    base64_encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/wav"

    request_data = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": mime_type, "data": base64_encoded_audio}}
            ]
        }],
        "config": {
            "temperature": 0.0,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }
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
                return None, None, f"HTTP error {resp.status}: {text}"

            data = await resp.json()

            try:
                response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                usage = data.get("usageMetadata", {})
            except Exception:
                response_text = json.dumps(data, ensure_ascii=False)
                usage = {}

            response_json = extract_json_from_text(response_text)
            if response_json is None:
                return None, usage, f"Failed to parse JSON from model output: {response_text[:1000]}"

            return response_json, usage, None

    except asyncio.TimeoutError:
        return None, None, "Request timed out"
    except aiohttp.ClientError as e:
        return None, None, f"Network error: {e}"
    except Exception as e:
        return None, None, f"Unexpected error: {e}"


# =========================
# Output schema validation
# =========================
def validate_eval_result_schema(result_json: Dict[str, Any]) -> None:
    required_keys = [
        "plan_id",
        "taxonomy_node",
        "instruction_following_level",
        "instruction_following_label",
        "naturalness_bonus",
        "expressiveness_bonus",
        "final_score",
        "brief_reason",
        "instruction_evidence",
        "bonus_evidence",
    ]
    for k in required_keys:
        if k not in result_json:
            raise ValueError(f"Missing required field: {k}")

    if result_json["instruction_following_level"] not in {1, 2, 3}:
        raise ValueError("instruction_following_level must be one of: 1, 2, 3")

    if result_json["instruction_following_label"] not in {"bad", "so-so", "good"}:
        raise ValueError("instruction_following_label has an invalid value")

    if not isinstance(result_json["naturalness_bonus"], bool):
        raise ValueError("naturalness_bonus must be a bool")

    if not isinstance(result_json["expressiveness_bonus"], bool):
        raise ValueError("expressiveness_bonus must be a bool")

    if result_json["final_score"] not in {1, 2, 3, 4, 5}:
        raise ValueError("final_score must be in the range 1..5")

    if not isinstance(result_json["instruction_evidence"], dict):
        raise ValueError("instruction_evidence must be a dict")

    if not isinstance(result_json["bonus_evidence"], dict):
        raise ValueError("bonus_evidence must be a dict")

    if_level = result_json["instruction_following_level"]
    nat_bonus = result_json["naturalness_bonus"]
    exp_bonus = result_json["expressiveness_bonus"]
    final_score = result_json["final_score"]

    if if_level < 3 and (nat_bonus or exp_bonus):
        raise ValueError("bonus fields must both be false when instruction_following_level < 3")

    expected_final = if_level + int(nat_bonus) + int(exp_bonus)
    if final_score != expected_final:
        raise ValueError(
            f"final_score mismatch: expected {expected_final}, got {final_score}"
        )


def is_valid_completed_result(item: Dict[str, Any]) -> bool:
    eval_result = item.get("lalm_eval")
    if not isinstance(eval_result, dict):
        return False
    try:
        validate_eval_result_schema(eval_result)
        return True
    except Exception:
        return False


# =========================
# Single task
# =========================
async def process_single_score_task(
    item: Dict[str, Any],
    easy_template: str,
    hard_template: str,
    special_template: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    key: str,
) -> Dict[str, Any]:
    result = item.copy()
    uid = build_uid(item)
    result["uid"] = uid

    wav_path = item.get("wav_path", "")
    if not wav_path or not os.path.isfile(wav_path):
        result["lalm_eval"] = None
        result["error"] = f"Audio file not found: {wav_path}"
        return result

    try:
        template = select_template_by_taxonomy(
            item=item,
            easy_template=easy_template,
            hard_template=hard_template,
            special_template=special_template,
        )
    except Exception as e:
        result["lalm_eval"] = None
        result["error"] = f"Failed to select prompt template: {e}"
        return result

    prompt_text = build_eval_prompt(template, item)

    async with semaphore:
        eval_json, usage, error = await analyze_audio_with_text(
            session=session,
            url=url,
            key=key,
            audio_path=wav_path,
            prompt_text=prompt_text,
        )

    if error:
        result["lalm_eval"] = None
        result["error"] = error
        logging.error(f"ERROR {uid}: {error}")
    else:
        try:
            validate_eval_result_schema(eval_json)
            result["lalm_eval"] = eval_json
            result["usage"] = usage
            result.pop("error", None)
            result.pop("raw_eval_json", None)
            logging.info(
                f"SUCCESS {uid}: final_score={eval_json.get('final_score')}, "
                f"if_level={eval_json.get('instruction_following_level')}, "
                f"if_label={eval_json.get('instruction_following_label')}, "
                f"nat_bonus={eval_json.get('naturalness_bonus')}, "
                f"exp_bonus={eval_json.get('expressiveness_bonus')}"
            )
        except Exception as e:
            result["lalm_eval"] = None
            result["error"] = f"Output schema validation failed: {e}"
            result["raw_eval_json"] = eval_json
            logging.error(f"SCHEMA_FAIL {uid}: {e}")

    return result


# =========================
# JSONL read/write
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_jsonl_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


# =========================
# Statistics
# =========================
def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "total": 0,
        "valid": 0,
        "null": 0,
        "avg_final_score": None,
        "avg_instruction_following_level": None,
        "instruction_following_label": {
            "bad": 0,
            "so-so": 0,
            "good": 0,
        },
        "bonus": {
            "naturalness_true": 0,
            "expressiveness_true": 0,
        },
        "per_taxonomy_node": {}
    }

    final_score_sum = 0.0
    if_level_sum = 0.0

    for item in results:
        stats["total"] += 1
        eval_result = item.get("lalm_eval")
        taxonomy_node = item.get("taxonomy_node", "UNK")

        if taxonomy_node not in stats["per_taxonomy_node"]:
            stats["per_taxonomy_node"][taxonomy_node] = {
                "total": 0,
                "valid": 0,
                "final_score_sum": 0.0,
                "if_level_sum": 0.0,
                "avg_final_score": None,
                "avg_instruction_following_level": None,
                "label_counts": {
                    "bad": 0,
                    "so-so": 0,
                    "good": 0,
                },
                "bonus": {
                    "naturalness_true": 0,
                    "expressiveness_true": 0,
                },
            }

        node_stat = stats["per_taxonomy_node"][taxonomy_node]
        node_stat["total"] += 1

        if not eval_result:
            stats["null"] += 1
            continue

        stats["valid"] += 1

        final_score = float(eval_result.get("final_score", 0.0))
        if_level = float(eval_result.get("instruction_following_level", 0.0))
        if_label = eval_result.get("instruction_following_label", "so-so")
        nat_bonus = bool(eval_result.get("naturalness_bonus", False))
        exp_bonus = bool(eval_result.get("expressiveness_bonus", False))

        final_score_sum += final_score
        if_level_sum += if_level

        node_stat["valid"] += 1
        node_stat["final_score_sum"] += final_score
        node_stat["if_level_sum"] += if_level

        if if_label in stats["instruction_following_label"]:
            stats["instruction_following_label"][if_label] += 1
            node_stat["label_counts"][if_label] += 1

        if nat_bonus:
            stats["bonus"]["naturalness_true"] += 1
            node_stat["bonus"]["naturalness_true"] += 1
        if exp_bonus:
            stats["bonus"]["expressiveness_true"] += 1
            node_stat["bonus"]["expressiveness_true"] += 1

    if stats["valid"] > 0:
        stats["avg_final_score"] = final_score_sum / stats["valid"]
        stats["avg_instruction_following_level"] = if_level_sum / stats["valid"]

    for node, node_stat in stats["per_taxonomy_node"].items():
        if node_stat["valid"] > 0:
            node_stat["avg_final_score"] = node_stat["final_score_sum"] / node_stat["valid"]
            node_stat["avg_instruction_following_level"] = node_stat["if_level_sum"] / node_stat["valid"]

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("LALM EVALUATION STATISTICS")
    print("=" * 72)
    print(f"Total samples                    : {stats['total']}")
    print(f"Valid evals                      : {stats['valid']}")
    print(f"Null / failed                    : {stats['null']}")
    print(f"Avg final_score                  : {stats['avg_final_score']}")
    print(f"Avg instruction_following_level  : {stats['avg_instruction_following_level']}")
    print(
        f"Instruction-following label      : "
        f"bad={stats['instruction_following_label']['bad']}, "
        f"so-so={stats['instruction_following_label']['so-so']}, "
        f"good={stats['instruction_following_label']['good']}"
    )
    print(
        f"Bonus hit count                  : "
        f"naturalness={stats['bonus']['naturalness_true']}, "
        f"expressiveness={stats['bonus']['expressiveness_true']}"
    )

    print("\nPer taxonomy node:")
    for node, node_stat in sorted(stats["per_taxonomy_node"].items()):
        print(
            f"  {node}: "
            f"final={node_stat['avg_final_score']}, "
            f"if={node_stat['avg_instruction_following_level']}, "
            f"labels={node_stat['label_counts']}, "
            f"bonus={node_stat['bonus']} "
            f"({node_stat['valid']}/{node_stat['total']} valid)"
        )
    print("=" * 72)


# =========================
# Main flow
# =========================
def infer_difficulty_from_payload_path(payload_path: str) -> str:
    stem = Path(payload_path).stem
    return stem.split("__", 1)[0]


def collect_payload_files(input_path: str, difficulty: str) -> List[str]:
    # The 'timbre' difficulty uses payload filenames without '__' prefix
    # (e.g. <model>_timbre_candidates_<lang>.jsonl). Skip filename-prefix
    # validation; taxonomy_node inside payloads still drives prompt routing.
    if difficulty == "timbre":
        if os.path.isfile(input_path):
            return [input_path]
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path does not exist: {input_path}")
        return [
            os.path.join(input_path, entry)
            for entry in sorted(os.listdir(input_path))
            if entry.endswith(".jsonl")
        ]

    if os.path.isfile(input_path):
        actual_difficulty = infer_difficulty_from_payload_path(input_path)
        if actual_difficulty != difficulty:
            raise ValueError(
                f"This evaluator run is configured for '{difficulty}' payloads, but got '{Path(input_path).name}'."
            )
        return [input_path]

    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    payload_files: List[str] = []
    for entry in sorted(os.listdir(input_path)):
        if not entry.endswith(".jsonl"):
            continue
        if entry.split("__", 1)[0] != difficulty:
            continue
        payload_files.append(os.path.join(input_path, entry))
    return payload_files


def derive_output_jsonl(results_dir: str, input_jsonl: str, difficulty: str) -> str:
    input_abs = Path(input_jsonl).resolve()
    payload_parent = input_abs.parent.name
    payload_stem = input_abs.stem
    if difficulty == "timbre":
        return str(Path(results_dir) / f"{payload_stem}.jsonl")
    else:
        return str(Path(results_dir) / payload_parent / f"{payload_stem}.jsonl")


def derive_log_file(logs_dir: str, input_jsonl: str) -> str:
    input_abs = Path(input_jsonl).resolve()
    payload_parent = input_abs.parent.name
    payload_stem = input_abs.stem
    return str(Path(logs_dir) / f"{payload_parent}__{payload_stem}.log")


async def process_one_payload_file(
    args,
    input_jsonl: str,
    output_jsonl: str,
    log_file: str,
    easy_template: str,
    hard_template: str,
    special_template: str,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",
        force=True,
    )

    semaphore = asyncio.Semaphore(args.num_workers)
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    print(f"Processing payload file: {input_jsonl}")
    print(f"Found {len(items)} payload entries")

    old_results = load_jsonl(output_jsonl)
    old_results_by_uid: Dict[str, Dict[str, Any]] = {}
    for data in old_results:
        try:
            uid = data.get("uid") or build_uid(data)
            data["uid"] = uid
            old_results_by_uid[uid] = data
        except Exception:
            continue

    # Keep valid existing results first
    final_results_by_uid: Dict[str, Dict[str, Any]] = {}
    for uid, data in old_results_by_uid.items():
        if is_valid_completed_result(data):
            final_results_by_uid[uid] = data

    # Rerun everything else: missing, null lalm_eval, or schema-invalid entries
    items_to_process: List[Dict[str, Any]] = []
    for item in items:
        uid = build_uid(item)
        if uid in final_results_by_uid:
            continue
        items_to_process.append(item)

    print(f"Kept {len(final_results_by_uid)} valid existing results; rerunning {len(items_to_process)} entries.")

    rerun_results: List[Dict[str, Any]] = []
    if items_to_process:
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    process_single_score_task(
                        item=item,
                        easy_template=easy_template,
                        hard_template=hard_template,
                        special_template=special_template,
                        session=session,
                        semaphore=semaphore,
                        url=args.api_url,
                        key=args.api_key,
                    )
                )
                for item in items_to_process
            ]

            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scoring progress"):
                try:
                    result = await task
                    rerun_results.append(result)
                except Exception as e:
                    logging.error(f"Task execution failed: {e}")

    # Overwrite old results with the newest results
    for result in rerun_results:
        uid = result.get("uid") or build_uid(result)
        result["uid"] = uid
        final_results_by_uid[uid] = result

    # Export final results in input order and overwrite the output file
    final_results: List[Dict[str, Any]] = []
    for item in items:
        uid = build_uid(item)
        if uid in final_results_by_uid:
            final_results.append(final_results_by_uid[uid])

    write_jsonl_atomic(output_jsonl, final_results)

    stats = calculate_statistics(final_results)
    print_statistics(stats)


async def process_all_scores(
    args,
    easy_template: str,
    hard_template: str,
    special_template: str,
):
    payload_files = collect_payload_files(args.input_path, args.difficulty)
    if not payload_files:
        raise ValueError(f"No '{args.difficulty}' payload JSONL files found under: {args.input_path}")

    for input_jsonl in payload_files:
        output_jsonl = derive_output_jsonl(args.results_dir, input_jsonl, args.difficulty)
        log_file = derive_log_file(args.logs_dir, input_jsonl)
        await process_one_payload_file(
            args=args,
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            log_file=log_file,
            easy_template=easy_template,
            hard_template=hard_template,
            special_template=special_template,
        )


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Use a Gemini-style LALM API to score instruction-following for TTS audio")
    p.add_argument("--input_path", required=True, help="Input payload JSONL path, or a directory containing payload JSONL files")
    p.add_argument("--difficulty", required=True, choices=["easy", "hard", "special", "timbre"], help="Difficulty group to evaluate")
    p.add_argument("--results_dir", default="results", help="Directory for result JSONL files")
    p.add_argument("--logs_dir", default="results/logs", help="Directory for evaluator log files")
    p.add_argument("--easy_prompt_file", required=True, help="Prompt template file for easy nodes")
    p.add_argument("--hard_prompt_file", required=True, help="Prompt template file for hard nodes")
    p.add_argument("--special_prompt_file", required=True, help="Prompt template file for special nodes")
    p.add_argument("--api_key", required=True, help="API key")
    p.add_argument("--api_url", default="https://generativelanguage.googleapis.com", help="API base URL")
    p.add_argument("--num_workers", type=int, default=5, help="Maximum number of concurrent requests")
    return p.parse_args()


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()

    easy_template = load_template(args.easy_prompt_file)
    hard_template = load_template(args.hard_prompt_file)
    special_template = load_template(args.special_prompt_file)

    for name, template in [
        ("easy", easy_template),
        ("hard", hard_template),
        ("special", special_template),
    ]:
        if "<INPUT_EVAL_JSON>" not in template:
            raise ValueError(f"The {name} prompt template must contain the placeholder <INPUT_EVAL_JSON>")

    try:
        asyncio.run(process_all_scores(
            args,
            easy_template=easy_template,
            hard_template=hard_template,
            special_template=special_template,
        ))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        logging.critical(f"Fatal evaluator error: {e}")
        print(f"\nFatal evaluator error: {e}")


if __name__ == "__main__":
    main()
