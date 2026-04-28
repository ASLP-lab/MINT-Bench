import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm


FEMALE_VOICES = [
    "Achernar", "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome",
    "Gacrux", "Kore", "Laomedeia", "Leda", "Pulcherrima", "Sulafat",
    "Vindemiatrix", "Zephyr",
]

MALE_VOICES = [
    "Achird", "Algenib", "Algieba", "Alnilam", "Charon", "Enceladus",
    "Fenrir", "Iapetus", "Orus", "Puck", "Rasalgethi", "Sadachbia",
    "Sadaltager", "Schedar", "Umbriel", "Zubenelgenubi",
]

ALL_VOICES = FEMALE_VOICES + MALE_VOICES

LANGUAGE_DISPLAY = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "es": "Spanish",
}

WAV_HEADER_BYTES = 44


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "__", str(name).strip("/"))


def wave_file(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def load_node_whitelist(path: str) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("The node whitelist file must contain a JSON list.")


def is_pair_json(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and isinstance(data.get("results"), list)


def collect_json_files(input_path: str, recursive: bool = False) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    def should_keep_file(fname: str) -> bool:
        if not fname.endswith(".json"):
            return False
        return not fname.startswith(("manifest", "summary"))

    if recursive:
        json_files: List[str] = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if should_keep_file(fname):
                    json_files.append(os.path.join(root, fname))
        return sorted(json_files)

    return [
        os.path.join(input_path, fname)
        for fname in sorted(os.listdir(input_path))
        if should_keep_file(fname)
    ]


def normalize_gender_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for x in value:
            norm = normalize_gender_value(x)
            if norm:
                return norm
        return None

    text = str(value).strip().lower().replace("_", " ")
    female_aliases = {
        "female", "woman", "women", "girl", "feminine", "女", "女性", "女声", "女生", "女孩"
    }
    male_aliases = {
        "male", "man", "men", "boy", "masculine", "男", "男性", "男声", "男生", "男孩"
    }
    if text in female_aliases:
        return "female"
    if text in male_aliases:
        return "male"
    return None


def infer_gender_from_target_values(target_values: Dict[str, Any]) -> Optional[str]:
    if not isinstance(target_values, dict):
        return None

    if "gender" in target_values:
        return normalize_gender_value(target_values.get("gender"))

    for key, value in target_values.items():
        if str(key).strip().lower() == "gender":
            norm = normalize_gender_value(value)
            if norm:
                return norm

    return None


def choose_voice_name(
    target_values: Dict[str, Any],
    uniq_name: str,
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    gender = infer_gender_from_target_values(target_values)
    if gender == "female":
        pool = FEMALE_VOICES
        pool_name = "female"
    elif gender == "male":
        pool = MALE_VOICES
        pool_name = "male"
    else:
        pool = ALL_VOICES
        pool_name = "all"

    if seed is None:
        return random.choice(pool), pool_name

    payload = f"{seed}::{uniq_name}::{pool_name}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    idx = int(digest, 16) % len(pool)
    return pool[idx], pool_name


def build_gemini_input(instruction: str, text: str, language_code: str) -> str:
    language_name = LANGUAGE_DISPLAY.get(language_code, language_code)
    return (
        "Follow the speaking instruction and synthesize the target text exactly as written. "
        "Do not add or omit words.\n\n"
        f"Speaking instruction:\n{instruction}\n\n"
        f"Target language: {language_name}\n"
        f"Text to speak:\n{text}"
    )


def load_all_instruction_items(
    input_path: str,
    node_whitelist: Optional[List[str]] = None,
    recursive: bool = False,
    use_plan_id_as_name: bool = True,
) -> List[Dict[str, Any]]:
    node_whitelist = node_whitelist or []
    items: List[Dict[str, Any]] = []

    json_files = collect_json_files(input_path, recursive=recursive)

    for fpath in json_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not is_pair_json(data):
            continue

        taxonomy_node = data.get("taxonomy_node", os.path.splitext(fname)[0].replace("__", "/"))
        language = data.get("language", "zh")

        if node_whitelist and taxonomy_node not in node_whitelist:
            continue

        results = data.get("results", [])
        if not isinstance(results, list):
            continue

        for obj in results:
            if not isinstance(obj, dict):
                continue

            plan_id = str(obj.get("plan_id", "UNKNOWN")).strip() or "UNKNOWN"
            meta_path = obj.get("meta_path", taxonomy_node)
            instruction = str(obj.get("instruction", "")).strip()
            text = str(obj.get("text", "")).strip()
            target_values = obj.get("target_values", {})
            quality_checks = obj.get("quality_checks", {})
            selected_candidate_index = obj.get("selected_candidate_index")
            selected_candidate_id = obj.get("selected_candidate_id")

            if not instruction or not text:
                continue

            node_slug = sanitize_filename(taxonomy_node)
            uniq_name = sanitize_filename(plan_id) if use_plan_id_as_name else f"{node_slug}__{sanitize_filename(plan_id)}"

            items.append(
                {
                    "file_name": fname,
                    "source_json_path": fpath,
                    "taxonomy_node": taxonomy_node,
                    "node_slug": node_slug,
                    "plan_id": plan_id,
                    "meta_path": meta_path,
                    "instruction": instruction,
                    "text": text,
                    "language": language,
                    "target_values": target_values,
                    "quality_checks": quality_checks,
                    "selected_candidate_index": selected_candidate_index,
                    "selected_candidate_id": selected_candidate_id,
                    "uniq_name": uniq_name,
                }
            )

    return items


def artifact_name(prefix: str, input_path: str) -> str:
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return f"{prefix}__{sanitize_filename(stem)}.json"
    return f"{prefix}.json"


def save_manifest(items: List[Dict[str, Any]], output_dir: str, input_path: str) -> str:
    manifest_path = os.path.join(output_dir, artifact_name("manifest", input_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return manifest_path


def save_summary(summary: Dict[str, Any], output_dir: str, input_path: str) -> str:
    summary_path = os.path.join(output_dir, artifact_name("summary", input_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def to_evaluation_relative_path(path: str) -> str:
    evaluation_dir = Path(__file__).resolve().parents[2] / "evaluation"
    return os.path.relpath(path, evaluation_dir).replace(os.sep, "/")


def is_audio_complete(path: str, audio_format: str) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if audio_format.lower() == "wav":
        return size > WAV_HEADER_BYTES
    return size > 0


async def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def append_error(path: str, payload: Dict[str, Any]) -> None:
    await append_jsonl(path, payload)


async def write_meta_json(path: str, payload: Dict[str, Any]) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, ensure_ascii=False, indent=2))


async def generate_gemini_tts(
    session: aiohttp.ClientSession,
    url: str,
    key: str,
    model_name: str,
    prompt_text: str,
    voice_name: str,
    timeout: int = 360,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[str]]:
    api_url_base = f"{url}/v1beta/models/{model_name}:generateContent"
    full_api_url = f"{api_url_base}?key={key}"

    shared_config = {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": voice_name,
                }
            }
        },
    }

    payloads = [
        {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "config": shared_config,
        },
        {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": shared_config,
        },
    ]

    headers = {"Content-Type": "application/json"}
    last_error: Optional[str] = None

    for payload in payloads:
        try:
            async with session.post(
                full_api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    last_error = f"HTTP error {resp.status}: {text}"
                    continue

                data = await resp.json()
                usage = data.get("usageMetadata", {})

                try:
                    part = data["candidates"][0]["content"][0]["parts"][0]
                except Exception:
                    try:
                        part = data["candidates"][0]["content"]["parts"][0]
                    except Exception:
                        return None, usage, f"Unexpected response structure: {json.dumps(data, ensure_ascii=False)[:1000]}"

                inline_data = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline_data, dict):
                    return None, usage, f"Missing inline audio data in response: {json.dumps(part, ensure_ascii=False)[:1000]}"

                encoded_audio = inline_data.get("data")
                if not encoded_audio:
                    return None, usage, f"inline audio data is empty: {json.dumps(inline_data, ensure_ascii=False)[:1000]}"

                try:
                    pcm_bytes = base64.b64decode(encoded_audio)
                except Exception as e:
                    return None, usage, f"Failed to decode base64 audio payload: {e}"

                return pcm_bytes, usage, None

        except asyncio.TimeoutError:
            last_error = "Request timed out."
        except aiohttp.ClientError as e:
            last_error = f"Network error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {e}"

    return None, None, last_error or "Unknown error."


async def process_single_item(
    item: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    out_wav = os.path.join(args.output_dir, f"{item['uniq_name']}.wav")
    out_meta = os.path.join(args.output_dir, f"{item['uniq_name']}.json")
    relative_wav_path = to_evaluation_relative_path(out_wav)
    relative_source_json_path = to_evaluation_relative_path(item["source_json_path"])

    result_meta: Dict[str, Any] = {
        "status": "pending",
        "provider": "gemini",
        "taxonomy_node": item["taxonomy_node"],
        "plan_id": item["plan_id"],
        "meta_path": item["meta_path"],
        "instruction": item["instruction"],
        "text": item["text"],
        "language": item.get("language", "zh"),
        "target_values": item["target_values"],
        "quality_checks": item["quality_checks"],
        "selected_candidate_index": item.get("selected_candidate_index"),
        "selected_candidate_id": item.get("selected_candidate_id"),
        "source_json": item["file_name"],
        "source_json_path": relative_source_json_path,
        "wav_path": relative_wav_path,
        "model_name": args.model_name,
        "sample_rate": args.sample_rate,
    }

    if is_audio_complete(out_wav, "wav"):
        result_meta["status"] = "skipped_existing"
        return result_meta

    gemini_input = build_gemini_input(
        instruction=item["instruction"],
        text=item["text"],
        language_code=item.get("language", "zh"),
    )
    chosen_voice_name, chosen_voice_pool = choose_voice_name(
        item.get("target_values", {}),
        uniq_name=item["uniq_name"],
        seed=args.seed,
    )

    async with semaphore:
        pcm_bytes, usage, error = await generate_gemini_tts(
            session=session,
            url=args.api_url.rstrip("/"),
            key=api_key,
            model_name=args.model_name,
            prompt_text=gemini_input,
            voice_name=chosen_voice_name,
            timeout=args.timeout,
        )

    if error:
        error_payload = {
            "provider": "gemini",
            "model_name": args.model_name,
            "uniq_name": item["uniq_name"],
            "taxonomy_node": item["taxonomy_node"],
            "plan_id": item["plan_id"],
            "source_json": item["file_name"],
            "source_json_path": item["source_json_path"],
            "voice_name": chosen_voice_name,
            "voice_pool": chosen_voice_pool,
            "error": error,
            "timestamp": utc_now_iso(),
        }
        await append_error(args.error_log, error_payload)
        result_meta["status"] = "error"
        result_meta["error"] = error
        result_meta["voice_name"] = chosen_voice_name
        result_meta["voice_pool"] = chosen_voice_pool
        return result_meta

    await asyncio.to_thread(wave_file, out_wav, pcm_bytes, 1, args.sample_rate, 2)

    result_meta.update(
        {
            "status": "ok",
            "gemini_input": gemini_input,
            "voice_name": chosen_voice_name,
            "voice_pool": chosen_voice_pool,
            "usage_metadata": usage or {},
            "completed_at": utc_now_iso(),
        }
    )
    await write_meta_json(out_meta, result_meta)
    return result_meta


async def process_all_items(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resolved_error_log = args.error_log or os.path.join(args.output_dir, "errors.jsonl")
    Path(resolved_error_log).parent.mkdir(parents=True, exist_ok=True)
    args.error_log = resolved_error_log

    node_whitelist = load_node_whitelist(args.node_whitelist_json) if args.node_whitelist_json else []

    started_at = utc_now_iso()
    print("Loading instruction-text items...")
    items = load_all_instruction_items(
        input_path=args.input_path,
        node_whitelist=node_whitelist,
        recursive=args.recursive,
        use_plan_id_as_name=args.use_plan_id_as_name,
    )
    print(f"Loaded {len(items)} items.")

    if len(items) == 0:
        print("No valid items found.")
        return

    discovered_count = len(items)
    skipped_existing_count = 0
    if args.skip_existing:
        filtered = []
        for item in items:
            wav_path = os.path.join(args.output_dir, f"{item['uniq_name']}.wav")
            if is_audio_complete(wav_path, "wav"):
                skipped_existing_count += 1
                continue
            filtered.append(item)
        print(f"Skipped {skipped_existing_count} completed items based on the current output directory.")
        items = filtered

    manifest_path = save_manifest(items, args.output_dir, args.input_path)

    if len(items) == 0:
        summary = {
            "provider": "gemini",
            "model_name": args.model_name,
            "input_path": args.input_path,
            "output_dir": args.output_dir,
            "error_log": args.error_log,
            "manifest_path": manifest_path,
            "summary_generated_at": utc_now_iso(),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "discovered_items": discovered_count,
            "attempted_items": 0,
            "skipped_existing": skipped_existing_count,
            "succeeded": 0,
            "failed": 0,
            "recursive": args.recursive,
            "node_whitelist_json": args.node_whitelist_json,
            "use_plan_id_as_name": args.use_plan_id_as_name,
            "evaluation_compatible_naming": args.use_plan_id_as_name,
            "note": "Nothing to run. Existing outputs already cover all discovered items.",
        }
        summary_path = save_summary(summary, args.output_dir, args.input_path)
        print("Nothing to run.")
        print(f"Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")
        return

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Pass --api_key or set GOOGLE_API_KEY.")

    semaphore = asyncio.Semaphore(args.num_workers)
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                process_single_item(
                    item=item,
                    session=session,
                    semaphore=semaphore,
                    args=args,
                    api_key=api_key,
                )
            )
            for item in items
        ]

        ok_count = 0
        err_count = 0
        skip_count = 0

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating with Gemini TTS"):
            try:
                result = await task
                status = result.get("status")
                if status == "ok":
                    ok_count += 1
                elif status == "skipped_existing":
                    skip_count += 1
                else:
                    err_count += 1
            except Exception as e:
                err_count += 1
                await append_error(
                    args.error_log,
                    {
                        "provider": "gemini",
                        "model_name": args.model_name,
                        "uniq_name": "UNKNOWN",
                        "taxonomy_node": "UNKNOWN",
                        "plan_id": "UNKNOWN",
                        "source_json": "UNKNOWN",
                        "error": f"Task failure: {e}",
                        "timestamp": utc_now_iso(),
                    },
                )

    summary = {
        "provider": "gemini",
        "model_name": args.model_name,
        "input_path": args.input_path,
        "output_dir": args.output_dir,
        "error_log": args.error_log,
        "manifest_path": manifest_path,
        "summary_generated_at": utc_now_iso(),
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "discovered_items": discovered_count,
        "attempted_items": len(items),
        "skipped_existing": skipped_existing_count + skip_count,
        "succeeded": ok_count,
        "failed": err_count,
        "recursive": args.recursive,
        "node_whitelist_json": args.node_whitelist_json,
        "use_plan_id_as_name": args.use_plan_id_as_name,
        "evaluation_compatible_naming": args.use_plan_id_as_name,
        "sample_rate": args.sample_rate,
        "note": "Review generated outputs and the error log. Safety filtering, provider instability, or transient failures may leave generation incomplete. Rerun with --skip_existing to fill missing items.",
    }
    summary_path = save_summary(summary, args.output_dir, args.input_path)

    print(f"Done. Outputs saved to: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Error log: {args.error_log}")
    print(f"Summary counts: ok={ok_count}, skipped={skipped_existing_count + skip_count}, error={err_count}")
    print("Review generated outputs and the error log. Safety filtering, provider instability, or transient failures may leave generation incomplete.")
    print("Rerun with --skip_existing to fill missing items without regenerating completed audio.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gemini TTS audio from instruction-text pair JSON files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to an instruction-text pair directory or a single JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for generated audio files and sidecar metadata JSON files.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-preview-tts", help="Gemini TTS model name.")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Output WAV sample rate.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip items whose output audio already looks complete in the output directory.")
    parser.add_argument("--node_whitelist_json", type=str, default="", help="Optional JSON list of taxonomy_node values to include.")
    parser.add_argument("--error_log", type=str, default="", help="Optional JSONL error log path. Defaults to <output_dir>/errors.jsonl.")
    parser.add_argument("--recursive", action="store_true", help="Recursively read pair JSON files under the input path.")
    parser.add_argument("--use_plan_id_as_name", action="store_true", help="Use plan_id as the output basename. This is the recommended mode for evaluation compatibility.")
    parser.add_argument("--api_key", type=str, default="", help="Explicit API key. If omitted, GOOGLE_API_KEY is used.")
    parser.add_argument("--api_url", type=str, default="https://generativelanguage.googleapis.com", help="Gemini API base URL.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed for voice selection.")
    parser.add_argument("--timeout", type=int, default=360, help="Per-request timeout in seconds.")
    parser.add_argument("--num_workers", type=int, default=5, help="Maximum number of concurrent requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(process_all_items(args))


if __name__ == "__main__":
    main()
