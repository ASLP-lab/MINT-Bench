import os
import re
import json
import wave
import base64
import argparse
import random
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
}


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
    raise ValueError("Node whitelist file must be a JSON list")


def collect_json_files(input_path: str, recursive: bool = False) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    if recursive:
        json_files: List[str] = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.endswith(".json") and not fname.startswith("manifest"):
                    json_files.append(os.path.join(root, fname))
        return sorted(json_files)

    return [
        os.path.join(input_path, fname)
        for fname in sorted(os.listdir(input_path))
        if fname.endswith(".json") and not fname.startswith("manifest")
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


def expand_candidate_jobs(
    base_item: Dict[str, Any],
    num_candidates: int,
) -> List[Dict[str, Any]]:
    jobs = []
    for idx in range(num_candidates):
        cand_idx = idx + 1
        cand_item = dict(base_item)
        cand_item["candidate_index"] = cand_idx
        cand_item["candidate_id"] = f"cand{cand_idx:02d}"
        cand_item["candidate_group_size"] = num_candidates
        cand_item["source_plan_id"] = base_item["plan_id"]
        cand_item["uniq_name"] = f"{base_item['uniq_name']}__cand{cand_idx:02d}"
        jobs.append(cand_item)
    return jobs


def load_all_instruction_items(
    input_path: str,
    node_whitelist: Optional[List[str]] = None,
    recursive: bool = False,
    use_plan_id_as_name: bool = True,
    timbre_taxonomy_node: str = "aggregate/timbre",
    num_timbre_candidates: int = 10,
    num_default_candidates: int = 1,
) -> List[Dict[str, Any]]:
    node_whitelist = node_whitelist or []
    items: List[Dict[str, Any]] = []

    json_files = collect_json_files(input_path, recursive=recursive)

    for fpath in json_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        taxonomy_node = data.get("taxonomy_node", os.path.splitext(fname)[0].replace("__", "/"))
        language = data.get("language", "zh")

        if node_whitelist and taxonomy_node not in node_whitelist:
            continue

        results = data.get("results", [])
        if not isinstance(results, list):
            continue

        is_timbre_subset = (taxonomy_node == timbre_taxonomy_node)
        per_item_num_candidates = num_timbre_candidates if is_timbre_subset else num_default_candidates

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

            base_item = {
                "file_name": fname,
                "source_json_path": fpath,
                "taxonomy_node": taxonomy_node,
                "node_slug": node_slug,
                "plan_id": plan_id,
                "meta_path": meta_path,
                "instruction": instruction,
                "text": text,
                "language": language,
                "target_values": target_values if isinstance(target_values, dict) else {},
                "quality_checks": quality_checks if isinstance(quality_checks, dict) else {},
                "selected_candidate_index": selected_candidate_index,
                "selected_candidate_id": selected_candidate_id,
                "uniq_name": uniq_name,
            }

            expanded_items = expand_candidate_jobs(
                base_item=base_item,
                num_candidates=per_item_num_candidates,
            )
            items.extend(expanded_items)

    return items


def save_manifest(items: List[Dict[str, Any]], output_dir: str, input_path: str) -> None:
    manifest_name = "manifest.json"
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        manifest_name = f"manifest__{sanitize_filename(stem)}.json"
    manifest_path = os.path.join(output_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


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
                    part = data["candidates"][0]["content"]["parts"][0]
                except Exception:
                    return None, usage, f"Unexpected response structure: {json.dumps(data, ensure_ascii=False)[:1000]}"

                inline_data = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline_data, dict):
                    return None, usage, f"Missing inlineData audio field: {json.dumps(part, ensure_ascii=False)[:1000]}"

                encoded_audio = inline_data.get("data")
                if not encoded_audio:
                    return None, usage, f"inlineData.data is empty: {json.dumps(inline_data, ensure_ascii=False)[:1000]}"

                try:
                    pcm_bytes = base64.b64decode(encoded_audio)
                except Exception as e:
                    return None, usage, f"Failed to decode base64 audio: {e}"

                return pcm_bytes, usage, None

        except asyncio.TimeoutError:
            last_error = "Request timeout"
        except aiohttp.ClientError as e:
            last_error = f"Network error: {e}"
        except Exception as e:
            last_error = f"Unknown error: {e}"

    return None, None, last_error or "Unknown error"


async def process_single_item(
    item: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    out_wav = os.path.join(args.output_dir, f"{item['uniq_name']}.wav")
    out_meta = os.path.join(args.output_dir, f"{item['uniq_name']}.json")

    result_meta: Dict[str, Any] = {
        "taxonomy_node": item["taxonomy_node"],
        "plan_id": item["plan_id"],
        "source_plan_id": item.get("source_plan_id", item["plan_id"]),
        "meta_path": item["meta_path"],
        "instruction": item["instruction"],
        "text": item["text"],
        "language": item.get("language", "zh"),
        "target_values": item["target_values"],
        "quality_checks": item["quality_checks"],
        "selected_candidate_index": item.get("selected_candidate_index"),
        "selected_candidate_id": item.get("selected_candidate_id"),
        "candidate_index": item.get("candidate_index"),
        "candidate_id": item.get("candidate_id"),
        "candidate_group_size": item.get("candidate_group_size"),
        "source_json": item["file_name"],
        "source_json_path": item["source_json_path"],
        "wav_path": out_wav,
        "model_name": args.model_name,
        "api_url": args.api_url,
        "sample_rate": args.sample_rate,
    }

    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 44:
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
            "uniq_name": item["uniq_name"],
            "taxonomy_node": item["taxonomy_node"],
            "plan_id": item["plan_id"],
            "source_plan_id": item.get("source_plan_id", item["plan_id"]),
            "candidate_index": item.get("candidate_index"),
            "candidate_id": item.get("candidate_id"),
            "source_json": item["file_name"],
            "voice_name": chosen_voice_name,
            "voice_pool": chosen_voice_pool,
            "error": error,
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
        }
    )
    await write_meta_json(out_meta, result_meta)
    return result_meta


async def process_all_items(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Please provide --api_key or set the environment variable GOOGLE_API_KEY")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)

    node_whitelist = load_node_whitelist(args.node_whitelist_json) if args.node_whitelist_json else []

    print("Loading instruction-text items...")
    items = load_all_instruction_items(
        input_path=args.input_path,
        node_whitelist=node_whitelist,
        recursive=args.recursive,
        use_plan_id_as_name=args.use_plan_id_as_name,
        timbre_taxonomy_node=args.timbre_taxonomy_node,
        num_timbre_candidates=args.num_timbre_candidates,
        num_default_candidates=args.num_default_candidates,
    )
    print(f"Loaded {len(items)} synthesis jobs.")

    if len(items) == 0:
        print("No valid items found.")
        return

    if args.skip_existing:
        filtered = []
        for item in items:
            wav_path = os.path.join(args.output_dir, f"{item['uniq_name']}.wav")
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 44:
                filtered.append(item)
        print(f"After skipping existing wavs: {len(filtered)} items remaining.")
        items = filtered

    if len(items) == 0:
        print("Nothing to run.")
        return

    save_manifest(items, args.output_dir, args.input_path)

    with open(args.error_log, "w", encoding="utf-8") as ferr:
        ferr.write("")

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
                        "uniq_name": "UNKNOWN",
                        "taxonomy_node": "UNKNOWN",
                        "plan_id": "UNKNOWN",
                        "error": f"Task exception: {e}",
                    },
                )

    print(f"Done. Outputs saved to: {args.output_dir}")
    print(f"Error log: {args.error_log}")
    print(f"Summary: ok={ok_count}, skipped={skip_count}, error={err_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer Gemini TTS audio from instruction-text pair JSON files with timbre diversity support. "
        "Allows multi-candidate generation for timbre-focused synthesis."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to instruction-text pair JSON directory or file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated WAV and JSON files")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-preview-tts", help="Gemini TTS model name")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Output audio sample rate in Hz")
    parser.add_argument("--skip_existing", action="store_true", help="Skip items with existing WAV files")
    parser.add_argument("--node_whitelist_json", type=str, default="", help="Optional JSON list file specifying which taxonomy nodes to process")
    parser.add_argument("--error_log", type=str, default="gemini_tts_infer_error.jsonl", help="Path to error log file")
    parser.add_argument("--recursive", action="store_true", help="Recursively process JSON files in input_path subdirectories")
    parser.add_argument("--use_plan_id_as_name", action="store_true", help="Use only plan_id as output filename (without taxonomy_node prefix)")
    parser.add_argument("--api_key", type=str, default="", help="API key; if empty, uses GOOGLE_API_KEY environment variable")
    parser.add_argument("--api_url", type=str, default="https://generativelanguage.googleapis.com", help="Gemini API URL base")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for voice selection determinism")
    parser.add_argument("--timeout", type=int, default=360, help="Request timeout in seconds")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--timbre_taxonomy_node", type=str, default="aggregate/timbre", help="Taxonomy node name identifying timbre subset")
    parser.add_argument("--num_timbre_candidates", type=int, default=10, help="Number of candidates to generate per timbre item")
    parser.add_argument("--num_default_candidates", type=int, default=1, help="Number of candidates to generate per non-timbre item")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(process_all_items(args))


if __name__ == "__main__":
    main()