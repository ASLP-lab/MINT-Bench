import argparse
import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


LANGUAGE_LABELS: Dict[str, str] = {
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


def load_node_whitelist(path: str) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("The node whitelist file must contain a JSON list.")


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
            for fname in sorted(files):
                if should_keep_file(fname):
                    json_files.append(os.path.join(root, fname))
        return json_files

    return [
        os.path.join(input_path, fname)
        for fname in sorted(os.listdir(input_path))
        if should_keep_file(fname)
    ]


def is_pair_json(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and isinstance(data.get("results"), list)


def normalize_pair_language(language: str) -> str:
    lang = str(language or "").strip().lower()
    return LANGUAGE_LABELS.get(lang, lang or "unknown")


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
        if node_whitelist and taxonomy_node not in node_whitelist:
            continue

        pair_language = str(data.get("language", "")).strip().lower()
        results = data.get("results", [])

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
            if use_plan_id_as_name:
                uniq_name = sanitize_filename(plan_id)
            else:
                uniq_name = f"{node_slug}__{sanitize_filename(plan_id)}"

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
                    "language_code": pair_language,
                    "language": normalize_pair_language(pair_language),
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


def append_error(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as ferr:
        ferr.write(json.dumps(payload, ensure_ascii=False) + "\n")


def is_audio_complete(path: str, audio_format: str) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if audio_format.lower() == "wav":
        return size > WAV_HEADER_BYTES
    return size > 0


def build_client(api_key: str, base_url: str):
    if not api_key:
        raise ValueError("Missing API key. Set MIMO_API_KEY or pass --api_key.")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def synthesize_audio(
    client,
    model: str,
    instruction: str,
    text: str,
    audio_format: str,
) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": text},
        ],
        audio={"format": audio_format},
    )

    message = completion.choices[0].message
    if not getattr(message, "audio", None) or not getattr(message.audio, "data", None):
        raise ValueError("The API response did not include audio.data.")

    return {
        "completion": completion,
        "message": message,
        "audio_bytes": base64.b64decode(message.audio.data),
    }


def completion_to_dict(completion: Any) -> Dict[str, Any]:
    if hasattr(completion, "model_dump"):
        return completion.model_dump()
    if hasattr(completion, "dict"):
        return completion.dict()
    return json.loads(json.dumps(completion, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MiMo TTS audio from instruction-text pair JSON files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to an instruction-text pair directory or a single JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for generated audio files and sidecar metadata JSON files.")
    parser.add_argument("--model", type=str, default="mimo-v2.5-tts-voicedesign", help="MiMo API model name.")
    parser.add_argument("--base_url", type=str, default="https://api.xiaomimimo.com/v1", help="MiMo API base URL.")
    parser.add_argument("--api_key", type=str, default="", help="Explicit API key. If omitted, MIMO_API_KEY is used.")
    parser.add_argument("--audio_format", type=str, default="wav", help="Output audio format, for example wav.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip items whose output audio already looks complete in the output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively read pair JSON files under the input path.")
    parser.add_argument("--node_whitelist_json", type=str, default="", help="Optional JSON list of taxonomy_node values to include.")
    parser.add_argument("--use_plan_id_as_name", action="store_true", help="Use plan_id as the output basename. This is the recommended mode for evaluation compatibility.")
    parser.add_argument("--error_log", type=str, default="", help="Optional JSONL error log path. Defaults to <output_dir>/errors.jsonl.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resolved_error_log = args.error_log or os.path.join(args.output_dir, "errors.jsonl")
    Path(os.path.dirname(resolved_error_log) or ".").mkdir(parents=True, exist_ok=True)
    args.error_log = resolved_error_log

    node_whitelist = load_node_whitelist(args.node_whitelist_json) if args.node_whitelist_json else []

    started_at = utc_now_iso()
    print("Loading instruction-text items...")
    items = load_all_instruction_items(
        args.input_path,
        node_whitelist=node_whitelist,
        recursive=args.recursive,
        use_plan_id_as_name=args.use_plan_id_as_name,
    )
    print(f"Loaded {len(items)} items.")

    if not items:
        print("No valid items found.")
        return

    discovered_count = len(items)
    skipped_existing_count = 0
    if args.skip_existing:
        filtered = []
        for item in items:
            audio_path = os.path.join(args.output_dir, f"{item['uniq_name']}.{args.audio_format}")
            if is_audio_complete(audio_path, args.audio_format):
                skipped_existing_count += 1
                continue
            filtered.append(item)
        print(f"Skipped {skipped_existing_count} completed items based on the current output directory.")
        items = filtered

    manifest_path = save_manifest(items, args.output_dir, args.input_path)

    if not items:
        summary = {
            "provider": "mimo",
            "model": args.model,
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
            "audio_format": args.audio_format,
            "note": "Nothing to run. Existing outputs already cover all discovered items.",
        }
        summary_path = save_summary(summary, args.output_dir, args.input_path)
        print("Nothing to run.")
        print(f"Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")
        return

    api_key = args.api_key or os.environ.get("MIMO_API_KEY", "")
    client = build_client(api_key, args.base_url)

    ok_count = 0
    err_count = 0

    for item in tqdm(items, desc="Generating with MiMo TTS"):
        output_audio_path = os.path.join(args.output_dir, f"{item['uniq_name']}.{args.audio_format}")
        output_meta_path = os.path.join(args.output_dir, f"{item['uniq_name']}.json")
        relative_wav_path = to_evaluation_relative_path(output_audio_path)
        relative_source_json_path = to_evaluation_relative_path(item["source_json_path"])

        try:
            result = synthesize_audio(
                client=client,
                model=args.model,
                instruction=item["instruction"],
                text=item["text"],
                audio_format=args.audio_format,
            )

            with open(output_audio_path, "wb") as f:
                f.write(result["audio_bytes"])

            message = result["message"]
            audio_info = getattr(message, "audio", None)

            with open(output_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "ok",
                        "provider": "mimo",
                        "taxonomy_node": item["taxonomy_node"],
                        "plan_id": item["plan_id"],
                        "meta_path": item["meta_path"],
                        "instruction": item["instruction"],
                        "text": item["text"],
                        "language": item["language"],
                        "language_code": item["language_code"],
                        "target_values": item["target_values"],
                        "quality_checks": item["quality_checks"],
                        "selected_candidate_index": item["selected_candidate_index"],
                        "selected_candidate_id": item["selected_candidate_id"],
                        "source_json": item["file_name"],
                        "source_json_path": relative_source_json_path,
                        "wav_path": relative_wav_path,
                        "audio_format": args.audio_format,
                        "model": args.model,
                        "response_audio_id": getattr(audio_info, "id", None) if audio_info else None,
                        "completion_id": getattr(result["completion"], "id", None),
                        "completed_at": utc_now_iso(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            ok_count += 1

        except Exception as e:
            err_count += 1
            append_error(
                args.error_log,
                {
                    "provider": "mimo",
                    "model": args.model,
                    "uniq_name": item["uniq_name"],
                    "taxonomy_node": item["taxonomy_node"],
                    "plan_id": item["plan_id"],
                    "source_json": item["file_name"],
                    "source_json_path": item["source_json_path"],
                    "error": str(e),
                    "timestamp": utc_now_iso(),
                },
            )

    summary = {
        "provider": "mimo",
        "model": args.model,
        "input_path": args.input_path,
        "output_dir": args.output_dir,
        "error_log": args.error_log,
        "manifest_path": manifest_path,
        "summary_generated_at": utc_now_iso(),
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "discovered_items": discovered_count,
        "attempted_items": len(items),
        "skipped_existing": skipped_existing_count,
        "succeeded": ok_count,
        "failed": err_count,
        "recursive": args.recursive,
        "node_whitelist_json": args.node_whitelist_json,
        "use_plan_id_as_name": args.use_plan_id_as_name,
        "evaluation_compatible_naming": args.use_plan_id_as_name,
        "audio_format": args.audio_format,
        "base_url": args.base_url,
        "note": "Review generated outputs and the error log. Safety filtering, provider instability, or transient failures may leave generation incomplete. Rerun with --skip_existing to fill missing items.",
    }
    summary_path = save_summary(summary, args.output_dir, args.input_path)

    print(f"Done. Outputs saved to: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Error log: {args.error_log}")
    print(f"Summary counts: ok={ok_count}, skipped={skipped_existing_count}, error={err_count}")
    print("Review generated outputs and the error log. Safety filtering, provider instability, or transient failures may leave generation incomplete.")
    print("Rerun with --skip_existing to fill missing items without regenerating completed audio.")


if __name__ == "__main__":
    main()
