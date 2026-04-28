"""Compute timbre diversity (pairwise speaker similarity) for a timbre summary JSON.

Inputs come from prepare_timbre_diversity_payload.py and live under
evaluation/payloads_timbre_diversity/<run>.summary.json. Each per_plan entry
lists a small group of valid candidate WAVs sharing one (plan_id, speaker)
identity. Mean cosine similarity within each group is the per-group score; we
also emit a global mean and a normalized "diversity" metric (1 - similarity)
weighted by valid item count.
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from tqdm import tqdm

from speaker_similarity.ecapa_tdnn import ECAPA_TDNN_SMALL


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "pretrained" / "wavlm_large_finetune.pth"

MODEL_LIST = [
    "ecapa_tdnn",
    "hubert_large",
    "wav2vec2_xlsr",
    "unispeech_sat",
    "wavlm_base_plus",
    "wavlm_large",
]


def init_model(model_name: str, checkpoint: str | None = None):
    if model_name == "unispeech_sat":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="unispeech_sat", config_path="config/unispeech_sat.th")
    elif model_name == "wavlm_base_plus":
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type="wavlm_base_plus", config_path=None)
    elif model_name == "wavlm_large":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    elif model_name == "hubert_large":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="hubert_large_ll60k", config_path=None)
    elif model_name == "wav2vec2_xlsr":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wav2vec2_xlsr", config_path=None)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type="fbank")

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)
    return model


def get_emb(model, wav_path: str, device: str = "cpu", sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

    if sr != sample_rate:
        resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resample(wav)

    with torch.no_grad():
        emb = model(wav)

    return emb


def verification_group_pairwise(
    model,
    group_wavs: Dict[str, List[str]],
    device: str = "cpu",
    sample_rate: int = 16000,
) -> Tuple[float, Dict[str, float], Dict[str, int], Dict[str, float]]:
    wav_embs: Dict[str, torch.Tensor] = {}
    all_wavs = [wav for wavs in group_wavs.values() for wav in wavs]
    for wav in tqdm(all_wavs, desc="Extract embeddings"):
        wav_embs[wav] = get_emb(model, wav, device, sample_rate)

    spk_group_means: Dict[str, float] = {}
    spk_group_pair_nums: Dict[str, int] = {}
    spk_group_metrics: Dict[str, float] = {}

    for spk_id, wavs in tqdm(group_wavs.items(), desc="Compute pairwise similarity"):
        pair_sims: List[float] = []
        for wav1, wav2 in combinations(wavs, 2):
            sim = F.cosine_similarity(wav_embs[wav1], wav_embs[wav2], dim=1).mean().item()
            pair_sims.append(sim)

        spk_group_pair_nums[spk_id] = len(pair_sims)
        if pair_sims:
            spk_avg = float(np.mean(pair_sims))
            spk_group_means[spk_id] = spk_avg
            spk_group_metrics[spk_id] = (1.0 - spk_avg) * (len(wavs) / 10.0)

    if spk_group_metrics:
        global_metric = float(np.mean(list(spk_group_metrics.values())))
    else:
        global_metric = float("nan")

    return global_metric, spk_group_means, spk_group_pair_nums, spk_group_metrics


def gen_group(wavs: List[str]) -> Dict[str, List[str]]:
    group_wavs: Dict[str, List[str]] = defaultdict(list)
    for wav in wavs:
        utt = os.path.basename(wav).split(".")[0]
        spk_id = "__".join(utt.split("__")[:-1])
        group_wavs[spk_id].append(wav)
    return group_wavs


def resolve_wav_path(wav_path: str, input_json: Path) -> str:
    """Resolve a wav path that may be absolute, or relative to either the
    summary JSON's directory or the parent evaluation/ dir."""
    p = Path(wav_path)
    if p.is_absolute():
        return str(p)

    input_json_path = input_json.resolve()
    evaluation_dir = input_json_path.parent.parent

    candidates = [
        (evaluation_dir / p).resolve(),
        (input_json_path.parent / p).resolve(),
        p.resolve(),
    ]

    for cand in candidates:
        if cand.exists():
            return str(cand)

    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute timbre diversity score from a summary JSON produced by "
        "prepare_timbre_diversity_payload.py. Uses WavLM-large speaker embeddings "
        "and writes per-group similarity, normalized diversity, and a global mean."
    )
    parser.add_argument(
        "--input_json", required=True, type=Path,
        help="Path to summary JSON (typically under evaluation/payloads_timbre_diversity/)",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory to write score files into. Created if missing.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Torch device to run the embedding model on (default: cuda:0)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"WavLM-large fine-tuned checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Audio sample rate fed to the embedding model (default: 16000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    wavs: List[str] = []
    for item in data.get("per_plan", []):
        wavs.extend(resolve_wav_path(wp, args.input_json) for wp in item.get("wav_paths", []))

    group_wavs = gen_group(wavs)

    model = init_model("wavlm_large", str(args.checkpoint))
    model.eval()
    model.to(args.device)

    global_metric, spk_group_means, spk_group_pair_nums, spk_group_metrics = verification_group_pairwise(
        model, group_wavs, args.device, args.sample_rate,
    )

    basename = args.input_json.stem
    if basename.endswith(".summary"):
        basename = basename[: -len(".summary")]

    txt_path = args.output_dir / f"{basename}_timbre_diversity_score.txt"
    json_path = args.output_dir / f"{basename}_timbre_diversity_score.json"

    per_group: List[Dict[str, Any]] = []
    for spk_id in sorted(group_wavs.keys()):
        per_group.append({
            "spk_id": spk_id,
            "items": len(group_wavs[spk_id]),
            "pairs": spk_group_pair_nums.get(spk_id, 0),
            "mean_similarity": spk_group_means.get(spk_id),
            "metric": spk_group_metrics.get(spk_id),
        })

    summary = {
        "input_json": str(args.input_json),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "sample_rate": args.sample_rate,
        "num_groups": len(group_wavs),
        "global_metric": global_metric,
        "per_group": per_group,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"global_metric: {global_metric}\n")
        for row in per_group:
            mean = row["mean_similarity"]
            metric = row["metric"]
            mean_s = "nan" if mean is None else f"{mean}"
            metric_s = "nan" if metric is None else f"{metric}"
            f.write(
                f"{row['spk_id']}: mean={mean_s}, items={row['items']}, "
                f"pairs={row['pairs']}, metric={metric_s}\n"
            )

    print(
        f"input_json={args.input_json.name} groups={len(group_wavs)} "
        f"global_metric={global_metric} out_txt={txt_path}"
    )


if __name__ == "__main__":
    main()
