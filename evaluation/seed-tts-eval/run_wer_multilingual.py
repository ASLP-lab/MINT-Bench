import sys
import os
import re
import json
import string
from typing import Tuple, List

from tqdm import tqdm
import numpy as np
import soundfile as sf
import scipy.signal
import zhconv

from jiwer import compute_measures
from zhon.hanzi import punctuation as zh_punctuation
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from funasr import AutoModel


# =========================
# Args
# =========================
wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3].lower()   # zh / en / ja / ko / pt / fr / de / it / ru
device = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"

SUPPORTED_LANGS = {"zh", "en", "ja", "ko", "pt", "fr", "de", "it", "ru", "es"}

if lang not in SUPPORTED_LANGS:
    raise ValueError(f"Unsupported lang={lang}, supported={sorted(SUPPORTED_LANGS)}")


# =========================
# Constants
# =========================
PUNCTUATION_ALL = zh_punctuation + string.punctuation + "“”‘’、。·《》【】（）—…￥"

# CJK 统一按字符级
CHAR_LEVEL_LANGS = {"zh", "ja", "ko"}

# 空格语言按词级
WORD_LEVEL_LANGS = {"en", "pt", "fr", "de", "it", "ru", "es"}

WHISPER_LANGUAGE_MAP = {
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ru": "russian",
    "es": "spanish"
}


# =========================
# Model Loaders
# =========================
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")


def load_whisper_model():
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(
        model_id,
        cache_dir=PRETRAINED_DIR,
        # local_files_only=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=PRETRAINED_DIR,
        # local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def load_zh_model():
    model = AutoModel(
        model="paraformer-zh",
        cache_dir=PRETRAINED_DIR,
        local_files_only=True,
        disable_update=True,
    )
    return model


# =========================
# Text Normalization
# =========================
def remove_punctuation(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    for ch in PUNCTUATION_ALL:
        if ch == "'":
            # 英文缩写先保留 apostrophe
            continue
        text = text.replace(ch, "")

    # 统一空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_for_eval(text: str, lang: str) -> str:
    text = remove_punctuation(text)

    if lang == "zh":
        text = zhconv.convert(text, "zh-cn")

    if lang in WORD_LEVEL_LANGS:
        text = text.lower().strip()
    else:
        text = text.strip()

    return text


def to_eval_units(text: str, lang: str) -> str:
    """
    返回给 jiwer 的字符串：
    - 词级语言：空格分词后的字符串
    - 字符级语言：字符间加空格
    """
    text = normalize_text_for_eval(text, lang)

    if lang in CHAR_LEVEL_LANGS:
        return " ".join(list(text))
    elif lang in WORD_LEVEL_LANGS:
        return text
    else:
        raise NotImplementedError(f"Unsupported eval unit lang={lang}")


# =========================
# ASR
# =========================
def load_audio_16k(wav_path: str) -> np.ndarray:
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))

    wav = wav.astype(np.float32)
    return wav


def transcribe_with_whisper(
    wav_path: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    lang: str,
) -> str:
    wav = load_audio_16k(wav_path)
    input_features = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)
    
    model_dtype = next(model.parameters()).dtype
    input_features = input_features.to(device=device, dtype=model_dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=WHISPER_LANGUAGE_MAP[lang],
        task="transcribe",
    )

    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
    )
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]
    return transcription


def transcribe_with_paraformer(wav_path: str, model) -> str:
    res = model.generate(
        input=wav_path,
        batch_size_s=300,
    )
    transcription = res[0]["text"]
    transcription = zhconv.convert(transcription, "zh-cn")
    return transcription


# =========================
# Eval
# =========================
def process_one(hypo: str, truth: str, lang: str) -> Tuple[str, str, float, float, float, float]:
    raw_truth = truth
    raw_hypo = hypo

    truth_eval = to_eval_units(truth, lang)
    hypo_eval = to_eval_units(hypo, lang)

    measures = compute_measures(truth_eval, hypo_eval)

    ref_list = truth_eval.split()
    ref_len = max(1, len(ref_list))

    wer = measures["wer"]
    subs = measures["substitutions"] / ref_len
    dele = measures["deletions"] / ref_len
    inse = measures["insertions"] / ref_len

    return raw_truth, raw_hypo, wer, subs, dele, inse


def parse_manifest_line(line: str) -> Tuple[str, str]:
    """
    支持以下格式：
    1) wav_res_path|text_ref
    2) wav_res_path|wav_ref_path|text_ref
    3) wav_res_path|_|text_ref|wav_ref_path
    """
    parts = line.strip().split("|")

    if len(parts) == 2:
        wav_res_path, text_ref = parts
    elif len(parts) == 3:
        wav_res_path, _, text_ref = parts
    elif len(parts) == 4:
        wav_res_path, _, text_ref, _ = parts
    else:
        raise NotImplementedError(f"Unexpected line format: {line}")

    return wav_res_path, text_ref


def run_asr(wav_res_text_path: str, res_path: str, lang: str):
    if lang == "zh":
        zh_model = load_zh_model()
        whisper_processor, whisper_model = None, None
    else:
        whisper_processor, whisper_model = load_whisper_model()
        zh_model = None

    params: List[Tuple[str, str]] = []
    with open(wav_res_text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            wav_res_path, text_ref = parse_manifest_line(line)
            if not os.path.exists(wav_res_path):
                continue
            params.append((wav_res_path, text_ref))

    fout = open(res_path, "w", encoding="utf-8")

    n_total = 0
    n_high_wer = 0
    wers = []

    for wav_res_path, text_ref in tqdm(params):
        try:
            if lang == "zh":
                transcription = transcribe_with_paraformer(wav_res_path, zh_model)
            else:
                transcription = transcribe_with_whisper(
                    wav_res_path,
                    whisper_processor,
                    whisper_model,
                    lang,
                )

            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(
                transcription,
                text_ref,
                lang,
            )

            fout.write(
                f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n"
            )
            fout.flush()

            n_total += 1
            wers.append(wer)
            if wer > 0.5:
                n_high_wer += 1

        except Exception as e:
            fout.write(f"{wav_res_path}\tERROR\t{text_ref}\t{repr(e)}\n")
            fout.flush()

    fout.close()

    if n_total > 0:
        print("=" * 60)
        print(f"lang={lang}")
        print(f"num_samples={n_total}")
        print(f"avg_error={float(np.mean(wers)):.4f}")
        print(f"median_error={float(np.median(wers)):.4f}")
        print(f"ratio_error_gt_0.5={n_high_wer / n_total:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    run_asr(wav_res_text_path, res_path, lang)