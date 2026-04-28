import os

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from funasr import AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained")

device = "cuda:0"
model_id = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_id, cache_dir=PRETRAINED_DIR)
model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=PRETRAINED_DIR).to(device)

model = AutoModel(model="paraformer-zh", cache_dir=PRETRAINED_DIR, local_files_only=True)
