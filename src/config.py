# config.py -- single source of truth for the dataset build.
#
# CODE lives in the cloned repo (ephemeral). DATA lives in persistent storage
# (Google Drive on Colab, /kaggle/working on Kaggle) so a disconnect never
# loses generated samples. _base_dir() auto-detects the platform.
from pathlib import Path

import torch


def _base_dir() -> Path:
    """Return the persistent DATA root for the current platform."""
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/cot-compliance-safety")   # Colab + Drive
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cot-compliance-safety")            # Kaggle
    # Local dev: repo root (parent of src/)
    return Path(__file__).resolve().parent.parent


BASE     = _base_dir()
DATA_DIR = BASE / "data"
JOB_LIST = DATA_DIR / "job_list.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def batch_dirs(batch: str):
    """Return dictionary of directories for a given batch (e.g., 'with_attacks'), creating them if needed."""
    bdir = DATA_DIR / batch
    dirs = {
        "generations": bdir / "generations",
        "cot": bdir / "cot",
        "final": bdir / "final",
        "labeled": bdir / "labeled"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

SEED = 42

# --- taxonomy ---
HARM_CATEGORIES = ["physical_harm", "cybercrime", "misinformation", "privacy", "hate_speech"]
ATTACKS         = ["none", "h_cot", "rto", "race"]   # "none" = benign / baseline

# --- sampling budget (start small, scale later) ---
PROMPTS_PER_CELL    = 120   # per (category x adversarial-attack)
BENIGN_PER_CATEGORY = 60    # attack="none", for false-positive-rate denominator

# --- models: run ONE per session, swap via --model CLI flag ---
# Each entry is a dict with:
#   hf_id            – HuggingFace model repo
#   think_open/close – delimiters used to separate CoT from final answer
#   enable_thinking  – passed to apply_chat_template for models that need it
#                      (None = omit the kwarg entirely; True/False = pass it)
#   shard            – if True, device_map="auto" is already set (32B across T4s)
MODELS = {
    "deepseek-r1-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": None,
    },
    "deepseek-r1-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": None,
    },
    "deepseek-r1-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": None,          # not needed for R1 distills
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": True,          # REQUIRED or Qwen3 emits no CoT trace
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": True,          # REQUIRED or Qwen3 emits no CoT trace
    },
    "qwen3-14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": True,          # REQUIRED or Qwen3 emits no CoT trace
    },
    "qwq-32b": {
        "hf_id": "Qwen/QwQ-32B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": None,
        "shard": True,                    # 32B -> split across both T4s via device_map=auto
    },
    "deepseek-r1-32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "think_open": "<think>", "think_close": "</think>",
        "enable_thinking": None,
        "shard": True,                    # 32B -> split across both T4s via device_map=auto
    },
}

# --- generation hyperparameters (DeepSeek-R1 recommended) ---
MAX_NEW_TOKENS = 4096   # reasoning traces are long
TEMPERATURE    = 0.6
TOP_P          = 0.95

# --- 4-bit loading (shared by inference.py and labeling scripts) ---
BNB_4BIT = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# --- labeling models (used in labeling.py, separate resumable pass) ---
GUARD_MODEL = "meta-llama/Llama-Guard-3-8B"   # output-label classifier (fits 4-bit)
# CoT judge (Llama-3.3-70B) is best run via a hosted API; see labeling.py.