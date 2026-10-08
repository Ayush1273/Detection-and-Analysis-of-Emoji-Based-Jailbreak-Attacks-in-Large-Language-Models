from pathlib import Path

# Project root directory (3 levels up from this file: src/utils/paths.py -> emoji-jailbreak-research/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LLM_EVAL_DATA_DIR = PROCESSED_DATA_DIR / "llm_evaluation"

# Key data files
BASE_PROMPTS_CSV = PROCESSED_DATA_DIR / "base_prompts.csv"
EMOJI_PROMPTS_CSV = PROCESSED_DATA_DIR / "emoji_prompts.csv"
EMOJI_FEATURES_CSV = PROCESSED_DATA_DIR / "emoji_features.csv"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure runtime directories exist
LLM_EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
