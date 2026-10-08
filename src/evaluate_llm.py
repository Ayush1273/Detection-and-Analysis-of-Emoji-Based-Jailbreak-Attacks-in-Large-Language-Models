"""
Target LLM Evaluation Script (Remote GPU Execution Only)

This module handles target LLM evaluation (e.g. Qwen/Qwen2.5-3B-Instruct).
In accordance with the project architecture:
- LOCAL INFERENCE IS STRICTLY DISABLED to prevent machine degradation.
- INFERENCE RUNS EXCLUSIVELY ON REMOTE GPU (Google Colab / Cloud GPU).

Usage on Remote GPU (Colab):
    python src/evaluate_llm.py --input data/processed/llm_evaluation/pilot_40_prompts.csv --model Qwen/Qwen2.5-3B-Instruct
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
from tqdm import tqdm

from src.utils.hardware import require_remote_gpu
from src.llm.runner import load_target_model, generate_response, RESULT_COLUMNS
from src.utils.paths import LLM_EVAL_DATA_DIR


def evaluate_batch(
    input_csv: str,
    output_csv: str,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    use_4bit: bool = True
):
    # Enforce remote GPU check before attempting any model import or loading
    require_remote_gpu()

    in_path = Path(input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    print(f"Loaded {len(df)} prompts from {in_path}")

    model, tokenizer = load_target_model(model_id, use_4bit=use_4bit)

    results = []
    print(f"Starting evaluation of {len(df)} samples against {model_id}...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        gen_result = generate_response(prompt, model, tokenizer)

        res_row = {
            "unique_pair_id": row.get("unique_pair_id", f"{row.get('safety_label', '')}_{row.get('pair_id', '')}"),
            "sample_id": row.get("sample_id", idx),
            "safety_label": row.get("safety_label", ""),
            "category": row.get("category", ""),
            "behavior": row.get("behavior", ""),
            "attack_type": row.get("attack_type", ""),
            "emoji": row.get("emoji", ""),
            "emoji_count": row.get("emoji_count", 0),
            "emoji_density": row.get("emoji_density", 0.0),
            "emoji_position": row.get("emoji_position", "none"),
            "prompt": prompt,
            "model": model_id,
            "response": gen_result["response"],
            "preliminary_safety_outcome": "PENDING",
            "final_safety_outcome": "PENDING",
            "generation_time_seconds": gen_result["generation_time_seconds"]
        }
        results.append(res_row)

    results_df = pd.DataFrame(results)[RESULT_COLUMNS]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Evaluation complete! Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remote LLM Evaluation Runner for Emoji-Jailbreak Research"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(LLM_EVAL_DATA_DIR / "pilot_40_prompts.csv"),
        help="Path to evaluation input CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(LLM_EVAL_DATA_DIR / "qwen_pilot_40_results.csv"),
        help="Path to save evaluation output CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires >8GB VRAM in bfloat16)"
    )

    args = parser.parse_args()

    # Hardware check ensures local runs abort gracefully
    evaluate_batch(
        input_csv=args.input,
        output_csv=args.output,
        model_id=args.model,
        use_4bit=not args.no_4bit
    )


if __name__ == "__main__":
    main()
