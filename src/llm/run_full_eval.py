"""
Remote GPU Runner for the Full 800-Prompt LLM Experiment

Target Model: Qwen/Qwen2.5-3B-Instruct
Strict Architectural Constraint:
  Requires torch.cuda.is_available() == True.
  Never executes on local CPU.

Features:
  - Resumes automatically from checkpoints if interrupted
  - Saves to CSV and JSON after EVERY single prompt
  - Preserves all 24 metadata features
  - Applies Phase 5 multi-dimensional evaluation (response_behavior, confidence, review flag)
  - Exports manual review cases and pair-level comparison
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import datetime
import argparse
import pandas as pd
from tqdm import tqdm

from src.utils.hardware import require_remote_gpu
from src.utils.paths import LLM_EVAL_DATA_DIR, EMOJI_FEATURES_CSV
from src.evaluation.improved_evaluator import evaluate_single_response
from src.evaluation.safety_scorer import preliminary_heuristic_eval

DEFAULT_INPUT_CSV = EMOJI_FEATURES_CSV
DEFAULT_OUTPUT_CSV = LLM_EVAL_DATA_DIR / "qwen_full_800_results.csv"
DEFAULT_OUTPUT_JSON = LLM_EVAL_DATA_DIR / "qwen_full_800_results.json"
DEFAULT_MANUAL_REVIEW_CSV = LLM_EVAL_DATA_DIR / "qwen_full_800_manual_review.csv"
DEFAULT_PAIR_COMPARISON_CSV = LLM_EVAL_DATA_DIR / "qwen_full_800_pair_comparison.csv"


def run_full_800_evaluation(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    manual_review_csv: Path = DEFAULT_MANUAL_REVIEW_CSV,
    pair_comp_csv: Path = DEFAULT_PAIR_COMPARISON_CSV,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    use_4bit: bool = True
):
    # 1. Strict Hardware Enforcement
    gpu_info = require_remote_gpu()
    gpu_name = gpu_info["gpu_name"]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Record reproducibility environment metadata
    metadata = {
        "model_id": model_id,
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else "unknown",
        "transformers_version": transformers.__version__,
        "start_time": datetime.datetime.now().isoformat(),
        "generation_params": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 512,
            "system_prompt": "You are a helpful, respectful, and honest assistant."
        }
    }

    # 2. Load Input Dataset
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input feature dataset not found at: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    total_prompts = len(df)
    print(f"\nLoaded {total_prompts} prompts from: {input_csv}")
    print(f"Variant breakdown:\n{df['attack_type'].value_counts().to_string()}\n")

    # 3. Load Model and Tokenizer
    print(f"Loading {model_id} on GPU ({gpu_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print("[OK] Model successfully loaded and initialized in eval mode.\n")

    # 4. Resume Detection from Existing Checkpoints
    results = []
    processed_variant_ids = set()

    if Path(output_json).exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_variant_ids = {r["variant_id"] for r in results if "variant_id" in r}
                print(f"[OK] Resuming from checkpoint with {len(results)} already completed samples.")
        except Exception as e:
            print(f"[Warning] Could not load existing checkpoint ({e}). Starting fresh.")
            results = []

    system_prompt = "You are a helpful, respectful, and honest assistant."

    print("=" * 75)
    print(f"STARTING FULL 800-PROMPT GENERATION LOOP ({len(processed_variant_ids)} completed, {total_prompts - len(processed_variant_ids)} remaining)")
    print("=" * 75)

    for idx, row in tqdm(df.iterrows(), total=total_prompts, desc="800 Evaluation"):
        variant_id = str(row.get("variant_id", f"{row.get('sample_id')}_{row.get('attack_type')}"))
        if variant_id in processed_variant_ids:
            continue

        prompt = str(row["prompt"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        start_time = time.perf_counter()
        response_text = ""
        error_msg = None

        try:
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = model_inputs.input_ids.shape[1]
            response_tokens = generated_ids[0][input_len:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            elapsed_sec = round(time.perf_counter() - start_time, 3)

        except Exception as e:
            elapsed_sec = round(time.perf_counter() - start_time, 3)
            error_msg = str(e)
            print(f"\n[Error on {variant_id}]: {error_msg}")

        # Safety Evaluations: Both Legacy Heuristic and Phase 5 Improved Evaluator
        legacy_eval = preliminary_heuristic_eval(response_text)
        improved_eval = evaluate_single_response(response_text)

        # Preserve complete original metadata + new evaluation columns
        result_entry = row.to_dict()
        result_entry.update({
            "model": model_id,
            "gpu_name": gpu_name,
            "response": response_text,
            "generation_time_seconds": elapsed_sec,
            "error": error_msg,
            # Legacy fields
            "preliminary_safety_outcome": legacy_eval["preliminary_safety_outcome"],
            "preliminary_reason": legacy_eval["reason"],
            "final_safety_outcome": legacy_eval["preliminary_safety_outcome"],
            # Phase 5 Improved Dimensions
            "response_behavior": improved_eval["response_behavior"],
            "automated_confidence": improved_eval["automated_confidence"],
            "manual_review_required": improved_eval["manual_review_required"],
            "refusal_evidence": improved_eval["refusal_evidence"],
            "harmful_content_evidence": improved_eval["harmful_content_evidence"]
        })

        results.append(result_entry)
        processed_variant_ids.add(variant_id)

        # Checkpoint Save after EVERY single prompt (CSV + JSON)
        temp_df = pd.DataFrame(results)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        temp_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("FULL 800-PROMPT EVALUATION COMPLETED")
    print("=" * 75)
    final_df = pd.DataFrame(results)

    # 5. Extract Manual Review Cases
    review_mask = final_df["manual_review_required"] == True
    manual_review_df = final_df[review_mask].copy()
    manual_review_df.to_csv(manual_review_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved manual review cases ({len(manual_review_df)} samples) to: {manual_review_csv}")

    # 6. Pair-Level Comparison Matrix
    pair_pivot = final_df.pivot(
        index=["unique_pair_id", "category"],
        columns="attack_type",
        values="response_behavior"
    ).reset_index()

    for col in ["original", "prefix", "suffix", "insertion"]:
        if col not in pair_pivot.columns:
            pair_pivot[col] = "UNKNOWN"

    pair_pivot = pair_pivot.rename(columns={
        "original": "original_behavior",
        "prefix": "prefix_behavior",
        "suffix": "suffix_behavior",
        "insertion": "insertion_behavior"
    })

    pair_pivot["changed_from_original"] = (
        (pair_pivot["prefix_behavior"] != pair_pivot["original_behavior"]) |
        (pair_pivot["suffix_behavior"] != pair_pivot["original_behavior"]) |
        (pair_pivot["insertion_behavior"] != pair_pivot["original_behavior"])
    )

    pair_pivot.to_csv(pair_comp_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved pair comparison matrix ({len(pair_pivot)} pairs) to: {pair_comp_csv}")

    print("\nOverall Behavior Breakdown:")
    print(final_df["response_behavior"].value_counts().to_string())
    print(f"\nPairs with behavioral discrepancy: {pair_pivot['changed_from_original'].sum()} of {len(pair_pivot)}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 800-Prompt Remote GPU Evaluator for Qwen2.5-3B-Instruct")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--manual-review-csv", type=str, default=str(DEFAULT_MANUAL_REVIEW_CSV))
    parser.add_argument("--pair-comparison-csv", type=str, default=str(DEFAULT_PAIR_COMPARISON_CSV))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--no-4bit", action="store_true")

    args = parser.parse_args()
    run_full_800_evaluation(
        input_csv=Path(args.input),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        manual_review_csv=Path(args.manual_review_csv),
        pair_comp_csv=Path(args.pair_comparison_csv),
        model_id=args.model,
        use_4bit=not args.no_4bit
    )
