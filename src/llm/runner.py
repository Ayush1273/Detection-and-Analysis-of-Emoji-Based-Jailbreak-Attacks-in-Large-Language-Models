import time
import sys
from pathlib import Path
import pandas as pd

from src.utils.hardware import require_remote_gpu

# Target model identifier
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Required output schema columns
RESULT_COLUMNS = [
    "unique_pair_id",
    "sample_id",
    "safety_label",
    "category",
    "behavior",
    "attack_type",
    "emoji",
    "emoji_count",
    "emoji_density",
    "emoji_position",
    "prompt",
    "model",
    "response",
    "preliminary_safety_outcome",
    "final_safety_outcome",
    "generation_time_seconds"
]


def load_target_model(model_id: str = DEFAULT_MODEL_ID, use_4bit: bool = True):
    """
    Loads target model on a remote GPU.
    Guarded: Will fail immediately if executed without a CUDA GPU.
    """
    require_remote_gpu()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_id} on remote GPU...")
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

    print(f"Successfully loaded {model_id} onto GPU.")
    return model, tokenizer


def generate_response(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful, respectful, and honest assistant."
) -> dict:
    """
    Reusable generation function for target LLMs.
    Uses model's official chat template, deterministic generation, and records latency.
    """
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    start_time = time.perf_counter()
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **gen_kwargs
            )

        # Slice generated response tokens beyond input prompt
        input_len = model_inputs.input_ids.shape[1]
        response_tokens = generated_ids[0][input_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        elapsed_sec = round(time.perf_counter() - start_time, 3)

        return {
            "response": response_text,
            "generation_time_seconds": elapsed_sec,
            "error": None
        }

    except Exception as e:
        elapsed_sec = round(time.perf_counter() - start_time, 3)
        return {
            "response": "",
            "generation_time_seconds": elapsed_sec,
            "error": str(e)
        }
