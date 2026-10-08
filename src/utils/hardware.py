import sys

def check_hardware_environment():
    """
    Inspects available hardware (CUDA, GPU device, PyTorch, Transformers).
    Safe to call in any environment without crashing if torch/transformers are missing.
    """
    env_info = {
        "cuda_available": False,
        "device_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
        "torch_version": None,
        "transformers_version": None
    }

    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if env_info["cuda_available"]:
            env_info["device_count"] = torch.cuda.device_count()
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
            env_info["gpu_memory_gb"] = round(total_mem_bytes / (1024 ** 3), 2)
    except ImportError:
        pass

    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    return env_info


def require_remote_gpu():
    """
    Strict enforcement guardrail:
    Ensures that heavy LLM inference is NEVER executed on the local CPU/machine.
    Halts execution immediately if CUDA is not available.
    """
    info = check_hardware_environment()
    if not info["cuda_available"]:
        error_msg = """
===========================================================================
REMOTE NVIDIA GPU REQUIRED — LLM INFERENCE NOT STARTED.
===========================================================================
Target LLM inference (Qwen2.5-3B-Instruct) requires a dedicated
remote NVIDIA GPU runtime (Google Colab / Cloud GPU with CUDA support).

Local execution on CPU/machine is intentionally DISABLED.
Workflow for LLM Evaluation:
  1. Use notebooks/05_full_remote_llm_evaluation.ipynb in Google Colab (T4 GPU)
  2. Save resulting outputs back to: data/processed/llm_evaluation/
===========================================================================
"""
        print(error_msg.strip())
        sys.exit(0)
    
    print(f"[Remote GPU Confirmed] Device: {info['gpu_name']} ({info['gpu_memory_gb']} GB VRAM)")
    return info
