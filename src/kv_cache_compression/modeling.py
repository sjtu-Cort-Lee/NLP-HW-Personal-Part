"""Model and tokenizer loading helpers."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m"


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    return device


def resolve_dtype(
    dtype_arg: Literal["auto", "float16", "bfloat16", "float32"] | str = "auto",
    device: torch.device | str = "cpu",
) -> torch.dtype:
    device = torch.device(device)
    if dtype_arg == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_arg!r}. Choose auto, float16, bfloat16, or float32.")
    if device.type == "cpu" and dtype_arg == "float16":
        raise ValueError("float16 CPU inference is not supported reliably; use float32 or auto.")
    return mapping[dtype_arg]


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    dtype: str = "auto",
    need_attentions: bool = False,
):
    """Load a causal LM and tokenizer for deterministic inference experiments."""

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    if need_attentions:
        kwargs["attn_implementation"] = "eager"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **legacy_kwargs)
        except TypeError:
            legacy_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **legacy_kwargs)

    model.to(resolved_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
        if need_attentions:
            model.config.output_attentions = True
    return model, tokenizer, resolved_device
