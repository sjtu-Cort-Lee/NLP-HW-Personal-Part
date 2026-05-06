"""Perplexity and latency evaluation loops."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Sequence

import psutil
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .cache import CachePolicyConfig, KVCacheRuntime, infer_cache_length


@dataclass
class PerplexityResult:
    model: str
    dataset: str
    split: str
    method: str
    token_count: int
    prediction_count: int
    sample_count: int
    mean_nll: float
    ppl: float
    max_retained_kv_tokens: int
    avg_retained_kv_tokens: float
    device: str
    dtype: str
    window_size: int
    sink_size: int
    important_size: int
    elapsed_seconds: float
    attention_fallback_used: bool = False
    note: str = ""


@dataclass
class LatencyResult:
    model: str
    dataset: str
    split: str
    method: str
    prompt_tokens: int
    generated_tokens: int
    ttft_seconds: float
    tpot_seconds: float
    throughput_tokens_per_second: float
    end_to_end_tokens_per_second: float
    peak_cuda_memory_mb: float | None
    max_retained_kv_tokens: int
    avg_retained_kv_tokens: float
    device: str
    dtype: str
    window_size: int
    sink_size: int
    important_size: int
    elapsed_seconds: float
    attention_fallback_used: bool = False
    note: str = ""


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _dtype_of(model: torch.nn.Module) -> str:
    return str(next(model.parameters()).dtype).replace("torch.", "")


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_texts(tokenizer: Any, texts: Sequence[str], max_tokens: int) -> torch.Tensor:
    joined = "\n\n".join(text.strip() for text in texts if text and text.strip())
    if not joined:
        raise ValueError("No non-empty text was provided for evaluation.")
    encoded = tokenizer(
        joined,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to evaluate next-token perplexity.")
    return input_ids


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    config: CachePolicyConfig,
    *,
    model_name: str,
    dataset: str,
    split: str = "validation",
    max_tokens: int = 1024,
    show_progress: bool = False,
) -> PerplexityResult:
    """Evaluate next-token PPL with token-by-token cached inference."""

    device = _device_of(model)
    dtype = _dtype_of(model)
    input_ids = _encode_texts(tokenizer, texts, max_tokens=max_tokens).to(device)
    runtime = KVCacheRuntime(config)
    losses: list[float] = []
    past_key_values: Any = None
    start_time = time.perf_counter()

    iterator = range(input_ids.shape[1] - 1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"ppl:{config.method}", leave=False)

    with torch.inference_mode():
        for position in iterator:
            current = input_ids[:, position : position + 1]
            target = input_ids[:, position + 1]
            past_length = infer_cache_length(past_key_values)
            attention_mask = torch.ones((1, past_length + 1), device=device, dtype=torch.long)
            position_ids = torch.tensor([[position]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=runtime.needs_attention,
            )
            logits = outputs.logits[:, -1, :]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "Non-finite logits encountered during PPL evaluation at "
                    f"position {position} with dtype={dtype} on device={device}. "
                    "Re-run with --dtype float32 or --dtype bfloat16 for numeric stability."
                )
            loss = F.cross_entropy(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss encountered during PPL evaluation at "
                    f"position {position} with dtype={dtype} on device={device}."
                )
            losses.append(float(loss.detach().cpu()))
            attentions = getattr(outputs, "attentions", None)
            past_key_values = runtime.prune(outputs.past_key_values, attentions=attentions)

    _sync_if_needed(device)
    elapsed = time.perf_counter() - start_time
    mean_nll = float(sum(losses) / len(losses))
    ppl = float(math.exp(mean_nll)) if mean_nll < 80 else float("inf")
    note = (
        "attention weights unavailable; fallback kept only sink/recent tokens"
        if runtime.attention_fallback_used
        else ""
    )
    return PerplexityResult(
        model=model_name,
        dataset=dataset,
        split=split,
        method=config.method,
        token_count=int(input_ids.shape[1]),
        prediction_count=len(losses),
        sample_count=len(texts),
        mean_nll=mean_nll,
        ppl=ppl,
        max_retained_kv_tokens=runtime.max_retained,
        avg_retained_kv_tokens=runtime.avg_retained,
        device=str(device),
        dtype=dtype,
        window_size=config.window_size,
        sink_size=config.sink_size,
        important_size=config.important_size,
        elapsed_seconds=elapsed,
        attention_fallback_used=runtime.attention_fallback_used,
        note=note,
    )


def evaluate_latency(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    config: CachePolicyConfig,
    *,
    model_name: str,
    dataset: str,
    split: str = "validation",
    max_prompt_tokens: int = 512,
    max_new_tokens: int = 64,
) -> LatencyResult:
    """Measure greedy generation latency with optional KV cache pruning."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    device = _device_of(model)
    dtype = _dtype_of(model)
    prompt_ids = _encode_texts(tokenizer, texts, max_tokens=max_prompt_tokens).to(device)
    prompt_length = int(prompt_ids.shape[1])
    runtime = KVCacheRuntime(config)

    warmup_length = min(prompt_length, 4)
    with torch.inference_mode():
        warmup_ids = prompt_ids[:, :warmup_length]
        warmup_position_ids = torch.arange(warmup_length, device=device, dtype=torch.long).unsqueeze(0)
        warmup_attention_mask = torch.ones((1, warmup_length), device=device, dtype=torch.long)
        warmup_outputs = model(
            input_ids=warmup_ids,
            attention_mask=warmup_attention_mask,
            position_ids=warmup_position_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=runtime.needs_attention,
        )
        warmup_logits = warmup_outputs.logits[:, -1, :]
        if not torch.isfinite(warmup_logits).all():
            raise FloatingPointError(
                "Non-finite logits encountered during latency warmup with "
                f"dtype={dtype} on device={device}. Re-run with --dtype float32 "
                "or --dtype bfloat16 for numeric stability."
            )
        del warmup_outputs
    _sync_if_needed(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    generated: list[int] = []
    past_key_values: Any = None
    _sync_if_needed(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        position_ids = torch.arange(prompt_length, device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, prompt_length), device=device, dtype=torch.long)
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=runtime.needs_attention,
        )
        logits = outputs.logits[:, -1, :]
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite logits encountered during latency prefill with "
                f"dtype={dtype} on device={device}. Re-run with --dtype float32 "
                "or --dtype bfloat16 for numeric stability."
            )
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        past_key_values = runtime.prune(outputs.past_key_values, attentions=getattr(outputs, "attentions", None))
        _sync_if_needed(device)
        first_token_time = time.perf_counter()

        current = next_token
        absolute_position = prompt_length
        for _ in range(1, max_new_tokens):
            past_length = infer_cache_length(past_key_values)
            attention_mask = torch.ones((1, past_length + 1), device=device, dtype=torch.long)
            position_ids = torch.tensor([[absolute_position]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=runtime.needs_attention,
            )
            logits = outputs.logits[:, -1, :]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "Non-finite logits encountered during latency decode at "
                    f"absolute position {absolute_position} with dtype={dtype} on device={device}. "
                    "Re-run with --dtype float32 or --dtype bfloat16 for numeric stability."
                )
            current = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(int(current.item()))
            past_key_values = runtime.prune(outputs.past_key_values, attentions=getattr(outputs, "attentions", None))
            absolute_position += 1

    _sync_if_needed(device)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = first_token_time - start_time
    tpot = (end_time - first_token_time) / (max_new_tokens - 1) if max_new_tokens > 1 else 0.0
    throughput = max_new_tokens / total_time if total_time > 0 else float("inf")
    end_to_end = (prompt_length + max_new_tokens) / total_time if total_time > 0 else float("inf")
    peak_cuda_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else None
    )
    rss_after = process.memory_info().rss
    note_parts = []
    if device.type != "cuda":
        note_parts.append("CPU run; peak CUDA memory is not available")
    if runtime.attention_fallback_used:
        note_parts.append("attention weights unavailable; fallback kept only sink/recent tokens")
    if device.type == "cpu" and rss_after > rss_before:
        note_parts.append(f"process RSS increased by {(rss_after - rss_before) / (1024 * 1024):.1f} MB")

    return LatencyResult(
        model=model_name,
        dataset=dataset,
        split=split,
        method=config.method,
        prompt_tokens=prompt_length,
        generated_tokens=max_new_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        throughput_tokens_per_second=throughput,
        end_to_end_tokens_per_second=end_to_end,
        peak_cuda_memory_mb=peak_cuda_memory,
        max_retained_kv_tokens=runtime.max_retained,
        avg_retained_kv_tokens=runtime.avg_retained,
        device=str(device),
        dtype=dtype,
        window_size=config.window_size,
        sink_size=config.sink_size,
        important_size=config.important_size,
        elapsed_seconds=total_time,
        attention_fallback_used=runtime.attention_fallback_used,
        note="; ".join(note_parts),
    )


def result_to_dict(result: Any) -> Any:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return {key: result_to_dict(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return [result_to_dict(value) for value in result]
    return result
