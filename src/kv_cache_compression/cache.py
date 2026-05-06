"""KV cache selection and pruning utilities.

The implementation targets batch size 1 and the legacy Hugging Face
``past_key_values`` layout used by GPT-NeoX/Pythia:
``(layer_key, layer_value)`` with tensors shaped
``[batch, heads, seq_len, head_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Sequence

import torch

SUPPORTED_METHODS = (
    "dense",
    "sliding_window",
    "streamingllm",
    "snapkv_lite",
    "sink_snapkv",
)
ATTENTION_METHODS = {"snapkv_lite", "sink_snapkv"}


@dataclass(frozen=True)
class CachePolicyConfig:
    """Configuration for a training-free KV cache policy."""

    method: str = "dense"
    window_size: int = 256
    sink_size: int = 4
    important_size: int = 32
    allow_attention_fallback: bool = True

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported cache method {self.method!r}. "
                f"Supported methods: {', '.join(SUPPORTED_METHODS)}"
            )
        for name in ("window_size", "sink_size", "important_size"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

    @property
    def needs_attention(self) -> bool:
        return self.method in ATTENTION_METHODS


def _policy_budget(config: CachePolicyConfig) -> int | None:
    if config.method == "dense":
        return None
    if config.method == "sliding_window":
        return config.window_size
    if config.method == "streamingllm":
        return config.sink_size + config.window_size
    if config.method == "snapkv_lite":
        return config.important_size + config.window_size
    if config.method == "sink_snapkv":
        return config.sink_size + config.important_size + config.window_size
    raise ValueError(f"Unsupported cache method: {config.method}")


def _sorted_unique(indices: Iterable[int]) -> list[int]:
    return sorted({int(index) for index in indices})


def _importance_to_scores(importance: Sequence[float] | torch.Tensor, cache_length: int) -> list[float]:
    if torch.is_tensor(importance):
        values = importance.detach().float().cpu().flatten().tolist()
    else:
        values = [float(value) for value in importance]
    if len(values) != cache_length:
        raise ValueError(
            f"Importance length ({len(values)}) must match cache length ({cache_length})."
        )
    cleaned: list[float] = []
    for value in values:
        cleaned.append(value if isfinite(value) else float("-inf"))
    return cleaned


def _select_topk_by_importance(
    importance: Sequence[float] | torch.Tensor | None,
    candidate_indices: Sequence[int],
    k: int,
    cache_length: int,
    allow_fallback: bool,
) -> list[int]:
    if k <= 0 or not candidate_indices:
        return []
    if importance is None:
        if allow_fallback:
            return []
        raise ValueError(
            "Attention importance is required for attention-based cache selection. "
            "Enable output_attentions=True or allow fallback selection."
        )
    scores = _importance_to_scores(importance, cache_length)
    ranked = sorted(candidate_indices, key=lambda index: (-scores[index], index))
    return sorted(ranked[:k])


def select_keep_indices(
    config: CachePolicyConfig,
    cache_length: int,
    importance: Sequence[float] | torch.Tensor | None = None,
) -> list[int]:
    """Return sorted token indices to keep for the configured cache policy."""

    if cache_length < 0:
        raise ValueError(f"cache_length must be non-negative, got {cache_length}.")
    if cache_length == 0:
        return []

    budget = _policy_budget(config)
    if budget is None:
        return list(range(cache_length))
    if cache_length <= budget:
        return list(range(cache_length))

    if config.method == "sliding_window":
        start = max(0, cache_length - config.window_size)
        return list(range(start, cache_length))

    if config.method == "streamingllm":
        sink_end = min(config.sink_size, cache_length)
        recent_start = max(sink_end, cache_length - config.window_size)
        return _sorted_unique(
            list(range(0, sink_end)) + list(range(recent_start, cache_length))
        )

    if config.method == "snapkv_lite":
        recent_start = max(0, cache_length - config.window_size)
        middle = list(range(0, recent_start))
        important = _select_topk_by_importance(
            importance,
            middle,
            config.important_size,
            cache_length,
            config.allow_attention_fallback,
        )
        return _sorted_unique(important + list(range(recent_start, cache_length)))

    if config.method == "sink_snapkv":
        sink_end = min(config.sink_size, cache_length)
        recent_start = max(sink_end, cache_length - config.window_size)
        middle = list(range(sink_end, recent_start))
        important = _select_topk_by_importance(
            importance,
            middle,
            config.important_size,
            cache_length,
            config.allow_attention_fallback,
        )
        return _sorted_unique(
            list(range(0, sink_end)) + important + list(range(recent_start, cache_length))
        )

    raise ValueError(f"Unsupported cache method: {config.method}")


def _to_legacy_cache(past_key_values: Any) -> tuple[Any, ...]:
    if past_key_values is None:
        return tuple()
    if isinstance(past_key_values, tuple):
        return past_key_values
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        if legacy is None:
            raise TypeError("DynamicCache.to_legacy_cache() returned None.")
        return tuple(legacy)
    if hasattr(past_key_values, "layers"):
        layers = []
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise TypeError("Cache layers must expose keys and values tensors.")
            layers.append((layer.keys, layer.values))
        return tuple(layers)
    raise TypeError(
        "Unsupported past_key_values type. Expected a legacy tuple/list cache or "
        "a Hugging Face Cache object with to_legacy_cache() or layers."
    )


def infer_cache_length(past_key_values: Any) -> int:
    """Infer the current sequence length stored in a KV cache."""

    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        length = past_key_values.get_seq_length()
        if length is not None:
            return int(length)
    legacy = _to_legacy_cache(past_key_values)
    if not legacy:
        return 0
    first_layer = legacy[0]
    if not isinstance(first_layer, (tuple, list)) or len(first_layer) < 2:
        raise TypeError("Each legacy cache layer must contain at least key and value tensors.")
    key = first_layer[0]
    if not torch.is_tensor(key) or key.ndim < 3:
        raise TypeError("Legacy cache key must be a tensor with a sequence dimension.")
    return int(key.shape[-2])


def _indices_to_list(keep_indices: Sequence[int] | torch.Tensor) -> list[int]:
    if torch.is_tensor(keep_indices):
        return [int(value) for value in keep_indices.detach().cpu().flatten().tolist()]
    return [int(value) for value in keep_indices]


def _validate_indices(indices_list: list[int], cache_length: int) -> None:
    if any(index < 0 or index >= cache_length for index in indices_list):
        raise IndexError(
            f"keep_indices must be within [0, {cache_length}); got {indices_list}."
        )


def _prune_layer_cache(past_key_values: Any, indices_list: list[int]) -> Any:
    cache_length = infer_cache_length(past_key_values)
    _validate_indices(indices_list, cache_length)
    if indices_list == list(range(cache_length)):
        return past_key_values
    for layer in past_key_values.layers:
        if not hasattr(layer, "keys") or not hasattr(layer, "values"):
            raise TypeError("Cache layers must expose keys and values tensors.")
        for name in ("keys", "values"):
            tensor = getattr(layer, name)
            if not torch.is_tensor(tensor):
                raise TypeError(f"Cache layer {name} must be a tensor.")
            seq_dim = tensor.ndim - 2
            if tensor.shape[seq_dim] != cache_length:
                raise ValueError(
                    "Key/value cache tensors must share the same sequence length. "
                    f"Expected {cache_length}, got {tensor.shape[seq_dim]}."
                )
            device_indices = torch.as_tensor(indices_list, dtype=torch.long, device=tensor.device)
            setattr(layer, name, tensor.index_select(seq_dim, device_indices))
    return past_key_values


def prune_legacy_cache(past_key_values: Any, keep_indices: Sequence[int] | torch.Tensor) -> Any:
    """Prune legacy ``past_key_values`` along the sequence dimension."""

    if hasattr(past_key_values, "layers") and not hasattr(past_key_values, "to_legacy_cache"):
        return _prune_layer_cache(past_key_values, _indices_to_list(keep_indices))

    legacy = _to_legacy_cache(past_key_values)
    if not legacy:
        return legacy

    cache_length = infer_cache_length(legacy)
    indices_list = _indices_to_list(keep_indices)
    _validate_indices(indices_list, cache_length)
    if indices_list == list(range(cache_length)):
        return legacy

    pruned_layers: list[Any] = []
    for layer in legacy:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Each legacy cache layer must contain key and value tensors.")
        updated = list(layer)
        for tensor_index in (0, 1):
            tensor = layer[tensor_index]
            if not torch.is_tensor(tensor):
                raise TypeError("Legacy cache key/value entries must be tensors.")
            seq_dim = tensor.ndim - 2
            if tensor.shape[seq_dim] != cache_length:
                raise ValueError(
                    "Key/value cache tensors must share the same sequence length. "
                    f"Expected {cache_length}, got {tensor.shape[seq_dim]}."
                )
            device_indices = torch.as_tensor(indices_list, dtype=torch.long, device=tensor.device)
            updated[tensor_index] = tensor.index_select(seq_dim, device_indices)
        pruned_layers.append(tuple(updated) if isinstance(layer, tuple) else updated)
    return tuple(pruned_layers)


def summarize_attention_importance(attentions: Any) -> torch.Tensor | None:
    """Average attention weights into one importance score per key token."""

    if attentions is None:
        return None
    layer_scores: list[torch.Tensor] = []
    for attention in attentions:
        if attention is None:
            continue
        if not torch.is_tensor(attention):
            continue
        scores = attention.detach().float()
        if scores.ndim < 2:
            continue
        reduce_dims = tuple(range(scores.ndim - 1))
        layer_scores.append(scores.mean(dim=reduce_dims).cpu())
    if not layer_scores:
        return None
    max_length = max(score.numel() for score in layer_scores)
    aligned: list[torch.Tensor] = []
    for score in layer_scores:
        if score.numel() == max_length:
            aligned.append(score)
            continue
        padded = torch.zeros(max_length, dtype=score.dtype)
        padded[-score.numel() :] = score
        aligned.append(padded)
    return torch.stack(aligned, dim=0).mean(dim=0)


class KVCacheRuntime:
    """Small runtime helper that applies one policy after each model step."""

    def __init__(self, config: CachePolicyConfig):
        self.config = config
        self.retained_lengths: list[int] = []
        self.attention_fallback_used = False

    @property
    def needs_attention(self) -> bool:
        return self.config.needs_attention

    def select(self, past_key_values: Any, attentions: Any = None) -> list[int]:
        cache_length = infer_cache_length(past_key_values)
        importance = summarize_attention_importance(attentions) if self.needs_attention else None
        if self.needs_attention and importance is None:
            self.attention_fallback_used = True
        return select_keep_indices(self.config, cache_length, importance=importance)

    def prune(self, past_key_values: Any, attentions: Any = None) -> Any:
        if past_key_values is None:
            self.retained_lengths.append(0)
            return tuple()
        keep_indices = self.select(past_key_values, attentions=attentions)
        pruned = prune_legacy_cache(past_key_values, keep_indices)
        self.retained_lengths.append(infer_cache_length(pruned))
        return pruned

    @property
    def max_retained(self) -> int:
        return max(self.retained_lengths) if self.retained_lengths else 0

    @property
    def avg_retained(self) -> float:
        if not self.retained_lengths:
            return 0.0
        return float(sum(self.retained_lengths) / len(self.retained_lengths))
