import torch

from kv_cache_compression.cache import CachePolicyConfig, select_keep_indices


def test_dense_selection_keeps_everything():
    config = CachePolicyConfig(method="dense")
    assert select_keep_indices(config, cache_length=6) == [0, 1, 2, 3, 4, 5]


def test_sliding_window_selection_keeps_recent_tokens():
    config = CachePolicyConfig(method="sliding_window", window_size=4)
    assert select_keep_indices(config, cache_length=10) == [6, 7, 8, 9]


def test_streamingllm_selection_keeps_sink_and_recent():
    config = CachePolicyConfig(method="streamingllm", sink_size=2, window_size=3)
    assert select_keep_indices(config, cache_length=10) == [0, 1, 7, 8, 9]


def test_sink_snapkv_selection_uses_synthetic_importance():
    importance = torch.zeros(12)
    importance[4] = 10.0
    importance[6] = 9.0
    importance[8] = 8.0
    config = CachePolicyConfig(
        method="sink_snapkv",
        sink_size=2,
        window_size=3,
        important_size=2,
    )
    assert select_keep_indices(config, cache_length=12, importance=importance) == [
        0,
        1,
        4,
        6,
        9,
        10,
        11,
    ]


def test_no_duplicate_indices_when_regions_overlap():
    config = CachePolicyConfig(method="streamingllm", sink_size=4, window_size=4)
    keep = select_keep_indices(config, cache_length=6)
    assert keep == [0, 1, 2, 3, 4, 5]
    assert len(keep) == len(set(keep))


def test_chronological_order_for_attention_policy():
    importance = torch.arange(20, dtype=torch.float32)
    config = CachePolicyConfig(
        method="snapkv_lite",
        window_size=4,
        important_size=3,
    )
    keep = select_keep_indices(config, cache_length=20, importance=importance)
    assert keep == sorted(keep)
    assert len(keep) == len(set(keep))
