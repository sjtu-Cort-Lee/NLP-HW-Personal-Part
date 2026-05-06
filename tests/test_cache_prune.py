import torch

from kv_cache_compression.cache import infer_cache_length, prune_legacy_cache


def _make_cache(seq_len=6):
    key = torch.arange(1 * 2 * seq_len * 3).reshape(1, 2, seq_len, 3)
    value = torch.arange(1000, 1000 + 1 * 2 * seq_len * 3).reshape(1, 2, seq_len, 3)
    return ((key, value),)


def test_pruning_tensor_shapes():
    past = _make_cache(seq_len=6)
    pruned = prune_legacy_cache(past, [0, 2, 5])
    key, value = pruned[0]
    assert key.shape == (1, 2, 3, 3)
    assert value.shape == (1, 2, 3, 3)
    assert infer_cache_length(pruned) == 3


def test_pruning_preserves_selected_tensor_values():
    past = _make_cache(seq_len=6)
    keep = torch.tensor([0, 2, 5])
    pruned = prune_legacy_cache(past, keep)
    expected_key = past[0][0].index_select(2, keep)
    expected_value = past[0][1].index_select(2, keep)
    assert torch.equal(pruned[0][0], expected_key)
    assert torch.equal(pruned[0][1], expected_value)


class _FakeDynamicLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeDynamicCache:
    def __init__(self, keys, values):
        self.layers = [_FakeDynamicLayer(keys, values)]

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def test_pruning_dynamic_layer_cache_preserves_object_type_and_values():
    past = _make_cache(seq_len=6)
    key, value = past[0]
    dynamic = _FakeDynamicCache(key.clone(), value.clone())
    keep = torch.tensor([1, 3, 4])
    pruned = prune_legacy_cache(dynamic, keep)
    assert pruned is dynamic
    assert infer_cache_length(pruned) == 3
    assert torch.equal(pruned.layers[0].keys, key.index_select(2, keep))
    assert torch.equal(pruned.layers[0].values, value.index_select(2, keep))
