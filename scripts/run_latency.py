#!/usr/bin/env python
"""Run greedy generation latency evaluation."""

from __future__ import annotations

import argparse

import torch

from kv_cache_compression.cache import ATTENTION_METHODS, CachePolicyConfig, SUPPORTED_METHODS
from kv_cache_compression.data import load_text_corpus
from kv_cache_compression.eval import evaluate_latency, result_to_dict
from kv_cache_compression.modeling import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from kv_cache_compression.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", choices=["wikitext", "pg19", "text"], default="text")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-file")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=200000)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS))
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--sink-size", type=int, default=8)
    parser.add_argument("--important-size", type=int, default=40)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--output", default="results/raw/latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    texts = load_text_corpus(
        args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        max_chars=args.max_chars,
        text_file=args.text_file,
    )
    need_attentions = any(method in ATTENTION_METHODS for method in args.methods)
    model, tokenizer, _ = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        need_attentions=need_attentions,
    )

    records = []
    for method in args.methods:
        config = CachePolicyConfig(
            method=method,
            window_size=args.window_size,
            sink_size=args.sink_size,
            important_size=args.important_size,
        )
        result = evaluate_latency(
            model,
            tokenizer,
            texts,
            config,
            model_name=args.model,
            dataset=args.dataset,
            split=args.split,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        records.append(result_to_dict(result))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output, records)
    print(f"Wrote {len(records)} latency records to {args.output}")


if __name__ == "__main__":
    main()
