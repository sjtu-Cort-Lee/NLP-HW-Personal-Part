#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/run_ppl.py \
  --dataset text \
  --text-file data/pg19_sample_tiny.txt \
  --max-tokens 64 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 16 \
  --sink-size 2 \
  --important-size 4 \
  --dtype float32 \
  --output results/raw/ppl_smoke.json

"${PYTHON}" scripts/run_latency.py \
  --dataset text \
  --text-file data/pg19_sample_tiny.txt \
  --max-prompt-tokens 64 \
  --max-new-tokens 8 \
  --methods dense streamingllm sink_snapkv \
  --window-size 16 \
  --sink-size 2 \
  --important-size 4 \
  --dtype float32 \
  --output results/raw/latency_smoke.json

"${PYTHON}" scripts/summarize_results.py
