# 语言模型高效推理 / KV Cache 压缩

这是一个 NLP 课程个人作业项目，目标是在 `EleutherAI/pythia-70m` 上实现 training-free inference-time KV cache compression，并用真实脚本输出的 JSON 结果生成报告表格。项目不训练模型、不修改模型参数，只在推理期间选择和裁剪 `past_key_values`。

本仓库是课程项目级 Python 实现，不是高性能 serving 系统。代码刻意保留了清晰的 token-by-token 评估逻辑，便于检查 PPL、latency 和 cache token 数之间的关系。

## 作业要求对应关系

- 模型：默认使用 `EleutherAI/pythia-70m`。
- 方法：实现 `dense`、`sliding_window`、`streamingllm`、`snapkv_lite`、`sink_snapkv`。
- 约束：不训练、不改模型参数，全部方法都是 inference-time policy。
- PPL：`scripts/run_ppl.py` 支持 WikiText、PG-19 和本地 text file。
- 资源指标：`scripts/run_latency.py` 输出 TTFT、TPOT、throughput、end-to-end tokens/s 和 peak CUDA memory；CPU 环境会标记 CUDA memory 不可用。
- 结果诚信：README 结果区域只由 `scripts/summarize_results.py` 从 `results/raw/*.json` 生成；未运行的正式实验不写数字。

## 方法概述

`dense` 是 full-cache baseline，不裁剪 KV cache。

`sliding_window` 只保留最近 `window_size` 个 token，强调局部上下文。

`streamingllm` 保留前 `sink_size` 个 attention sink token 和最近 `window_size` 个 token。

`snapkv_lite` 是课程项目级的轻量 SnapKV 风格策略。它在需要时请求 attention weights，用当前 query 对历史 token 的 attention 平均值作为 importance，从中间区域保留 `important_size` 个高分 token，同时保留最近窗口。

`sink_snapkv` 是本项目的小扩展方法：

```text
[sink tokens] + [attention-selected middle tokens] + [recent window]
```

它是一个 training-free hybrid policy，结合 StreamingLLM 的 attention sink、SnapKV 风格 attention-selected memory 和 recent local window。若当前 transformers/model 路径无法返回 attention weights，attention-based 方法会退化为只保留 sink/recent token，并在 JSON 的 `attention_fallback_used` 和 `note` 字段中记录。

## WSL Ubuntu 安装

推荐使用 `uv` 创建 Python 3.11.14 虚拟环境：

```bash
uv python install 3.11.14
uv venv --python 3.11.14 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

如果机器没有 CUDA，也可以直接执行：

```bash
python -m pip install -e .
```

此时 latency JSON 会把设备标记为 CPU，`peak_cuda_memory_mb` 为 `not available`。

## Windows PowerShell 简要方式

```powershell
uv python install 3.11.14
uv venv --python 3.11.14 .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

## Quick Smoke Test

```bash
make smoke
```

等价于：

```bash
python scripts/run_ppl.py --dataset text --text-file data/pg19_sample_tiny.txt --max-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 16 --sink-size 2 --important-size 4 --dtype float32 --output results/raw/ppl_smoke.json
python scripts/run_latency.py --dataset text --text-file data/pg19_sample_tiny.txt --max-prompt-tokens 64 --max-new-tokens 8 --methods dense streamingllm sink_snapkv --window-size 16 --sink-size 2 --important-size 4 --dtype float32 --output results/raw/latency_smoke.json
python scripts/summarize_results.py
```

Smoke test 只用于验证流程可运行，不是正式实验结论。

## 完整实验命令

WikiText PPL：

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --dataset wikitext \
  --split validation \
  --max-samples 16 \
  --max-chars 200000 \
  --max-tokens 1024 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/ppl_wikitext.json
```

PG-19 PPL。脚本会先尝试 Hugging Face `pg19`，如果当前 `datasets` 版本不再支持旧式 dataset script，会自动使用 `deepmind/pg19` 的官方 split 列表下载 test split 的第一个真实 PG-19 文本到 `data/pg19_raw/`：

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --dataset pg19 \
  --split test \
  --max-samples 1 \
  --max-chars 200000 \
  --max-tokens 1024 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/ppl_pg19.json
```

本地 PG-19 单文本：

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --dataset text \
  --text-file data/pg19_sample.txt \
  --max-samples 1 \
  --max-chars 200000 \
  --max-tokens 1024 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/ppl_pg19_local.json
```

WikiText latency：

```bash
python scripts/run_latency.py \
  --model EleutherAI/pythia-70m \
  --dataset wikitext \
  --split validation \
  --max-samples 16 \
  --max-chars 200000 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/latency_wikitext.json
```

PG-19 latency：

```bash
python scripts/run_latency.py \
  --model EleutherAI/pythia-70m \
  --dataset pg19 \
  --split test \
  --max-samples 1 \
  --max-chars 200000 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/latency_tiny.json
```

Tiny latency smoke：

```bash
python scripts/run_latency.py \
  --model EleutherAI/pythia-70m \
  --dataset text \
  --text-file data/pg19_sample_tiny.txt \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --methods dense sliding_window streamingllm snapkv_lite sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --dtype float32 \
  --output results/raw/latency_tiny.json
```

生成表格并更新 README：

```bash
python scripts/summarize_results.py
```

## 自动结果区域

下面的表格由 `results/raw/*.json` 自动生成。`ppl_wikitext.json`、`ppl_pg19.json`、`latency_wikitext.json` 和 `latency_pg19.json` 是正式 WikiText/PG-19 实验；`*_smoke.json` 只用于流程检查。

<!-- RESULTS_START -->
### Perplexity Results

| dataset | split | method | ppl | mean_nll | max_kv | avg_kv | device | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | dense | 31.1004 | 3.4372 | 1023 | 512.00 | cuda:0 | ppl_pg19.json |
| pg19 | test | sliding_window | 36.2957 | 3.5917 | 256 | 224.09 | cuda:0 | ppl_pg19.json |
| pg19 | test | streamingllm | 31.4272 | 3.4477 | 260 | 227.09 | cuda:0 | ppl_pg19.json |
| pg19 | test | snapkv_lite | 31.2309 | 3.4414 | 288 | 247.60 | cuda:0 | ppl_pg19.json |
| pg19 | test | sink_snapkv | 31.2339 | 3.4415 | 292 | 250.47 | cuda:0 | ppl_pg19.json |
| text | validation | dense | 119.0749 | 4.7798 | 63 | 32.00 | cuda:0 | ppl_smoke.json |
| text | validation | sliding_window | 156.3245 | 5.0519 | 16 | 14.10 | cuda:0 | ppl_smoke.json |
| text | validation | streamingllm | 137.8975 | 4.9265 | 18 | 15.57 | cuda:0 | ppl_smoke.json |
| text | validation | snapkv_lite | 131.1646 | 4.8765 | 20 | 16.98 | cuda:0 | ppl_smoke.json |
| text | validation | sink_snapkv | 130.3357 | 4.8701 | 22 | 18.33 | cuda:0 | ppl_smoke.json |
| wikitext | validation | dense | 30.1470 | 3.4061 | 1023 | 512.00 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | sliding_window | 42.9500 | 3.7600 | 256 | 224.09 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | streamingllm | 36.1872 | 3.5887 | 260 | 227.09 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | snapkv_lite | 36.1573 | 3.5879 | 288 | 247.60 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | sink_snapkv | 35.6847 | 3.5747 | 292 | 250.47 | cuda:0 | ppl_wikitext.json |

### Latency Results

| dataset | split | method | TTFT ms | TPOT ms | new tok/s | e2e tok/s | peak CUDA MB | device | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | dense | 15.18 | 6.50 | 150.78 | 525.37 | 314.72 | cuda:0 | latency_pg19.json |
| pg19 | test | sliding_window | 8.62 | 5.95 | 166.89 | 581.50 | 314.72 | cuda:0 | latency_pg19.json |
| pg19 | test | streamingllm | 8.87 | 5.70 | 174.06 | 606.49 | 314.72 | cuda:0 | latency_pg19.json |
| pg19 | test | snapkv_lite | 20.93 | 6.43 | 150.26 | 523.56 | 319.37 | cuda:0 | latency_pg19.json |
| pg19 | test | sink_snapkv | 11.87 | 6.13 | 160.84 | 560.44 | 319.37 | cuda:0 | latency_pg19.json |
| text | validation | dense | 13.02 | 7.44 | 122.83 | 1105.44 | 294.25 | cuda:0 | latency_smoke.json |
| text | validation | streamingllm | 9.12 | 5.38 | 171.06 | 1539.55 | 293.55 | cuda:0 | latency_smoke.json |
| text | validation | sink_snapkv | 20.13 | 8.17 | 103.51 | 931.57 | 294.30 | cuda:0 | latency_smoke.json |
| wikitext | validation | dense | 25.92 | 5.81 | 163.25 | 1469.27 | 394.99 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sliding_window | 16.21 | 5.93 | 164.10 | 1476.90 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | streamingllm | 17.65 | 5.56 | 173.83 | 1564.45 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | snapkv_lite | 26.98 | 6.22 | 152.70 | 1374.30 | 454.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sink_snapkv | 20.71 | 6.32 | 152.87 | 1375.83 | 454.69 | cuda:0 | latency_wikitext.json |

<!-- RESULTS_END -->

## Discussion

KV cache 压缩的核心权衡是上下文保留质量和推理资源之间的平衡。`dense` 通常给出最稳定的 PPL，因为它保留全部历史 token。`sliding_window` 的 cache tokens 最少，但可能丢掉长程依赖。`streamingllm` 通过保留最前面的 sink token 改善了纯局部窗口的行为。`snapkv_lite` 和 `sink_snapkv` 使用 attention 选择中间 token，理论上可以在相同预算下保留更有用的历史信息。

Latency 结果不一定优于 `dense`。本实现使用 Python token-by-token loop，attention-based 方法还会请求 attention weights 并维护 selection bookkeeping，这些额外开销可能抵消小模型上的 cache 缩减收益。`EleutherAI/pythia-70m` 本身很小，因此加速收益可能不明显；但 `max_retained_kv_tokens`、`avg_retained_kv_tokens` 和 CUDA memory 更能直接体现压缩效果。当前 smoke/Makefile 默认用 `--dtype float32` 是为了避免某些 CUDA/transformers 组合下 Pythia float16 logits 变成非有限值；`auto` 仍会在 CUDA 上选择 float16。

## Limitations

- 当前实现只面向 batch size 1。
- 主要兼容 Hugging Face legacy tuple `past_key_values`，对 `DynamicCache` 会尽量转换为 legacy cache。
- `snapkv_lite` 不是 SnapKV 论文的完整复现，只是课程项目级 attention-selected KV pruning。
- Smoke test 的 `data/pg19_sample_tiny.txt` 不是 PG-19 benchmark 样本。
- CPU 环境可以跑通功能测试，但 latency 和 memory 指标不能代表 CUDA 推理性能。

## References

- EleutherAI. `pythia-70m`.
- Hugging Face Transformers documentation.
- Hugging Face Datasets documentation.
- Xiao et al. `Efficient Streaming Language Models with Attention Sinks`.
- Li et al. `SnapKV: LLM Knows What You are Looking for Before Generation`.
