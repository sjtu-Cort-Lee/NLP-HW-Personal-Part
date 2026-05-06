# 语言模型高效推理：KV Cache 压缩实验报告

## 摘要

本项目实现并评估了 `EleutherAI/pythia-70m` 上的 training-free KV Cache 压缩方法。实验不训练模型、不修改模型参数，只在推理阶段选择和裁剪 `past_key_values`。目标是比较不同 cache policy 在 WikiText 和 PG-19 上的困惑度、保留 KV token 数和生成 latency 指标之间的权衡。

本仓库是课程项目级 Python 实现，不是优化 serving 系统。评估采用清晰的 token-by-token cached loop，便于复现实验和检查每一步 cache 裁剪行为。

## 作业要求对应

- 模型：`EleutherAI/pythia-70m`。
- 优化方式：全部为无训练 inference-time policy。
- 数据集：WikiText validation；PG-19 test 单样本。
- PPL：`scripts/run_ppl.py`。
- 加速指标：`scripts/run_latency.py`，记录 TTFT、TPOT、throughput、end-to-end tokens/s、peak CUDA memory。
- README 报告：本文件中的正式结果表格来自 `results/raw/*.json`，由 `scripts/summarize_results.py` 自动生成。

## 方法

`dense` 是 full-cache baseline，不裁剪 KV cache。

`sliding_window` 只保留最近 `window_size` 个 token。

`streamingllm` 保留前 `sink_size` 个 attention sink token 和最近 `window_size` 个 token。

`snapkv_lite` 是课程项目级轻量 SnapKV 风格策略。它请求 attention weights，用当前 query 对历史 token 的 attention 平均值作为 importance，从中间区域保留 `important_size` 个高分 token，同时保留最近窗口。

`sink_snapkv` 是本项目的小改进：

```text
[sink tokens] + [attention-selected middle tokens] + [recent window]
```

它是一个 training-free hybrid policy，结合 StreamingLLM 的 attention sink、SnapKV 风格 attention-selected memory 和 recent local window。

## 实验设置

正式 PPL 设置：

- WikiText：`validation` split，`max_samples=16`，`max_tokens=1024`。
- PG-19：`test` split，单个真实长文本 sample，`max_tokens=1024`。
- cache 参数：`window_size=256`，`sink_size=4`，`important_size=32`。
- dtype：`float32`。当前环境中 Pythia-70M 的 float16 logits 会出现非有限值，因此正式实验使用 float32 保证 JSON 结果有效。

正式 latency 设置：

- WikiText：`max_prompt_tokens=512`，`max_new_tokens=64`。
- PG-19：`max_prompt_tokens=512`，`max_new_tokens=64`。
- 指标：TTFT、TPOT、new-token throughput、end-to-end throughput、peak CUDA memory。
- 设备：`cuda:0`。

PG-19 加载说明：新版 `datasets` 不再支持 Hugging Face 上旧式 `pg19.py` dataset script。本项目会 fallback 到 `deepmind/pg19` 的官方 split 文件列表，并下载 test split 的真实文本到 `data/pg19_raw/`；该目录不会被提交。

## 效果摘要

下面的百分比均由正式 JSON 结果计算得出。`PPL Δ` 越低越好，`avg KV reduction` 表示相对 `dense` 的平均保留 KV token 降低比例。

| dataset | method | PPL | PPL Δ vs dense | avg KV reduction |
| --- | --- | --- | --- | --- |
| WikiText | sliding_window | 42.9500 | +42.47% | 56.23% |
| WikiText | streamingllm | 36.1872 | +20.04% | 55.65% |
| WikiText | snapkv_lite | 36.1573 | +19.94% | 51.64% |
| WikiText | sink_snapkv | 35.6847 | +18.37% | 51.08% |
| PG-19 | sliding_window | 36.2957 | +16.70% | 56.23% |
| PG-19 | streamingllm | 31.4272 | +1.05% | 55.65% |
| PG-19 | snapkv_lite | 31.2309 | +0.42% | 51.64% |
| PG-19 | sink_snapkv | 31.2339 | +0.43% | 51.08% |

加速结果以 new-token throughput 相对 `dense` 的变化衡量。注意 attention-based 方法需要额外请求 attention weights，因此吞吐不一定更高。

| dataset | method | throughput | throughput Δ vs dense | TPOT Δ vs dense | avg KV reduction |
| --- | --- | --- | --- | --- | --- |
| WikiText | sliding_window | 164.10 tok/s | +0.52% | +2.10% | 52.90% |
| WikiText | streamingllm | 173.83 tok/s | +6.48% | -4.26% | 52.16% |
| WikiText | snapkv_lite | 152.70 tok/s | -6.46% | +7.11% | 47.01% |
| WikiText | sink_snapkv | 152.87 tok/s | -6.36% | +8.69% | 46.27% |
| PG-19 | sliding_window | 167.59 tok/s | -1.40% | +7.88% | 52.90% |
| PG-19 | streamingllm | 180.61 tok/s | +6.26% | -0.08% | 52.16% |
| PG-19 | snapkv_lite | 148.99 tok/s | -12.34% | +12.47% | 47.01% |
| PG-19 | sink_snapkv | 164.04 tok/s | -3.48% | +9.54% | 46.27% |

主要观察：

- `sliding_window` 压缩最强，但 PPL 损失也最大，尤其在 WikiText 上 PPL 从 `30.1470` 升至 `42.9500`。
- `streamingllm` 在 PG-19 上保持了较好的质量，PPL 只比 dense 高 `1.05%`，同时平均 KV token 减少 `55.65%`。
- `snapkv_lite` 和 `sink_snapkv` 在 PG-19 上最接近 dense PPL，分别只高 `0.42%` 和 `0.43%`。
- `sink_snapkv` 在 WikiText 上是压缩方法中 PPL 最低的一个，PPL 为 `35.6847`，但由于 attention bookkeeping，latency throughput 低于 dense。
- 生成 latency 的收益主要出现在 `streamingllm`：WikiText throughput 提升 `6.48%`，PG-19 提升 `6.26%`。
- attention-based 方法虽然保留了更有选择性的中间 token，但请求 attention weights 的额外开销抵消了小模型上的部分速度收益。

## 自动结果表

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
| pg19 | test | dense | 38.03 | 5.37 | 169.97 | 1529.69 | 394.99 | cuda:0 | latency_pg19.json |
| pg19 | test | sliding_window | 16.68 | 5.80 | 167.59 | 1508.33 | 390.69 | cuda:0 | latency_pg19.json |
| pg19 | test | streamingllm | 16.12 | 5.37 | 180.61 | 1625.48 | 390.69 | cuda:0 | latency_pg19.json |
| pg19 | test | snapkv_lite | 48.85 | 6.04 | 148.99 | 1340.88 | 454.69 | cuda:0 | latency_pg19.json |
| pg19 | test | sink_snapkv | 19.34 | 5.89 | 164.04 | 1476.39 | 454.69 | cuda:0 | latency_pg19.json |
| text | validation | dense | 13.02 | 7.44 | 122.83 | 1105.44 | 294.25 | cuda:0 | latency_smoke.json |
| text | validation | streamingllm | 9.12 | 5.38 | 171.06 | 1539.55 | 293.55 | cuda:0 | latency_smoke.json |
| text | validation | sink_snapkv | 20.13 | 8.17 | 103.51 | 931.57 | 294.30 | cuda:0 | latency_smoke.json |
| wikitext | validation | dense | 25.92 | 5.81 | 163.25 | 1469.27 | 394.99 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sliding_window | 16.21 | 5.93 | 164.10 | 1476.90 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | streamingllm | 17.65 | 5.56 | 173.83 | 1564.45 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | snapkv_lite | 26.98 | 6.22 | 152.70 | 1374.30 | 454.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sink_snapkv | 20.71 | 6.32 | 152.87 | 1375.83 | 454.69 | cuda:0 | latency_wikitext.json |

<!-- RESULTS_END -->

## 复现方式

最小环境配置：

```bash
uv python install 3.11.14
uv venv --python 3.11.14 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

Windows PowerShell 只需要把激活命令换成：

```powershell
.venv\Scripts\Activate.ps1
```

运行测试和 smoke：

```bash
make test
make smoke
```

正式实验：

```bash
make ppl-wikitext
make ppl-pg19
make latency-wikitext
make latency-pg19
make report
```

也可以直接调用脚本。示例：

```bash
python scripts/run_ppl.py --dataset wikitext --split validation --max-samples 16 --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/ppl_wikitext.json
python scripts/run_latency.py --dataset pg19 --split test --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/latency_pg19.json
python scripts/summarize_results.py
```

## 限制

- 当前实现只面向 batch size 1。
- `snapkv_lite` 是课程项目级轻量实现，不是 SnapKV 论文完整复现。
- attention-based 方法需要请求 attention weights，会带来额外 latency 和 memory bookkeeping。
- `EleutherAI/pythia-70m` 很小，因此速度差异容易受到 Python loop、CUDA warmup 和 attention 开销影响。
- 本实验统计 peak CUDA memory，但没有统计 FLOPs。
- CPU 环境可以跑通功能测试，但 latency 和 CUDA memory 指标不可直接比较。

## References

- EleutherAI. `pythia-70m`.
- Hugging Face Transformers documentation.
- Hugging Face Datasets documentation.
- Xiao et al. `Efficient Streaming Language Models with Attention Sinks`.
- Li et al. `SnapKV: LLM Knows What You are Looking for Before Generation`.
