# 语言模型高效推理：KV Cache 压缩实验报告

## 摘要

本项目实现并评估了 `EleutherAI/pythia-70m` 上的 training-free KV Cache 压缩方法。实验不训练模型、不修改模型参数，只在推理阶段选择和裁剪 `past_key_values`。目标是比较不同 cache policy 在 WikiText 和 PG-19 上的困惑度、保留 KV token 数和生成 latency 指标之间的权衡。

本仓库是课程项目级 Python 实现，评估采用清晰的 token-by-token cached loop，便于复现实验和检查每一步 cache 裁剪行为。

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

- WikiText：`Salesforce/wikitext`，config 为 `wikitext-2-raw-v1`，`validation` split，`max_samples=16`，`max_chars=200000`，`max_tokens=1024`。
- PG-19：`test` split，单个真实长文本 sample，`max_samples=1`，`max_chars=200000`，`max_tokens=1024`。
- cache 参数（PPL 和 latency 均使用）：`window_size=240`，`sink_size=8`，`important_size=40`。本次重新实验将旧设置 `256/4/32` 调整为 `240/8/40`，总 KV budget 接近原设置，但把一部分 recent window 预算转给 sink token 和 attention-selected token。
- dtype：`float32`。当前环境中 Pythia-70M 的 float16 logits 会出现非有限值，因此正式实验使用 float32 保证 JSON 结果有效。

正式 latency 设置：

- WikiText：`validation` split，`max_samples=16`，`max_chars=200000`，`max_prompt_tokens=512`，`max_new_tokens=64`。
- PG-19：`test` split，`max_samples=1`，`max_chars=200000`，`max_prompt_tokens=512`，`max_new_tokens=64`。
- 指标：TTFT、TPOT、new-token throughput、end-to-end throughput、peak CUDA memory。
- 设备：`cuda:0`。

复现性固定项：

- 随机种子：`scripts/run_ppl.py` 和 `scripts/run_latency.py` 都调用 `set_seed(0)`；模型处于 `eval()`，latency 使用 greedy `argmax` 解码。
- 模型和 tokenizer：`EleutherAI/pythia-70m`。本次运行的本地 Hugging Face snapshot 为 `a39f36b100fe8a5377810d56c3f4789b9c53ac42`。
- WikiText：本次运行的本地 Hugging Face dataset snapshot 为 `b08601e04326c79dfdd32d625aee71d232d685c3`。
- PG-19：新版 `datasets` 不再支持 Hugging Face 上旧式 `pg19.py` dataset script。本项目会 fallback 到 `deepmind/pg19` 的官方 split 文件列表；本次运行的 `deepmind/pg19` snapshot 为 `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`，实际选择排序后的第一个 test 文件 `test_10146.txt`，下载到 `data/pg19_raw/`。该目录不会被提交。
- 上述 snapshot 用于标识本次实验实际使用的上游文件；如果未来 Hugging Face 上游内容发生变化，需要确保本地缓存或下载版本仍对应这些 snapshot。
- 环境快照：`results/raw/environment.json` 记录本次 Python、Linux kernel、CUDA、GPU 和关键包版本；当前正式结果来自 `2026-05-14T02:18:01.279238+00:00` 生成的环境快照。
- latency 结果不是 bitwise deterministic。即使参数、模型和数据相同，TTFT/TPOT/throughput 也会随 GPU 负载、驱动调度和缓存状态有小幅波动；PPL 结果更适合作为严格数值复现目标。

## 效果摘要

下面的百分比均由正式 JSON 结果计算得出。`PPL Δ` 越低越好，`avg KV reduction` 表示相对 `dense` 的平均保留 KV token 降低比例。

| dataset | method | PPL | PPL Δ vs dense | avg KV reduction |
| --- | --- | --- | --- | --- |
| WikiText | sliding_window | 42.9560 | +42.49% | 58.60% |
| WikiText | streamingllm | 35.2546 | +16.94% | 57.41% |
| WikiText | snapkv_lite | 36.1304 | +19.85% | 52.77% |
| WikiText | sink_snapkv | 34.9901 | +16.06% | 51.64% |
| PG-19 | sliding_window | 37.3152 | +19.98% | 58.60% |
| PG-19 | streamingllm | 31.9968 | +2.88% | 57.41% |
| PG-19 | snapkv_lite | 31.4713 | +1.19% | 52.77% |
| PG-19 | sink_snapkv | 31.3827 | +0.91% | 51.64% |

加速结果以 new-token throughput 相对 `dense` 的变化衡量。注意 attention-based 方法需要额外请求 attention weights，因此吞吐不一定更高。

| dataset | method | throughput | throughput Δ vs dense | TPOT Δ vs dense | avg KV reduction |
| --- | --- | --- | --- | --- | --- |
| WikiText | sliding_window | 126.99 tok/s | -10.77% | +15.73% | 55.84% |
| WikiText | streamingllm | 147.68 tok/s | +3.77% | -1.12% | 54.37% |
| WikiText | snapkv_lite | 123.91 tok/s | -12.93% | +16.14% | 48.48% |
| WikiText | sink_snapkv | 122.92 tok/s | -13.63% | +19.25% | 47.01% |
| PG-19 | sliding_window | 121.80 tok/s | -13.97% | +18.52% | 55.84% |
| PG-19 | streamingllm | 123.79 tok/s | -12.56% | +15.54% | 54.37% |
| PG-19 | snapkv_lite | 107.05 tok/s | -24.39% | +32.76% | 48.48% |
| PG-19 | sink_snapkv | 104.23 tok/s | -26.38% | +37.58% | 47.01% |

主要观察：

- `sliding_window` 压缩最强，但 PPL 损失也最大，尤其在 WikiText 上 PPL 从 `30.1470` 升至 `42.9560`。
- `streamingllm` 在 WikiText 上保持了较好的质量，PPL 比 dense 高 `16.94%`，同时平均 KV token 减少 `57.41%`。
- `sink_snapkv` 在 PG-19 上最接近 dense PPL，只高 `0.91%`，平均 KV token 减少 `51.64%`。
- `sink_snapkv` 在 WikiText 上是压缩方法中 PPL 最低的一个，PPL 为 `34.9901`，但由于 attention bookkeeping，latency throughput 低于 dense。
- 本次 latency 重测中，只有 WikiText 上的 `streamingllm` new-token throughput 高于 dense，提升 `3.77%`；PG-19 上所有压缩策略的吞吐均低于 dense。
- attention-based 方法虽然保留了更有选择性的中间 token，但请求 attention weights 的额外开销抵消了小模型上的部分速度收益。

## 自动结果表

下面的表格由 `results/raw/*.json` 自动生成。`ppl_wikitext.json`、`ppl_pg19.json`、`latency_wikitext.json` 和 `latency_pg19.json` 是正式 WikiText/PG-19 实验；`*_smoke.json` 只用于流程检查。

<!-- RESULTS_START -->
### Perplexity Results

| dataset | split | method | ppl | mean_nll | max_kv | avg_kv | device | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | dense | 31.1004 | 3.4372 | 1023 | 512.00 | cuda:0 | ppl_pg19.json |
| pg19 | test | sliding_window | 37.3152 | 3.6194 | 240 | 211.96 | cuda:0 | ppl_pg19.json |
| pg19 | test | streamingllm | 31.9968 | 3.4656 | 248 | 218.06 | cuda:0 | ppl_pg19.json |
| pg19 | test | snapkv_lite | 31.4713 | 3.4491 | 280 | 241.82 | cuda:0 | ppl_pg19.json |
| pg19 | test | sink_snapkv | 31.3827 | 3.4463 | 288 | 247.60 | cuda:0 | ppl_pg19.json |
| text | validation | dense | 119.0749 | 4.7798 | 63 | 32.00 | cuda:0 | ppl_smoke.json |
| text | validation | sliding_window | 156.3245 | 5.0519 | 16 | 14.10 | cuda:0 | ppl_smoke.json |
| text | validation | streamingllm | 137.8975 | 4.9265 | 18 | 15.57 | cuda:0 | ppl_smoke.json |
| text | validation | snapkv_lite | 131.1646 | 4.8765 | 20 | 16.98 | cuda:0 | ppl_smoke.json |
| text | validation | sink_snapkv | 130.3357 | 4.8701 | 22 | 18.33 | cuda:0 | ppl_smoke.json |
| wikitext | validation | dense | 30.1470 | 3.4061 | 1023 | 512.00 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | sliding_window | 42.9560 | 3.7602 | 240 | 211.96 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | streamingllm | 35.2546 | 3.5626 | 248 | 218.06 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | snapkv_lite | 36.1304 | 3.5871 | 280 | 241.82 | cuda:0 | ppl_wikitext.json |
| wikitext | validation | sink_snapkv | 34.9901 | 3.5551 | 288 | 247.60 | cuda:0 | ppl_wikitext.json |

### Latency Results

| dataset | split | method | TTFT ms | TPOT ms | new tok/s | e2e tok/s | peak CUDA MB | device | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | dense | 25.29 | 6.77 | 141.58 | 1274.18 | 394.99 | cuda:0 | latency_pg19.json |
| pg19 | test | sliding_window | 19.66 | 8.03 | 121.80 | 1096.23 | 390.69 | cuda:0 | latency_pg19.json |
| pg19 | test | streamingllm | 23.90 | 7.83 | 123.79 | 1114.15 | 390.69 | cuda:0 | latency_pg19.json |
| pg19 | test | snapkv_lite | 31.26 | 8.99 | 107.05 | 963.46 | 454.69 | cuda:0 | latency_pg19.json |
| pg19 | test | sink_snapkv | 26.88 | 9.32 | 104.23 | 938.08 | 454.69 | cuda:0 | latency_pg19.json |
| text | validation | dense | 13.02 | 7.44 | 122.83 | 1105.44 | 294.25 | cuda:0 | latency_smoke.json |
| text | validation | streamingllm | 9.12 | 5.38 | 171.06 | 1539.55 | 293.55 | cuda:0 | latency_smoke.json |
| text | validation | sink_snapkv | 20.13 | 8.17 | 103.51 | 931.57 | 294.30 | cuda:0 | latency_smoke.json |
| wikitext | validation | dense | 32.59 | 6.62 | 142.31 | 1280.82 | 394.99 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sliding_window | 21.25 | 7.66 | 126.99 | 1142.89 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | streamingllm | 20.95 | 6.55 | 147.68 | 1329.08 | 390.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | snapkv_lite | 32.06 | 7.69 | 123.91 | 1115.16 | 454.69 | cuda:0 | latency_wikitext.json |
| wikitext | validation | sink_snapkv | 23.25 | 7.90 | 122.92 | 1106.26 | 454.69 | cuda:0 | latency_wikitext.json |

<!-- RESULTS_END -->

## 复现方式

依赖文件分工：

- `pyproject.toml` 描述本项目的本地包和基本依赖，用于 `pip install -e .`。
- `requirements.txt` 固定了本次报告实际使用的实验环境版本，包括 `torch==2.6.0+cu124`、`transformers==5.8.0`、`datasets==4.8.5` 等。
- README 使用 `requirements.txt` 作为主安装入口，是为了让复现实验时尽量得到和本报告一致的依赖版本。

最小环境配置：

```bash
uv python install 3.11.14
uv venv --python 3.11.14 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
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
make DEVICE=cuda:0 DTYPE=float32 ppl-wikitext
make DEVICE=cuda:0 DTYPE=float32 ppl-pg19
make DEVICE=cuda:0 DTYPE=float32 latency-wikitext
make DEVICE=cuda:0 DTYPE=float32 latency-pg19
make report
```

`make report` 会先写入 `results/raw/environment.json`，记录 Python、CUDA、GPU 名称和关键包版本，然后再更新 README 自动结果表。

正式实验默认使用 `MODEL=EleutherAI/pythia-70m`、`WINDOW_SIZE=240`、`SINK_SIZE=8`、`IMPORTANT_SIZE=40`、`DTYPE=float32`。`DEVICE` 默认为 `auto`，本报告的正式结果是在 `cuda:0` 上生成；严格复现时建议显式传入 `DEVICE=cuda:0`。

也可以直接调用脚本。以下四条命令对应本报告的四个正式 JSON 结果文件：

```bash
python scripts/run_ppl.py --model EleutherAI/pythia-70m --dataset wikitext --split validation --max-samples 16 --max-chars 200000 --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 240 --sink-size 8 --important-size 40 --device cuda:0 --dtype float32 --output results/raw/ppl_wikitext.json
python scripts/run_ppl.py --model EleutherAI/pythia-70m --dataset pg19 --split test --max-samples 1 --max-chars 200000 --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 240 --sink-size 8 --important-size 40 --device cuda:0 --dtype float32 --output results/raw/ppl_pg19.json
python scripts/run_latency.py --model EleutherAI/pythia-70m --dataset wikitext --split validation --max-samples 16 --max-chars 200000 --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 240 --sink-size 8 --important-size 40 --device cuda:0 --dtype float32 --output results/raw/latency_wikitext.json
python scripts/run_latency.py --model EleutherAI/pythia-70m --dataset pg19 --split test --max-samples 1 --max-chars 200000 --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 240 --sink-size 8 --important-size 40 --device cuda:0 --dtype float32 --output results/raw/latency_pg19.json
python scripts/collect_env.py --output results/raw/environment.json
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
