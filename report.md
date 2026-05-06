# 个人作业完成报告

## 工作概述

本项目围绕 PPT 中“语言模型高效推理”的个人作业要求，搭建了一个可复现的 KV Cache 压缩实验仓库。实现基于 `EleutherAI/pythia-70m`，不训练模型、不修改模型参数，只在推理阶段对 `past_key_values` 进行选择和裁剪。

已完成的主要工作：

- 创建完整项目结构：`src/kv_cache_compression/`、`scripts/`、`tests/`、`data/`、`results/`、`README.md`、`Makefile` 等。
- 实现 5 种 inference-time KV cache policy：`dense`、`sliding_window`、`streamingllm`、`snapkv_lite`、`sink_snapkv`。
- 实现 PPL 评估脚本：`scripts/run_ppl.py`。
- 实现 latency 评估脚本：`scripts/run_latency.py`，记录 TTFT、TPOT、throughput、end-to-end tokens/s、peak CUDA memory。
- 实现结果汇总脚本：`scripts/summarize_results.py`，自动读取 `results/raw/*.json` 并更新 README 中的结果表格。
- 编写单元测试，覆盖 cache selection、chronological order、无重复 indices、legacy cache pruning、DynamicCache 风格 pruning。
- 配置 Python 3.11.14 虚拟环境，并安装 CUDA 12.4 对应 PyTorch：`torch 2.6.0+cu124`。
- 完成正式 WikiText 和 PG-19 实验，包括 PPL 与 latency。

## 与 PPT 个人部分要求的对照

| PPT 个人部分要求 | 当前完成情况 |
| --- | --- |
| 在 KVPress 或其他加速优化方法中选择并复现/实现算法 | 已实现 `dense` baseline、`sliding_window`、`streamingllm`、`snapkv_lite`，并设计一个小改进 `sink_snapkv`。 |
| 所有实验使用 `Pythia-70M` | 已使用 `EleutherAI/pythia-70m`。 |
| 使用无训练方法进行优化 | 已满足。所有方法都是 training-free inference-time cache policy，不训练、不改模型参数。 |
| 在 PG-19、WikiText 等数据集上做 PPL 测试 | 已完成。结果文件为 `results/raw/ppl_wikitext.json` 和 `results/raw/ppl_pg19.json`。 |
| PG-19 可取单一 sample | 已完成。当前 PG-19 使用 test split 的一个真实文本样本。 |
| 做加速测试 | 已完成。结果文件为 `results/raw/latency_wikitext.json` 和 `results/raw/latency_pg19.json`。 |
| 创建个人独立公开 GitHub 仓库 | 本地仓库内容已准备好；尚未执行 commit 和 push。 |
| README 包含如何运行代码 | 已完成。README 包含安装、smoke、正式 WikiText/PG-19、latency、汇总命令。 |
| README 包含简短报告展示加速/优化效果 | 已完成。README 的自动结果区由真实 JSON 生成，并包含 Discussion 和 Limitations。 |
| 可复现性 | 已完成本地可复现脚本、Makefile target、测试和 JSON 结果；最终公开仓库还需要 push。 |

## 已生成的正式结果

正式 PPL 结果：

- `results/raw/ppl_wikitext.json`
- `results/raw/ppl_pg19.json`

正式 latency 结果：

- `results/raw/latency_wikitext.json`
- `results/raw/latency_pg19.json`

Smoke test 结果：

- `results/raw/ppl_smoke.json`
- `results/raw/latency_smoke.json`

README 中的结果表格由 `scripts/summarize_results.py` 自动生成，未手写或编造实验数字。

## 目前保留的简化与说明

- `snapkv_lite` 是课程项目级轻量实现，不是 SnapKV 论文的完整复现。
- `sink_snapkv` 是一个 training-free hybrid policy，结合 attention sink、attention-selected middle tokens 和 recent local window，但它的有效性主要依赖当前实验结果分析。
- 当前实现是 Python token-by-token evaluation，不是高性能 serving 系统。
- attention-based 方法需要请求 attention weights，因此 latency 不一定优于 `dense`。
- 当前 smoke 和正式实验使用 `--dtype float32`，因为本环境中 `Pythia-70M` 的 float16 logits 会出现非有限值；代码已加入保护，避免写出无效 JSON。
- 尚未完成 GitHub push，这不是代码/实验缺口，但仍是最终提交前需要执行的步骤。
