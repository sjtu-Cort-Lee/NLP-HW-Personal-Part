PYTHON ?= .venv/bin/python
MODEL ?= EleutherAI/pythia-70m

.PHONY: install test smoke ppl-wikitext ppl-pg19 ppl-pg19-tiny latency-wikitext latency-pg19 latency-tiny report quick

install:
	uv python install 3.11.14
	test -d .venv || uv venv --python 3.11.14 .venv
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu124
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/run_ppl.py --model $(MODEL) --dataset text --text-file data/pg19_sample_tiny.txt --max-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 16 --sink-size 2 --important-size 4 --dtype float32 --output results/raw/ppl_smoke.json
	$(PYTHON) scripts/run_latency.py --model $(MODEL) --dataset text --text-file data/pg19_sample_tiny.txt --max-prompt-tokens 64 --max-new-tokens 8 --methods dense streamingllm sink_snapkv --window-size 16 --sink-size 2 --important-size 4 --dtype float32 --output results/raw/latency_smoke.json
	$(PYTHON) scripts/summarize_results.py

ppl-wikitext:
	$(PYTHON) scripts/run_ppl.py --model $(MODEL) --dataset wikitext --split validation --max-samples 16 --max-chars 200000 --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/ppl_wikitext.json

ppl-pg19:
	$(PYTHON) scripts/run_ppl.py --model $(MODEL) --dataset pg19 --split test --max-samples 1 --max-chars 200000 --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/ppl_pg19.json

ppl-pg19-tiny:
	$(PYTHON) scripts/run_ppl.py --model $(MODEL) --dataset text --text-file data/pg19_sample_tiny.txt --max-tokens 1024 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/ppl_pg19_tiny.json

latency-wikitext:
	$(PYTHON) scripts/run_latency.py --model $(MODEL) --dataset wikitext --split validation --max-samples 16 --max-chars 200000 --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 256 --sink-size 4 --important-size 32 --dtype float32 --output results/raw/latency_wikitext.json

latency-pg19:
	$(PYTHON) scripts/run_latency.py --model $(MODEL) --dataset pg19 --split test --max-samples 1 --max-chars 200000 --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --window-size 256 --sink-size 4 --important-size 32 --dtype float32 --output results/raw/latency_pg19.json

latency-tiny:
	$(PYTHON) scripts/run_latency.py --model $(MODEL) --dataset text --text-file data/pg19_sample_tiny.txt --max-prompt-tokens 512 --max-new-tokens 64 --methods dense sliding_window streamingllm snapkv_lite sink_snapkv --dtype float32 --output results/raw/latency_tiny.json

report:
	$(PYTHON) scripts/summarize_results.py

quick: test smoke report
