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
