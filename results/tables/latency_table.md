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
