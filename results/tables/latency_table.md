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
