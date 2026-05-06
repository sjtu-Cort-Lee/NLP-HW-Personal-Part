#!/usr/bin/env python
"""Summarize raw JSON results into markdown tables and README."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "results" / "tables"
README = ROOT / "README.md"


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched["_source"] = path.name
                records.append(enriched)
    return records


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, (int, str)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    return f"{number:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "No experiment results have been generated yet.\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def build_ppl_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for row in records:
        if "ppl" not in row:
            continue
        rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("split", "")),
                str(row.get("method", "")),
                _fmt(row.get("ppl"), 4),
                _fmt(row.get("mean_nll"), 4),
                _fmt(row.get("max_retained_kv_tokens"), 0),
                _fmt(row.get("avg_retained_kv_tokens"), 2),
                str(row.get("device", "")),
                str(row.get("_source", "")),
            ]
        )
    return _table(
        [
            "dataset",
            "split",
            "method",
            "ppl",
            "mean_nll",
            "max_kv",
            "avg_kv",
            "device",
            "source",
        ],
        rows,
    )


def build_latency_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for row in records:
        if "ttft_seconds" not in row:
            continue
        rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("split", "")),
                str(row.get("method", "")),
                _fmt(float(row.get("ttft_seconds", 0.0)) * 1000, 2),
                _fmt(float(row.get("tpot_seconds", 0.0)) * 1000, 2),
                _fmt(row.get("throughput_tokens_per_second"), 2),
                _fmt(row.get("end_to_end_tokens_per_second"), 2),
                _fmt(row.get("peak_cuda_memory_mb"), 2),
                str(row.get("device", "")),
                str(row.get("_source", "")),
            ]
        )
    return _table(
        [
            "dataset",
            "split",
            "method",
            "TTFT ms",
            "TPOT ms",
            "new tok/s",
            "e2e tok/s",
            "peak CUDA MB",
            "device",
            "source",
        ],
        rows,
    )


def update_readme(ppl_table: str, latency_table: str) -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    if start not in text or end not in text:
        raise RuntimeError("README.md must contain RESULTS_START and RESULTS_END markers.")
    generated = (
        f"{start}\n"
        "### Perplexity Results\n\n"
        f"{ppl_table}\n"
        "### Latency Results\n\n"
        f"{latency_table}\n"
        f"{end}"
    )
    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]
    README.write_text(before + generated + after, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_records()
    ppl_table = build_ppl_table(records)
    latency_table = build_latency_table(records)
    (TABLE_DIR / "ppl_table.md").write_text(ppl_table, encoding="utf-8")
    (TABLE_DIR / "latency_table.md").write_text(latency_table, encoding="utf-8")
    update_readme(ppl_table, latency_table)
    print(f"Wrote {TABLE_DIR / 'ppl_table.md'}")
    print(f"Wrote {TABLE_DIR / 'latency_table.md'}")


if __name__ == "__main__":
    main()
