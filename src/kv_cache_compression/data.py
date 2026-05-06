"""Dataset loading utilities for the KV cache compression homework."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

PG19_REPO_ID = "deepmind/pg19"
PG19_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"


def _limit_texts(texts: Iterable[str], max_samples: int | None, max_chars: int | None) -> list[str]:
    selected: list[str] = []
    total_chars = 0
    for text in texts:
        text = str(text).strip()
        if not text:
            continue
        if max_chars is not None and total_chars >= max_chars:
            break
        if max_chars is not None:
            remaining = max_chars - total_chars
            text = text[:remaining]
        selected.append(text)
        total_chars += len(text)
        if max_samples is not None and len(selected) >= max_samples:
            break
    return selected


def _read_text_file(path: str | Path, max_chars: int | None = None) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file does not exist: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    if max_chars is not None:
        text = text[:max_chars]
    return [text]


def _load_pg19_from_asset_list(split: str, max_chars: int | None = None) -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the PG-19 asset-list fallback.") from exc

    split_file = hf_hub_download(
        repo_id=PG19_REPO_ID,
        repo_type="dataset",
        filename=f"data/{split}_files.txt",
    )
    file_names = [
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not file_names:
        raise RuntimeError(f"PG-19 split file list is empty for split={split!r}.")

    selected = sorted(file_names)[0]
    cache_dir = Path("data") / "pg19_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{split}_{Path(selected).name}"
    if not local_path.exists():
        urlretrieve(PG19_ASSET_ROOT_URL + selected, local_path)
    return _read_text_file(local_path, max_chars=max_chars)


def load_wikitext(
    split: str = "validation",
    max_samples: int | None = 16,
    max_chars: int | None = 200000,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load WikiText.") from exc

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return _limit_texts((row["text"] for row in dataset), max_samples, max_chars)


def load_pg19_single(
    split: str = "test",
    max_chars: int | None = 200000,
    text_file: str | Path | None = None,
) -> list[str]:
    if text_file:
        return _read_text_file(text_file, max_chars=max_chars)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load PG-19.") from exc

    try:
        dataset = load_dataset("pg19", split=split, streaming=True)
        for row in dataset:
            text = row.get("text", "")
            if text:
                return _limit_texts([text], max_samples=1, max_chars=max_chars)
    except Exception as exc:  # noqa: BLE001 - give a course-friendly dataset hint.
        try:
            return _load_pg19_from_asset_list(split=split, max_chars=max_chars)
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load PG-19 from Hugging Face and from the DeepMind asset list. "
                "Use --text-file data/pg19_sample_tiny.txt for smoke tests or provide "
                "a local PG-19 text file with --text-file data/pg19_sample.txt."
            ) from fallback_exc
    raise RuntimeError("PG-19 loaded but no non-empty text sample was found.")


def load_text_corpus(
    dataset: str,
    split: str = "validation",
    max_samples: int | None = 16,
    max_chars: int | None = 200000,
    text_file: str | Path | None = None,
) -> list[str]:
    dataset = dataset.lower()
    if dataset == "wikitext":
        return load_wikitext(split=split, max_samples=max_samples, max_chars=max_chars)
    if dataset == "pg19":
        return load_pg19_single(split=split, max_chars=max_chars, text_file=text_file)
    if dataset == "text":
        if not text_file:
            raise ValueError("--dataset text requires --text-file.")
        return _read_text_file(text_file, max_chars=max_chars)
    raise ValueError("Unsupported dataset. Choose one of: wikitext, pg19, text.")
