#!/usr/bin/env python
"""Write a small reproducibility snapshot for the current environment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import sys
from datetime import datetime, timezone

import torch

from kv_cache_compression.utils import write_json


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "tqdm",
    "numpy",
    "pandas",
    "PyYAML",
    "psutil",
    "pytest",
]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/raw/environment.json")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    write_json(args.output, data)
    print(f"Wrote environment snapshot to {args.output}")


if __name__ == "__main__":
    main()
