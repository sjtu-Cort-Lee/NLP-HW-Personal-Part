#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"${PYTHON}" -m pytest -q
bash scripts/run_smoke.sh
