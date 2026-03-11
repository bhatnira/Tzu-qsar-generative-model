#!/usr/bin/env bash
set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
echo "Creating conda environment 'qsar-env' from environment.yml (requires conda or mamba)..."
if command -v mamba >/dev/null 2>&1; then
  mamba env create -f "$WORKDIR/environment.yml" -n qsar-env || mamba env update -f "$WORKDIR/environment.yml" -n qsar-env
else
  conda env create -f "$WORKDIR/environment.yml" -n qsar-env || conda env update -f "$WORKDIR/environment.yml" -n qsar-env
fi
echo "Environment created. Activate with: conda activate qsar-env"
echo "Then run: python main.py (ensure you set the Excel filename in main.py)
" 