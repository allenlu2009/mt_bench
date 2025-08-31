#!/bin/bash
# Sync script for cuda

export UV_CACHE_DIR="/home/allen/.cache/uv"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Activate environment and sync
source "./activate-cuda.sh"

# Install to the activated global venv (not local .venv)
# Use uv pip install to avoid markupsafe compatibility issues with uv sync
uv pip install -e . --index-strategy unsafe-best-match

echo "Synced packages for cuda"
echo "PyTorch index used: https://download.pytorch.org/whl/cu124"
