#!/bin/bash
# Sync script for linux-cuda

export UV_CACHE_DIR="/home/allen/.cache/uv"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"

# Activate environment and sync
source "./activate-linux-cuda.sh"

# Check if UV supports frozen sync
if uv sync --help | grep -q "\--frozen"; then
    uv sync --frozen
else
    uv sync
fi

echo "Synced packages for linux-cuda"
echo "PyTorch index used: https://download.pytorch.org/whl/cu118"
