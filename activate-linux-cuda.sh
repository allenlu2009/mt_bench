#!/bin/bash
# Activation script for linux-cuda

export UV_CACHE_DIR="/home/allen/.cache/uv"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"

# Activate virtual environment
if [[ "linux-gnu" == "msys" ]] || [[ "linux-gnu" == "win32" ]]; then
    source "/home/allen/.venvs/mt_bench-linux-cuda/Scripts/activate"
else
    source "/home/allen/.venvs/mt_bench-linux-cuda/bin/activate"
fi

echo "Activated environment: mt_bench-linux-cuda"
echo "UV Cache: /home/allen/.cache/uv"
