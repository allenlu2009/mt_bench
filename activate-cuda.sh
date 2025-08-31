#!/bin/bash
# Activation script for cuda

export UV_CACHE_DIR="/home/allen/.cache/uv"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Activate virtual environment
if [[ "linux-gnu" == "msys" ]] || [[ "linux-gnu" == "win32" ]]; then
    source "/home/allen/.venvs/mt_bench-cuda/Scripts/activate"
else
    source "/home/allen/.venvs/mt_bench-cuda/bin/activate"
fi

echo "Activated environment: mt_bench-cuda"
echo "UV Cache: /home/allen/.cache/uv"
