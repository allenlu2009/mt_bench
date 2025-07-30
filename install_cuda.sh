#!/bin/bash
# Installation script for CUDA 12.1 systems (RTX 3060)

echo "Installing PyTorch with CUDA 12.1 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Flash Attention 2 (requires nvcc and CUDA_HOME)..."
# Set CUDA_HOME if not already set
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
uv pip install flash-attn --no-build-isolation

echo "Installing remaining requirements..."
uv pip install -r requirements_no_flash.txt

echo "Installation complete!"
echo "To verify GPU support, run: python -c 'import torch; print(torch.cuda.is_available())'"