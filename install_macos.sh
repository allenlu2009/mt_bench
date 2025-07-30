#!/bin/bash
# Installation script for macOS systems (CPU only)

echo "Installing PyTorch CPU version for macOS..."
uv pip install torch torchvision torchaudio

echo "Installing remaining requirements..."
uv pip install -r requirements_no_flash.txt

echo "Installation complete!"
echo "Note: Flash Attention is not available on macOS. GPU acceleration not supported."