#!/bin/bash
# Simple test runner for debugging environment issues

echo "🔍 Environment Check:"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "🐍 Python modules check:"
python -c "
try:
    import pytest
    print('✅ pytest available:', pytest.__version__)
except ImportError as e:
    print('❌ pytest missing:', e)

try:
    import torch
    print('✅ torch available:', torch.__version__)
except ImportError as e:
    print('⚠️  torch missing:', e)
"

echo ""
echo "📦 Installed packages:"
python -m pip list | grep -E "(pytest|torch)"

echo ""
echo "🧪 Running simple test:"
python -m pytest tests/test_models/test_model_configs.py::TestModelConfig::test_model_config_creation -v