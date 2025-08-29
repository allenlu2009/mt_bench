"""Tests for memory monitoring utilities."""

import os
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.utils.memory_utils import MemoryMonitor, get_flash_attention_config, optimize_for_rtx3060


class TestMemoryMonitor:
    """Test cases for MemoryMonitor class."""
    
    def test_init(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(gpu_memory_limit_gb=8.0)
        assert monitor.gpu_memory_limit == 8.0
        assert monitor.peak_memory == 0.0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024**3)  # 2GB
    def test_get_gpu_memory_usage_cuda_available(self, mock_allocated, mock_available):
        """Test GPU memory usage when CUDA is available."""
        monitor = MemoryMonitor()
        usage = monitor.get_gpu_memory_usage()
        assert usage == 2.0  # 2GB
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_gpu_memory_usage_no_cuda(self, mock_available):
        """Test GPU memory usage when CUDA is not available."""
        monitor = MemoryMonitor()
        usage = monitor.get_gpu_memory_usage()
        assert usage == 0.0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_reserved', return_value=1.5 * 1024**3)  # 1.5GB
    def test_get_gpu_memory_cached(self, mock_reserved, mock_available):
        """Test GPU cached memory retrieval."""
        monitor = MemoryMonitor()
        cached = monitor.get_gpu_memory_cached()
        assert cached == 1.5
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_cleanup_gpu_memory(self, mock_sync, mock_empty, mock_available):
        """Test GPU memory cleanup."""
        monitor = MemoryMonitor()
        monitor.cleanup_gpu_memory()
        
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=7 * 1024**3)  # 7GB
    def test_check_memory_limit_exceeded(self, mock_allocated, mock_available):
        """Test memory limit check when limit is exceeded."""
        monitor = MemoryMonitor(gpu_memory_limit_gb=6.0)
        
        with pytest.raises(RuntimeError, match="GPU memory limit exceeded"):
            monitor.check_memory_limit("test operation")
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=4 * 1024**3)  # 4GB
    def test_check_memory_limit_within_bounds(self, mock_allocated, mock_available):
        """Test memory limit check when within bounds."""
        monitor = MemoryMonitor(gpu_memory_limit_gb=6.0)
        
        result = monitor.check_memory_limit("test operation")
        assert result is True
        assert monitor.peak_memory == 4.0
    
    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        monitor = MemoryMonitor()
        
        # Test known models
        assert monitor.estimate_model_memory("gpt2-large") == 1.5
        assert monitor.estimate_model_memory("llama-3.2-1b") == 2.0
        assert monitor.estimate_model_memory("gemma-7b") == 14.0
        
        # Test unknown model (should return default)
        assert monitor.estimate_model_memory("unknown-model") == 4.0
    
    def test_can_load_model(self):
        """Test model loading feasibility check."""
        monitor = MemoryMonitor(gpu_memory_limit_gb=6.0)
        
        # Mock current usage
        with patch.object(monitor, 'get_gpu_memory_usage', return_value=1.0):
            assert monitor.can_load_model("gpt2-large") is True  # 1.0 + 1.5 = 2.5 < 6.0
            assert monitor.can_load_model("gemma-7b") is False   # 1.0 + 14.0 = 15.0 > 6.0
    
    def test_get_memory_optimization_config(self):
        """Test memory optimization configuration."""
        monitor = MemoryMonitor(gpu_memory_limit_gb=6.0)
        config = monitor.get_memory_optimization_config()
        
        required_keys = ["torch_dtype", "low_cpu_mem_usage"]
        for key in required_keys:
            assert key in config
        
        # CUDA-specific keys only present when CUDA is available
        if torch.cuda.is_available():
            cuda_keys = ["device_map", "max_memory"]
            for key in cuda_keys:
                assert key in config
        
        assert config["torch_dtype"] == torch.float16
        
        # Only check CUDA-specific configs when CUDA is available
        if torch.cuda.is_available():
            assert config["max_memory"] == {0: "5.5GB"}  # 6.0 - 0.5


class TestFlashAttentionConfig:
    """Test cases for Flash Attention configuration."""
    
    @patch('builtins.__import__')
    @patch('src.utils.memory_utils.torch.cuda.is_available', return_value=True)
    def test_get_flash_attention_config_available(self, mock_cuda, mock_import):
        """Test Flash Attention config when available."""
        # Mock successful import of flash_attn
        def mock_import_func(name, *args):
            if name == 'flash_attn':
                return MagicMock()
            return __import__(name, *args)
        
        mock_import.side_effect = mock_import_func
        config = get_flash_attention_config()
        
        assert "attn_implementation" in config
        assert config["attn_implementation"] == "flash_attention_2"
        assert config["use_cache"] is True
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_flash_attention_config_not_available(self, mock_cuda):
        """Test Flash Attention config when not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            config = get_flash_attention_config()
            
            assert config["attn_implementation"] == "eager"
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_flash_attention_config_no_cuda(self, mock_cuda):
        """Test Flash Attention config when CUDA not available."""
        config = get_flash_attention_config()
        assert config == {}


class TestRTXOptimization:
    """Test cases for RTX 3060 optimization."""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.set_per_process_memory_fraction')
    @patch.dict('os.environ', {}, clear=True)
    def test_optimize_for_rtx3060(self, mock_set_fraction, mock_cuda):
        """Test RTX 3060 optimization setup."""
        optimize_for_rtx3060()
        
        mock_set_fraction.assert_called_once_with(0.95)
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_optimize_for_rtx3060_no_cuda(self, mock_cuda):
        """Test RTX 3060 optimization when CUDA not available."""
        # Should not raise any errors
        optimize_for_rtx3060()


if __name__ == "__main__":
    pytest.main([__file__])