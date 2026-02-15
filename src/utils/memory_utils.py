"""GPU memory monitoring and optimization utilities."""

import torch
import psutil
import gc
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory statistics for monitoring GPU and system memory usage."""
    gpu_allocated_gb: float
    gpu_cached_gb: float
    gpu_total_gb: float
    system_memory_percent: float
    system_memory_available_gb: float


class MemoryMonitor:
    """Monitor and optimize GPU memory usage for RTX 3060 constraints."""
    
    def __init__(self, gpu_memory_limit_gb: float = 8.0):
        """
        Initialize memory monitor.
        
        Args:
            gpu_memory_limit_gb: GPU memory limit in GB (default: 8.0 for RTX 3060)
        """
        self.gpu_memory_limit = gpu_memory_limit_gb
        self.peak_memory = 0.0
        self.initial_memory = self.get_gpu_memory_usage()
        
    def get_gpu_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB.
        
        Returns:
            GPU memory usage in GB, 0.0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0.0
            
        return torch.cuda.memory_allocated() / (1024**3)
    
    def get_gpu_memory_cached(self) -> float:
        """
        Get current GPU cached memory in GB.
        
        Returns:
            GPU cached memory in GB, 0.0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0.0
            
        return torch.cuda.memory_reserved() / (1024**3)
    
    def get_total_gpu_memory(self) -> float:
        """
        Get total GPU memory in GB.
        
        Returns:
            Total GPU memory in GB, 0.0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0.0
            
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def get_system_memory_stats(self) -> Dict[str, float]:
        """
        Get system memory statistics.
        
        Returns:
            Dictionary with system memory stats
        """
        memory = psutil.virtual_memory()
        return {
            "percent_used": memory.percent,
            "available_gb": memory.available / (1024**3),
            "total_gb": memory.total / (1024**3)
        }
    
    def get_comprehensive_stats(self) -> MemoryStats:
        """
        Get comprehensive memory statistics.
        
        Returns:
            MemoryStats object with all memory information
        """
        system_stats = self.get_system_memory_stats()
        
        return MemoryStats(
            gpu_allocated_gb=self.get_gpu_memory_usage(),
            gpu_cached_gb=self.get_gpu_memory_cached(),
            gpu_total_gb=self.get_total_gpu_memory(),
            system_memory_percent=system_stats["percent_used"],
            system_memory_available_gb=system_stats["available_gb"]
        )
    
    def cleanup_gpu_memory(self) -> None:
        """
        Cleanup GPU memory by emptying CUDA cache.
        
        Critical for RTX 3060 memory management between model loads.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()
    
    def check_memory_limit(self, operation_name: str = "operation") -> bool:
        """
        Check if current memory usage is within limits.
        
        Args:
            operation_name: Name of the operation being checked
            
        Returns:
            True if within limits, False otherwise
            
        Raises:
            RuntimeError: If memory usage exceeds limit
        """
        current_usage = self.get_gpu_memory_usage()
        
        if current_usage > self.gpu_memory_limit:
            raise RuntimeError(
                f"GPU memory limit exceeded during {operation_name}: "
                f"{current_usage:.2f}GB > {self.gpu_memory_limit:.2f}GB"
            )
        
        # Update peak memory tracking
        self.peak_memory = max(self.peak_memory, current_usage)
        return True
    
    def log_memory_usage(self, operation: str, logger=None) -> None:
        """
        Log current memory usage.
        
        Args:
            operation: Description of the current operation
            logger: Logger instance, prints to console if None
        """
        stats = self.get_comprehensive_stats()
        
        # Update peak memory tracking
        self.peak_memory = max(self.peak_memory, stats.gpu_allocated_gb)
        
        # Calculate GPU percentage safely (avoid division by zero)
        gpu_percentage = (
            (stats.gpu_allocated_gb / stats.gpu_total_gb * 100) 
            if stats.gpu_total_gb > 0 else 0.0
        )
        
        message = (
            f"[{operation}] GPU: {stats.gpu_allocated_gb:.2f}GB "
            f"({gpu_percentage:.1f}%), "
            f"Cached: {stats.gpu_cached_gb:.2f}GB, "
            f"System: {stats.system_memory_percent:.1f}%"
        )
        
        if logger:
            logger.info(message)
        else:
            print(message)
    
    def get_memory_optimization_config(self) -> Dict[str, any]:
        """
        Get recommended memory optimization configuration for transformers.
        
        Returns:
            Dictionary with optimization settings for model loading
        """
        config = {
            "dtype": "auto",  # Let HuggingFace pick optimal dtype per model
            "low_cpu_mem_usage": True,
        }
        
        # Only add CUDA-specific optimizations for CUDA devices
        if torch.cuda.is_available():
            config["device_map"] = "cuda"
            
        return config
    
    def estimate_model_memory(self, model_name: str) -> float:
        """
        Estimate memory requirements for different models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimates based on model parameters and fp16 precision
        memory_estimates = {
            "gpt2-large": 1.5,     # 774M parameters
            "llama-3.2-1b": 2.0,   # 1B parameters  
            "llama-3.2-3b": 6.0,   # 3B parameters (close to limit)
            "phi-3-mini": 2.5,     # 3.8B parameters but optimized
            "qwen2.5-3b": 6.0,     # 3B parameters
            "gemma-7b": 14.0,      # 7B parameters (exceeds RTX 3060)
        }
        
        # Default estimate if model not found
        return memory_estimates.get(model_name.lower(), 4.0)
    
    def can_load_model(self, model_name: str) -> bool:
        """
        Check if model can be loaded within memory constraints.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model can be loaded, False otherwise
        """
        estimated_memory = self.estimate_model_memory(model_name)
        current_usage = self.get_gpu_memory_usage()
        
        return (current_usage + estimated_memory) <= self.gpu_memory_limit


def get_flash_attention_config() -> Dict[str, any]:
    """
    Get Flash Attention 2 configuration for memory optimization.
    
    Returns:
        Configuration dictionary for Flash Attention 2
    """
    if not torch.cuda.is_available():
        return {}
        
    # Check if Flash Attention 2 is available
    try:
        import flash_attn
        return {
            "attn_implementation": "flash_attention_2",
            "use_cache": True,
        }
    except ImportError:
        print("Warning: Flash Attention 2 not available, using default attention")
        return {"attn_implementation": "eager"}


def optimize_for_rtx3060() -> None:
    """
    Apply RTX 3060 specific optimizations.
    
    Sets CUDA memory configuration for optimal performance.
    """
    if torch.cuda.is_available():
        # Enable memory fraction to prevent full GPU allocation
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Set CUDA allocator configuration for better memory management
        import os
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"