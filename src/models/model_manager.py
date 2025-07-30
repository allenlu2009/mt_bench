"""Memory-optimized model management for RTX 3060 GPU constraints."""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from typing import Optional, Dict, Any, Tuple
import logging
import warnings

from ..utils.memory_utils import MemoryMonitor, get_flash_attention_config, optimize_for_rtx3060
from .model_configs import ModelConfig, get_model_config, get_generation_config, get_optimization_config


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading and inference with memory optimization for RTX 3060.
    
    Key features:
    - Flash Attention 2 integration for 30% memory reduction
    - Automatic memory cleanup between model loads
    - fp16 precision for memory efficiency
    - Gradient checkpointing for reduced memory usage
    """
    
    def __init__(self, device: str = "cuda", memory_limit_gb: float = 6.0):
        """
        Initialize model manager.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            memory_limit_gb: GPU memory limit in GB (default: 6.0 for RTX 3060)
        """
        # Force CPU on macOS to avoid MPS memory issues
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        self.memory_monitor = MemoryMonitor(gpu_memory_limit_gb=memory_limit_gb)
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        # Apply RTX 3060 optimizations
        if self.device == "cuda":
            optimize_for_rtx3060()
            
        logger.info(f"ModelManager initialized on {self.device}")
        
    def _cleanup_current_model(self) -> None:
        """
        Cleanup currently loaded model to free memory.
        
        Critical for RTX 3060 memory management.
        """
        if self.current_model is not None:
            logger.info(f"Cleaning up model: {self.current_model_name}")
            
            # Move model to CPU to free GPU memory
            if hasattr(self.current_model, 'to'):
                self.current_model.to('cpu')
            
            # Delete references
            del self.current_model
            del self.current_tokenizer
            
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Force GPU memory cleanup
            self.memory_monitor.cleanup_gpu_memory()
            
    def _get_model_loading_config(self, model_config: ModelConfig) -> Dict[str, Any]:
        """
        Get optimized loading configuration for the model.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Dictionary with loading parameters
        """
        # Base memory optimization config
        config = self.memory_monitor.get_memory_optimization_config()
        
        # Add Flash Attention config
        flash_config = get_flash_attention_config()
        config.update(flash_config)
        
        # Add model-specific optimizations
        opt_config = get_optimization_config(model_config)
        config.update(opt_config)
        
        # Add 4-bit quantization for larger models if needed
        if model_config.estimated_memory_gb > 5.0:
            logger.info("Using 4-bit quantization for large model")
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        return config
        
    def load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model with memory optimization.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model is not supported
            RuntimeError: If model exceeds memory limits
        """
        # Check if model is already loaded
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return self.current_model, self.current_tokenizer
            
        # Get model configuration
        model_config = get_model_config(model_name)
        
        # Check memory constraints
        if not self.memory_monitor.can_load_model(model_name):
            raise RuntimeError(
                f"Model {model_name} estimated memory "
                f"({model_config.estimated_memory_gb:.1f}GB) exceeds limit "
                f"({self.memory_monitor.gpu_memory_limit:.1f}GB)"
            )
        
        # Cleanup current model
        self._cleanup_current_model()
        
        logger.info(f"Loading model: {model_name} ({model_config.model_path})")
        self.memory_monitor.log_memory_usage("Before model loading", logger)
        
        try:
            # Get optimized loading configuration
            loading_config = self._get_model_loading_config(model_config)
            
            # Load tokenizer first (minimal memory impact)
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_path,
                trust_remote_code=loading_config.get("trust_remote_code", False),
                padding_side="left"  # For batch inference
            )
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded, vocab size: {len(tokenizer)}")
            
            # Load model with optimizations
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                model_kwargs = {
                    "torch_dtype": loading_config["torch_dtype"],
                    "attn_implementation": loading_config.get("attn_implementation", "eager"),
                    "low_cpu_mem_usage": loading_config["low_cpu_mem_usage"],
                    "trust_remote_code": loading_config.get("trust_remote_code", False),
                    "use_cache": loading_config.get("use_cache", True)
                }
                
                # Only add device_map and quantization_config if present
                if "device_map" in loading_config:
                    model_kwargs["device_map"] = loading_config["device_map"]
                if "quantization_config" in loading_config:
                    model_kwargs["quantization_config"] = loading_config["quantization_config"]
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_path,
                    **model_kwargs
                )
            
            # Enable gradient checkpointing if specified
            if loading_config.get("gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Move model to the correct device and set to evaluation mode
            if "device_map" not in loading_config:
                model = model.to(self.device)
            model.eval()
            
            # Store references
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name
            
            # Check memory usage
            self.memory_monitor.check_memory_limit(f"loading {model_name}")
            self.memory_monitor.log_memory_usage("After model loading", logger)
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Cleanup on failure
            self._cleanup_current_model()
            raise
    
    def generate_response(self, prompt: str, model_name: Optional[str] = None,
                         **generation_kwargs) -> str:
        """
        Generate response using the current or specified model.
        
        Args:
            prompt: Input prompt
            model_name: Model to use (loads if not current)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Load model if needed
        if model_name and model_name != self.current_model_name:
            self.load_model(model_name)
        elif self.current_model is None:
            raise RuntimeError("No model loaded. Please specify model_name.")
        
        # Get model configuration for generation config
        config = get_model_config(self.current_model_name)
        gen_config = get_generation_config(config)
        
        # Override with any provided kwargs
        gen_config.update(generation_kwargs)
        
        # Set pad_token_id if not provided
        if gen_config["pad_token_id"] is None:
            gen_config["pad_token_id"] = self.current_tokenizer.eos_token_id
        if gen_config["eos_token_id"] is None:
            gen_config["eos_token_id"] = self.current_tokenizer.eos_token_id
        
        self.memory_monitor.log_memory_usage("Before generation", logger)
        
        try:
            # Tokenize input
            inputs = self.current_tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            
            # Move inputs to the same device as the model
            model_device = next(self.current_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    **gen_config
                )
            
            # Decode response (remove input tokens)
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.current_tokenizer.decode(
                response_tokens, 
                skip_special_tokens=True
            ).strip()
            
            self.memory_monitor.log_memory_usage("After generation", logger)
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during generation: {str(e)}")
            self.memory_monitor.cleanup_gpu_memory()
            raise RuntimeError(f"GPU out of memory during generation: {str(e)}")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model is None:
            return {"status": "no_model_loaded"}
        
        config = get_model_config(self.current_model_name)
        memory_stats = self.memory_monitor.get_comprehensive_stats()
        
        return {
            "model_name": self.current_model_name,
            "model_path": config.model_path,
            "model_family": config.model_family,
            "estimated_memory_gb": config.estimated_memory_gb,
            "actual_memory_gb": memory_stats.gpu_allocated_gb,
            "device": self.device,
            "vocab_size": len(self.current_tokenizer) if self.current_tokenizer else None,
            "memory_stats": memory_stats
        }
    
    def cleanup(self) -> None:
        """
        Cleanup all resources and free memory.
        """
        logger.info("Cleaning up ModelManager")
        self._cleanup_current_model()
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()