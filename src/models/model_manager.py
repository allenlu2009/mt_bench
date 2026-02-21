"""Memory-optimized model management for RTX 3060 GPU constraints."""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Optional, Dict, Any, Tuple
import logging

from ..evaluation.generation_runtime import generate_response_with_loaded_model
from ..runtime.model_runtime import ModelRuntime


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading and inference with memory optimization for RTX 3060.

    Key features:
    - Flash Attention 2 integration for 30% memory reduction
    - Automatic memory cleanup between model loads
    - Auto dtype selection for optimal precision per model
    """
    
    def __init__(self, device: str = "cuda", memory_limit_gb: float = 8.0):
        """
        Initialize model manager.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            memory_limit_gb: GPU memory limit in GB (default: 8.0 for RTX 3060)
        """
        self.runtime = ModelRuntime(device=device, memory_limit_gb=memory_limit_gb)
        self.device = self.runtime.device
        self.memory_monitor = self.runtime.memory_monitor
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

        logger.info("ModelManager initialized on %s", self.device)

    def _sync_runtime_state(self) -> None:
        self.current_model = self.runtime.current_model
        self.current_tokenizer = self.runtime.current_tokenizer
        self.current_model_name = self.runtime.current_model_name
        
    def _cleanup_current_model(self) -> None:
        """
        Cleanup currently loaded model to free memory.

        Critical for RTX 3060 memory management.
        Avoids model.to('cpu') which can crash if CUDA is in an error state.
        """
        self.runtime.unload_model()
        self._sync_runtime_state()
        
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
        model, tokenizer = self.runtime.load_model(model_name)
        self._sync_runtime_state()
        return model, tokenizer
    
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

        return generate_response_with_loaded_model(
            prompt=prompt,
            current_model_name=self.current_model_name,
            current_model=self.current_model,
            current_tokenizer=self.current_tokenizer,
            memory_monitor=self.memory_monitor,
            generation_kwargs=generation_kwargs,
        )
    
    def get_current_tokenizer(self):
        """
        Get the currently loaded tokenizer.
        
        Returns:
            Current tokenizer instance or None if no model is loaded
        """
        return self.current_tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model is None:
            return {"status": "no_model_loaded"}
        
        return self.runtime.get_model_info()
    
    def cleanup(self) -> None:
        """
        Cleanup all resources and free memory.
        """
        logger.info("Cleaning up ModelManager")
        self.runtime.unload_model()
        self._sync_runtime_state()
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
