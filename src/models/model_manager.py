"""Memory-optimized model management for RTX 3060 GPU constraints."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Optional, Dict, Any, Tuple
import logging

from ..runtime.model_runtime import ModelRuntime
from .model_configs import get_model_config, get_generation_config, get_family_behavior


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
        
        # Get model configuration for generation config
        config = get_model_config(self.current_model_name)
        gen_config = get_generation_config(config)
        
        # Override with any provided kwargs
        gen_config.update(generation_kwargs)
        
        # Set token IDs based on tokenizer (don't override if tokenizer has them)
        if gen_config["pad_token_id"] is None:
            gen_config["pad_token_id"] = self.current_tokenizer.pad_token_id if self.current_tokenizer.pad_token_id is not None else self.current_tokenizer.eos_token_id
        if gen_config["eos_token_id"] is None:
            gen_config["eos_token_id"] = self.current_tokenizer.eos_token_id
            
        # Ensure token IDs are within valid range to prevent CUDA assertion errors
        vocab_size = len(self.current_tokenizer)
        for key in ["pad_token_id", "eos_token_id", "bos_token_id"]:
            if key in gen_config and gen_config[key] is not None:
                if gen_config[key] >= vocab_size:
                    logger.warning(f"{key} ({gen_config[key]}) >= vocab_size ({vocab_size}), using eos_token_id")
                    gen_config[key] = self.current_tokenizer.eos_token_id
        
        self.memory_monitor.log_memory_usage("Before generation", logger)
        
        try:
            # Get current model config for prompt formatting
            current_config = get_model_config(self.current_model_name)
            
            family_behavior = get_family_behavior(current_config)

            # Format prompt based on family behavior
            if family_behavior.use_chat_template_generation:
                # Use the exact same approach as the working minimal test
                messages = [{"role": "user", "content": prompt}]
                try:
                    inputs = self.current_tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                except Exception as e:
                    logger.warning(f"Chat template failed for gemma3 ({e}), using fallback")
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    inputs = self.current_tokenizer(formatted_prompt, return_tensors="pt")
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Generate using EXACTLY the same call as working minimal test
                with torch.no_grad():
                    outputs = self.current_model.generate(
                        **inputs,
                        max_new_tokens=256
                    )
                
                # Decode response exactly like working minimal test  
                input_length = inputs["input_ids"].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                # Filter artifacts but keep original logic simple
                unwanted_patterns = ["<pad>", "<mask>"]
                for pattern in unwanted_patterns:
                    response = response.replace(pattern, "")
                response = response.strip()
                
            elif self.current_model_name in ["gpt2-large-conversational", "dialogpt-large"]:
                # Use the exact tokenization approach from the example
                input_ids = self.current_tokenizer.encode(
                    prompt, 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                
                # Move to device
                model_device = next(self.current_model.parameters()).device
                input_ids = input_ids.to(model_device)
                attention_mask = torch.ones_like(input_ids).to(model_device)
                
                # Use max_length instead of max_new_tokens for this model
                gen_config_copy = gen_config.copy()
                if "max_new_tokens" in gen_config_copy:
                    max_length = len(input_ids[0]) + gen_config_copy.pop("max_new_tokens")
                    gen_config_copy["max_length"] = min(max_length, 1024)  # Cap at 1024 as in example
                
                # Generate response with exact parameters from example
                with torch.no_grad():
                    outputs = self.current_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        **gen_config_copy
                    )
            else:
                # Standard tokenization for other models
                inputs = self.current_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=2048
                )
                
                # Validate input tokens are within vocabulary range
                vocab_size = len(self.current_tokenizer)
                input_ids = inputs["input_ids"]
                max_token_id = input_ids.max().item()
                if max_token_id >= vocab_size:
                    logger.error(f"Input contains token IDs >= vocab_size: max_id={max_token_id}, vocab_size={vocab_size}")
                    # Clamp token IDs to valid range
                    inputs["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)
                    logger.warning("Clamped input token IDs to valid range")
                
                # Move inputs to the same device as the model
                model_device = next(self.current_model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.current_model.generate(
                        **inputs,
                        **gen_config
                    )
            
            # Decode response (remove input tokens) - only for chat-template family path
            if family_behavior.use_chat_template_generation:
                # Response already handled above for gemma3
                pass
            elif self.current_model_name in ["gpt2-large-conversational", "dialogpt-large"]:
                input_length = input_ids.shape[1]
                
                if self.current_model_name == "gpt2-large-conversational":
                    # For gpt2-conversational model, decode the full output and extract the response part
                    full_output = self.current_tokenizer.decode(outputs[0], skip_special_tokens=False)
                    # Extract only the assistant's response after <|ASSISTANT|>
                    if "<|ASSISTANT|>" in full_output:
                        response = full_output.split("<|ASSISTANT|>")[-1].strip()
                    else:
                        # Fallback: remove input tokens
                        response_tokens = outputs[0][input_length:]
                        response = self.current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                else:  # dialogpt-large
                    # For DialoGPT, just remove input tokens
                    response_tokens = outputs[0][input_length:]
                    response = self.current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            else:
                # Standard decoding for other models
                input_length = inputs["input_ids"].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            self.memory_monitor.log_memory_usage("After generation", logger)
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during generation: {str(e)}")
            self.memory_monitor.cleanup_gpu_memory()
            raise RuntimeError(f"GPU out of memory during generation: {str(e)}")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
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
