"""Model configurations for supported LLMs in MT-bench evaluation."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_path: str
    model_family: str
    prompt_template: str
    max_new_tokens: int
    temperature: float
    top_p: float
    estimated_memory_gb: float
    requires_system_prompt: bool = False
    chat_template_name: Optional[str] = None
    quantization_format: str = "FP16"  # FP32, FP16, BF16, FP8, INT8, INT4


# Supported models with their configurations
AVAILABLE_MODELS = {
    "gpt2": ModelConfig(
        model_path="gpt2",
        model_family="gpt2", 
        prompt_template="{instruction}",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=0.5,
        requires_system_prompt=False
    ),
    
    "gpt2-large": ModelConfig(
        model_path="gpt2-large",
        model_family="gpt2",
        prompt_template="{instruction}",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=1.5,  # Actual: 1.48GB from logs
        requires_system_prompt=False,
        quantization_format="FP16"
    ),
    
    "llama-3.2-1b": ModelConfig(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        model_family="llama",
        prompt_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=2.3,  # Actual: 2.31GB from logs
        requires_system_prompt=False,
        chat_template_name="llama",
        quantization_format="FP16"
    ),
    
    "llama-3.2-3b": ModelConfig(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        model_family="llama", 
        prompt_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=6.0,  # Actual: 5.98GB from logs (FP16, no quantization)
        requires_system_prompt=False,
        chat_template_name="llama",
        quantization_format="FP16"
    ),
    
    "phi-3-mini": ModelConfig(
        model_path="microsoft/Phi-3-mini-4k-instruct",
        model_family="phi",
        prompt_template="<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=7.1,  # Actual: 7.13GB from logs (FP16, no quantization)
        requires_system_prompt=False,
        chat_template_name="phi",
        quantization_format="FP16"
    ),
    
    "qwen2.5-3b": ModelConfig(
        model_path="Qwen/Qwen2.5-3B-Instruct",
        model_family="qwen",
        prompt_template="<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=5.9,  # Actual: 5.85GB from logs (FP16, no quantization)
        requires_system_prompt=False,
        chat_template_name="qwen",
        quantization_format="FP16"
    ),
    
    "gemma-2b": ModelConfig(
        model_path="google/gemma-2b-it",
        model_family="gemma",
        prompt_template="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=4.0,
        requires_system_prompt=False,
        chat_template_name="gemma"
    ),
    
    "gpt2-large-conversational": ModelConfig(
        model_path="Locutusque/gpt2-large-conversational-retrain",  # Problematic model - produces garbage output
        model_family="gpt2",
        prompt_template="<|USER|> {instruction} <|ASSISTANT|> ",
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.7,
        estimated_memory_gb=1.5,
        requires_system_prompt=False,
        chat_template_name="gpt2_conversational"
    ),
    
    "dialogpt-large": ModelConfig(
        model_path="microsoft/DialoGPT-large",
        model_family="gpt2",
        prompt_template="{instruction}",  # DialoGPT uses simple prompting
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        estimated_memory_gb=1.5,
        requires_system_prompt=False,
        chat_template_name="dialogpt"
    ),
    
    "gemma3-270m": ModelConfig(
        model_path="google/gemma-3-270m-it",  # Use correct gemma-3-270m-it path 
        model_family="gemma3",  # Use gemma3 family for proper CUDA handling
        prompt_template="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",  # Fallback template (chat template will override this)
        max_new_tokens=512,  # Use standard length for MT-bench
        temperature=0.0,  # Use greedy decoding to avoid CUDA errors 
        top_p=1.0,        # Use deterministic generation
        estimated_memory_gb=0.6,  # ~536MB + overhead
        requires_system_prompt=False,
        chat_template_name="gemma3",  # Use gemma3 chat template
        quantization_format="BF16"  # Native BF16 model
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig object
        
    Raises:
        ValueError: If model is not supported
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    return AVAILABLE_MODELS[model_name]


def get_available_models() -> Dict[str, ModelConfig]:
    """
    Get all available model configurations.
    
    Returns:
        Dictionary of model names to configurations
    """
    return AVAILABLE_MODELS.copy()


def get_models_within_memory_limit(memory_limit_gb: float) -> Dict[str, ModelConfig]:
    """
    Get models that can fit within the specified memory limit.
    
    Args:
        memory_limit_gb: Memory limit in GB
        
    Returns:
        Dictionary of model names to configurations that fit
    """
    return {
        name: config 
        for name, config in AVAILABLE_MODELS.items() 
        if config.estimated_memory_gb <= memory_limit_gb
    }


def get_generation_config(model_config: ModelConfig) -> Dict[str, Any]:
    """
    Get generation configuration for a model.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Dictionary with generation parameters
    """
    base_config = {
        "max_new_tokens": model_config.max_new_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "do_sample": True,
        "pad_token_id": None,  # Will be set based on tokenizer
        "eos_token_id": None,  # Will be set based on tokenizer
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    
    # Apply model-specific generation settings
    if model_config.chat_template_name == "gpt2_conversational":
        # Use optimized settings from Locutusque model
        base_config.update({
            "top_k": 23,
            "repetition_penalty": 1.176,
        })
    elif model_config.model_family == "phi":
        # Disable cache for phi models to fix DynamicCache compatibility
        base_config.update({
            "use_cache": False,
        })
    elif model_config.model_family == "gemma3":
        # Gemma-3 models need special handling due to new tokenizer and EOS behavior
        return {
            "max_new_tokens": model_config.max_new_tokens,
            "min_new_tokens": 50,  # Force at least 50 tokens to ensure substantial output
            "do_sample": False,  # Keep greedy decoding to avoid CUDA errors
            "pad_token_id": None,  # Will be set based on tokenizer
            "eos_token_id": None,  # Will be set based on tokenizer
            "repetition_penalty": 1.05,  # Minimal penalty to avoid special token loops
            "no_repeat_ngram_size": 0,  # Disable n-gram blocking completely
            # No sampling parameters to avoid CUDA issues
        }
    elif model_config.model_family == "gemma":
        # Original Gemma models (2B, etc.)
        if "gemma-3-270m" in model_config.model_path:
            # This shouldn't happen anymore since we changed to gemma3 family
            return {
                "max_new_tokens": model_config.max_new_tokens,
                "min_new_tokens": 20,  # Force at least 20 tokens to prevent empty responses
                "do_sample": False,  # Disable sampling to avoid CUDA assertion errors
                "pad_token_id": None,  # Will be set based on tokenizer
                "eos_token_id": None,  # Will be set based on tokenizer
                "repetition_penalty": 1.05,  # Minimal repetition penalty 
                "no_repeat_ngram_size": 0,  # Disable to prevent blocking
                # No temperature, top_p, or top_k since we're using greedy decoding
            }
        else:
            # More conservative sampling for other Gemma models
            base_config.update({
                "temperature": max(0.1, model_config.temperature * 0.5),  # Reduce temperature
                "top_p": 0.8,  # More conservative top_p
                "top_k": 50,   # Add top_k filtering
                "repetition_penalty": 1.05,  # Reduce repetition penalty
                "do_sample": model_config.temperature > 0.0,  # Only sample if temperature > 0
            })
    
    return base_config


def format_prompt_for_model(instruction: str, model_config: ModelConfig, 
                          conversation_history: Optional[str] = None,
                          tokenizer=None) -> str:
    """
    Format instruction using model-specific prompt template or chat template.
    
    Args:
        instruction: The instruction/question to format
        model_config: Model configuration containing prompt template
        conversation_history: Optional conversation history for multi-turn
        tokenizer: Optional tokenizer with apply_chat_template support
        
    Returns:
        Formatted prompt string
    """
    # Try to use tokenizer's built-in chat template first (for modern models)
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            # Build messages list
            messages = []
            
            # Add conversation history as previous turns if available
            if conversation_history:
                # Parse conversation history into alternating user/assistant messages
                # This is a simple parser - could be enhanced for complex histories
                history_lines = conversation_history.strip().split('\n')
                current_user_msg = None
                for line in history_lines:
                    line = line.strip()
                    if line.startswith('User: '):
                        current_user_msg = line[6:]  # Remove 'User: ' prefix
                    elif line.startswith('Assistant: ') and current_user_msg:
                        assistant_msg = line[11:]  # Remove 'Assistant: ' prefix
                        messages.append({"role": "user", "content": current_user_msg})
                        messages.append({"role": "assistant", "content": assistant_msg})
                        current_user_msg = None
            
            # Add current instruction as user message
            messages.append({"role": "user", "content": instruction})
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False  # Return string, not tokens
            )
            
            return formatted_prompt
            
        except Exception as e:
            # Log the attempt but fall back gracefully
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Chat template failed for {model_config.model_path}: {e}. Using fallback.")
    
    # Fallback to existing manual template logic
    if conversation_history:
        # For multi-turn conversations, include history
        full_instruction = f"{conversation_history}\n\nUser: {instruction}\nAssistant:"
    else:
        full_instruction = instruction
    
    return model_config.prompt_template.format(instruction=full_instruction)


def get_system_message(model_config: ModelConfig) -> Optional[str]:
    """
    Get system message for models that support it.
    
    Args:
        model_config: Model configuration
        
    Returns:
        System message string or None
    """
    if not model_config.requires_system_prompt:
        return None
    
    return (
        "You are a helpful AI assistant. Please provide accurate, "
        "informative, and engaging responses to the user's questions."
    )


# Model family specific optimization settings
OPTIMIZATION_CONFIGS = {
    "gpt2": {
        "use_cache": True,
        "gradient_checkpointing": False,  # GPT-2 is small enough
    },
    "llama": {
        "use_cache": True,
        "gradient_checkpointing": True,
        "rope_scaling": None,
    },
    "phi": {
        "use_cache": False,  # Disable cache to fix DynamicCache compatibility issue
        "gradient_checkpointing": True,
        "trust_remote_code": True,
    },
    "qwen": {
        "use_cache": True,
        "gradient_checkpointing": True,
        "trust_remote_code": True,
    },
    "gemma": {
        "use_cache": True,
        "gradient_checkpointing": True,
    },
    "gemma3": {
        "use_cache": True,
        "gradient_checkpointing": True,
        "trust_remote_code": True,  # May be needed for newer Gemma 3 models
    }
}


def get_optimization_config(model_config: ModelConfig) -> Dict[str, Any]:
    """
    Get optimization configuration for a model family.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Dictionary with optimization settings
    """
    return OPTIMIZATION_CONFIGS.get(model_config.model_family, {
        "use_cache": True,
        "gradient_checkpointing": True,
    })