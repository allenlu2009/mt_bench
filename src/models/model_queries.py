"""Query helpers for model configuration access."""

from typing import Any, Dict, List, Optional

from .model_registry import AVAILABLE_MODELS
from .model_types import ModelConfig


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model_name]


def get_available_models() -> Dict[str, ModelConfig]:
    """Get all available model configurations."""
    return AVAILABLE_MODELS.copy()


def get_models_within_memory_limit(memory_limit_gb: float) -> Dict[str, ModelConfig]:
    """Get models that can fit within the specified memory limit."""
    return {
        name: config
        for name, config in AVAILABLE_MODELS.items()
        if config.estimated_memory_gb <= memory_limit_gb
    }


def get_models_by_family(family: str) -> List[str]:
    """Get all model names belonging to the specified family."""
    family_models = []
    for name, config in AVAILABLE_MODELS.items():
        if config.model_family == family:
            family_models.append(name)
    return family_models


def get_available_families() -> List[str]:
    """Get all available model families."""
    families = set()
    for config in AVAILABLE_MODELS.values():
        families.add(config.model_family)
    return sorted(list(families))


def get_generation_config(model_config: ModelConfig) -> Dict[str, Any]:
    """Get generation configuration for a model."""
    base_config = {
        "max_new_tokens": model_config.max_new_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }

    if model_config.chat_template_name == "gpt2_conversational":
        base_config.update({"top_k": 23, "repetition_penalty": 1.176})
    elif model_config.model_family == "phi":
        base_config.update({"use_cache": False})
    elif model_config.model_family == "gemma3":
        return {
            "max_new_tokens": model_config.max_new_tokens,
            "do_sample": False,
            "pad_token_id": None,
            "eos_token_id": None,
        }
    elif model_config.model_family == "gemma":
        base_config.update(
            {
                "temperature": max(0.1, model_config.temperature * 0.5),
                "top_p": 0.8,
                "top_k": 50,
                "repetition_penalty": 1.05,
                "do_sample": model_config.temperature > 0.0,
            }
        )

    return base_config


def format_prompt_for_model(
    instruction: str,
    model_config: ModelConfig,
    conversation_history: Optional[str] = None,
    tokenizer: Any = None,
) -> str:
    """Format instruction using model-specific prompt or tokenizer chat template."""
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if conversation_history:
                history_lines = conversation_history.strip().split("\n")
                current_user_msg = None
                for line in history_lines:
                    line = line.strip()
                    if line.startswith("User: "):
                        current_user_msg = line[6:]
                    elif line.startswith("Assistant: ") and current_user_msg:
                        assistant_msg = line[11:]
                        messages.append({"role": "user", "content": current_user_msg})
                        messages.append({"role": "assistant", "content": assistant_msg})
                        current_user_msg = None

            messages.append({"role": "user", "content": instruction})
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                "Chat template failed for %s. Falling back to prompt template.",
                model_config.model_path,
            )

    if conversation_history:
        full_instruction = f"{conversation_history}\n\nUser: {instruction}\nAssistant:"
    else:
        full_instruction = instruction

    return model_config.prompt_template.format(instruction=full_instruction)


def get_system_message(model_config: ModelConfig) -> Optional[str]:
    """Get system message for models that require it."""
    if not model_config.requires_system_prompt:
        return None

    return (
        "You are a helpful AI assistant. Please provide accurate, "
        "informative, and engaging responses to the user's questions."
    )
