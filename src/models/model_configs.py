"""Compatibility facade for model configuration modules."""

from .model_policies import (
    FAMILY_BEHAVIORS,
    GENERATION_BEHAVIORS,
    OPTIMIZATION_CONFIGS,
    get_family_behavior,
    get_generation_behavior,
    get_optimization_config,
)
from .model_queries import (
    format_prompt_for_model,
    get_available_families,
    get_available_models,
    get_generation_config,
    get_model_config,
    get_models_by_family,
    get_models_within_memory_limit,
    get_system_message,
    normalize_model_name,
)
from .model_registry import AVAILABLE_MODELS, MODEL_ALIASES
from .model_types import FamilyBehavior, GenerationBehavior, ModelConfig

__all__ = [
    "AVAILABLE_MODELS",
    "MODEL_ALIASES",
    "FAMILY_BEHAVIORS",
    "GENERATION_BEHAVIORS",
    "OPTIMIZATION_CONFIGS",
    "FamilyBehavior",
    "GenerationBehavior",
    "ModelConfig",
    "format_prompt_for_model",
    "get_available_families",
    "get_available_models",
    "get_family_behavior",
    "get_generation_behavior",
    "get_generation_config",
    "get_model_config",
    "get_models_by_family",
    "get_models_within_memory_limit",
    "get_optimization_config",
    "get_system_message",
    "normalize_model_name",
]
