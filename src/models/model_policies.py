"""Model family and generation behavior policies."""

from typing import Any, Dict

from .model_types import FamilyBehavior, GenerationBehavior, ModelConfig


FAMILY_BEHAVIORS: Dict[str, FamilyBehavior] = {
    "gemma3": FamilyBehavior(
        use_chat_template_generation=True,
        force_greedy_generation=True,
    )
}

GENERATION_BEHAVIORS: Dict[str, GenerationBehavior] = {
    "gpt2_conversational": GenerationBehavior(
        use_legacy_encode_generation=True,
        decode_with_assistant_marker=True,
        assistant_marker="<|ASSISTANT|>",
    ),
    "dialogpt": GenerationBehavior(
        use_legacy_encode_generation=True,
        decode_with_assistant_marker=False,
    ),
}

OPTIMIZATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt2": {
        "use_cache": True,
        "gradient_checkpointing": False,
    },
    "llama": {
        "use_cache": True,
        "gradient_checkpointing": True,
        "rope_scaling": None,
    },
    "phi": {
        "use_cache": False,
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
        "trust_remote_code": True,
    },
}


def get_family_behavior(model_config: ModelConfig) -> FamilyBehavior:
    """Get behavior flags for a model family."""
    return FAMILY_BEHAVIORS.get(model_config.model_family, FamilyBehavior())


def get_generation_behavior(model_config: ModelConfig) -> GenerationBehavior:
    """Get generation-path behavior for the model's chat template style."""
    key = model_config.chat_template_name
    if key is None:
        return GenerationBehavior()
    return GENERATION_BEHAVIORS.get(key, GenerationBehavior())


def get_optimization_config(model_config: ModelConfig) -> Dict[str, Any]:
    """Get optimization configuration for a model family."""
    return OPTIMIZATION_CONFIGS.get(
        model_config.model_family,
        {"use_cache": True, "gradient_checkpointing": True},
    )
