"""Core model configuration datatypes."""

from dataclasses import dataclass
from typing import Optional


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
    load_in_4bit: bool = False


@dataclass
class FamilyBehavior:
    """Behavior flags shared by all models in a family."""

    use_chat_template_generation: bool = False
    force_greedy_generation: bool = False


@dataclass
class GenerationBehavior:
    """Generation-path behavior overrides for specific chat template styles."""

    use_legacy_encode_generation: bool = False
    decode_with_assistant_marker: bool = False
    assistant_marker: str = "<|ASSISTANT|>"
