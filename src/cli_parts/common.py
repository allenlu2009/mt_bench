"""Shared CLI helpers: logging, model selection, and model listing."""

import argparse
import logging
import os
import sys
from typing import List

from ..models.model_configs import (
    get_available_families,
    get_available_models,
    get_models_by_family,
    get_models_within_memory_limit,
    normalize_model_name,
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("mtbench_evaluation.log")],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def validate_models(model_names: List[str], memory_limit_gb: float) -> List[str]:
    """Validate and filter model names."""
    available_models = get_available_models()
    memory_compatible_models = get_models_within_memory_limit(memory_limit_gb)

    valid_models: List[str] = []
    for requested_name in model_names:
        model_name = normalize_model_name(requested_name)
        if model_name not in available_models:
            print(f"Warning: Model '{model_name}' not available. Available models:")
            for name in available_models.keys():
                print(f"  - {name}")
            continue

        config = available_models[model_name]
        if model_name not in memory_compatible_models:
            print(
                f"Note: Model '{model_name}' requires {config.estimated_memory_gb:.1f}GB "
                f"(memory limit: {memory_limit_gb:.1f}GB). Ensure your GPU has sufficient VRAM."
            )
        if model_name not in valid_models:
            valid_models.append(model_name)

    if not valid_models:
        raise ValueError("No valid models found after filtering")
    return valid_models


def print_available_models() -> None:
    """Print all available models with their specifications."""
    models = get_available_models()
    print("Available Models:")
    print("-" * 100)
    print(f"{'Model Name':<20} {'Family':<10} {'Memory (GB)':<12} {'Format':<8} {'Model Path'}")
    print("-" * 100)

    for name, config in models.items():
        print(
            f"{name:<20} {config.model_family:<10} {config.estimated_memory_gb:<12.1f} "
            f"{config.quantization_format:<8} {config.model_path}"
        )

    print(f"\nTotal models: {len(models)}")
    print("\nFormat Legend:")
    print("  FP32 = 32-bit floating point")
    print("  FP16 = 16-bit floating point")
    print("  INT4 = 4-bit integer quantization")


def resolve_model_names(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    """Resolve model names from --models or --family."""
    if args.family:
        available_families = get_available_families()
        if args.family not in available_families:
            parser.error(
                f"Family '{args.family}' not found. Available families: {', '.join(available_families)}"
            )
        family_models = get_models_by_family(args.family)
        if not family_models:
            parser.error(f"No models found in family '{args.family}'")
        print(f"Evaluating all models in '{args.family}' family: {', '.join(family_models)}")
        return family_models

    if args.models:
        return args.models

    parser.error("Either --models or --family is required (use --list-models to see available models)")
    return []


def require_mtbench_api_key(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Validate and return API key for MT-bench project."""
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if args.project == "mt-bench" and not api_key:
        parser.error("OpenAI API key required for mt-bench. Set OPENAI_API_KEY env var or use --openai-api-key")
    return api_key
