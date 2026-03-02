"""Tests for PerplexityEvaluator configuration behavior."""

from src.evaluation.perplexity_evaluator import PerplexityEvaluator


def test_perplexity_evaluator_defaults_to_safe_block_size_clamp() -> None:
    evaluator = PerplexityEvaluator(model_names=["qwen2.5-3b"])
    assert evaluator.allow_block_size_over_max_length is False


def test_perplexity_evaluator_supports_max_length_override() -> None:
    evaluator = PerplexityEvaluator(
        model_names=["qwen2.5-3b"],
        allow_block_size_over_max_length=True,
    )
    assert evaluator.allow_block_size_over_max_length is True
