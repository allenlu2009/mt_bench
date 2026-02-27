"""Tests for CLI provider routing and API key semantics."""

import argparse

import pytest

from src.cli_parts.common import require_mtbench_api_key
from src.cli_parts.parser import create_parser, normalize_cli_argv


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="cli-test")


def test_parser_supports_new_judge_models() -> None:
    parser = create_parser()
    args = parser.parse_args(
        normalize_cli_argv(
            ["mt-bench", "--models", "qwen2.5-3b", "--judge-model", "deepseek-chat"]
        )
    )
    assert args.judge_model == "deepseek-chat"


def test_openai_key_resolution_prefers_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    args = argparse.Namespace(
        project="mt-bench",
        judge_model="gpt-5-nano",
        openai_api_key="flag-openai",
    )
    assert require_mtbench_api_key(args, _parser()) == "flag-openai"


def test_deepseek_requires_deepseek_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    args = argparse.Namespace(
        project="mt-bench",
        judge_model="deepseek-chat",
        openai_api_key=None,
    )
    with pytest.raises(SystemExit):
        require_mtbench_api_key(args, _parser())


def test_deepseek_reads_deepseek_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    args = argparse.Namespace(
        project="mt-bench",
        judge_model="deepseek-reasoner",
        openai_api_key=None,
    )
    assert require_mtbench_api_key(args, _parser()) == "deepseek-key"


def test_minimax_reads_minimax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
    args = argparse.Namespace(
        project="mt-bench",
        judge_model="minimax-mini",
        openai_api_key=None,
    )
    assert require_mtbench_api_key(args, _parser()) == "minimax-key"


def test_non_mtbench_project_skips_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = argparse.Namespace(
        project="perplexity",
        judge_model="gpt-5-nano",
        openai_api_key=None,
    )
    assert require_mtbench_api_key(args, _parser()) == ""
