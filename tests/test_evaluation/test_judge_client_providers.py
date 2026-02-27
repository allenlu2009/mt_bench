"""Tests for judge-client provider configuration."""

import pytest

from src.evaluation.judge_client import JudgeClient


def test_judge_client_deepseek_uses_provider_mapping() -> None:
    client = JudgeClient(api_key="x", model="deepseek-chat")
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_model_name == "deepseek-chat"


def test_judge_client_minimax_uses_provider_mapping() -> None:
    client = JudgeClient(api_key="x", model="minimax-mini")
    assert client.base_url == "https://api.minimax.chat/v1"
    assert client.api_model_name == "MiniMax-Text-01"


def test_judge_client_openai_defaults_no_base_url() -> None:
    client = JudgeClient(api_key="x", model="gpt-5-nano")
    assert client.base_url == ""
    assert client.api_model_name == "gpt-5-nano"


def test_judge_client_requires_provider_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        JudgeClient(api_key=None, model="deepseek-chat")

