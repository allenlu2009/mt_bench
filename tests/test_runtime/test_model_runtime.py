"""Unit tests for ModelRuntime shared lifecycle behavior."""

from unittest.mock import Mock, patch

import pytest

from src.runtime.model_runtime import ModelRuntime


class _FakeParam:
    is_meta = False


class _FakeModel:
    def __init__(self):
        self._eval_called = False

    def parameters(self):
        return iter([_FakeParam()])

    def to(self, _device):
        return self

    def to_empty(self, device=None):
        return self

    def eval(self):
        self._eval_called = True
        return self


@pytest.fixture
def runtime_cpu():
    with patch("torch.cuda.is_available", return_value=False):
        return ModelRuntime(device="cuda", memory_limit_gb=8.0)


def test_runtime_initializes_cpu_when_cuda_unavailable(runtime_cpu):
    assert runtime_cpu.device == "cpu"
    assert runtime_cpu.current_model is None
    assert runtime_cpu.current_tokenizer is None


def test_load_model_sets_current_state(runtime_cpu):
    fake_tokenizer = Mock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.get_vocab.return_value = {"a": 1}
    fake_tokenizer.__len__ = Mock(return_value=100)
    fake_model = _FakeModel()

    with patch("src.runtime.model_runtime.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), patch(
        "src.runtime.model_runtime.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ):
        model, tokenizer = runtime_cpu.load_model("gpt2")

    assert model is runtime_cpu.current_model
    assert tokenizer is runtime_cpu.current_tokenizer
    assert runtime_cpu.current_model_name == "gpt2"
    assert fake_model._eval_called is True


def test_unload_model_clears_state(runtime_cpu):
    fake_tokenizer = Mock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.get_vocab.return_value = {"a": 1}
    fake_tokenizer.__len__ = Mock(return_value=100)
    fake_model = _FakeModel()

    with patch("src.runtime.model_runtime.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), patch(
        "src.runtime.model_runtime.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ):
        runtime_cpu.load_model("gpt2")

    runtime_cpu.unload_model()
    assert runtime_cpu.current_model is None
    assert runtime_cpu.current_tokenizer is None
    assert runtime_cpu.current_model_name is None


def test_load_model_failure_cleans_state(runtime_cpu):
    fake_tokenizer = Mock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.get_vocab.return_value = {"a": 1}
    fake_tokenizer.__len__ = Mock(return_value=100)

    with patch("src.runtime.model_runtime.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), patch(
        "src.runtime.model_runtime.AutoModelForCausalLM.from_pretrained", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="boom"):
            runtime_cpu.load_model("gpt2")

    assert runtime_cpu.current_model is None
    assert runtime_cpu.current_tokenizer is None
    assert runtime_cpu.current_model_name is None

