"""Tests for model configuration management."""

import pytest
from src.models.model_configs import (
    ModelConfig, 
    get_model_config, 
    get_available_models,
    get_models_within_memory_limit,
    get_generation_config,
    format_prompt_for_model,
    get_system_message,
    get_optimization_config,
    AVAILABLE_MODELS
)


class TestModelConfig:
    """Test cases for ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test ModelConfig instantiation."""
        config = ModelConfig(
            model_path="test/model",
            model_family="test",
            prompt_template="{instruction}",
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            estimated_memory_gb=2.0
        )
        
        assert config.model_path == "test/model"
        assert config.model_family == "test"
        assert config.max_new_tokens == 256
        assert config.temperature == 0.8
        assert config.estimated_memory_gb == 2.0
        assert config.requires_system_prompt is False  # default value


class TestModelConfigRetrieval:
    """Test cases for model configuration retrieval functions."""
    
    def test_get_model_config_valid(self):
        """Test retrieving configuration for valid model."""
        config = get_model_config("gpt2-large")
        
        assert isinstance(config, ModelConfig)
        assert config.model_path == "gpt2-large"
        assert config.model_family == "gpt2"
        assert config.estimated_memory_gb == 1.5
    
    def test_get_model_config_invalid(self):
        """Test retrieving configuration for invalid model."""
        with pytest.raises(ValueError, match="Model nonexistent-model not supported"):
            get_model_config("nonexistent-model")
    
    def test_get_available_models(self):
        """Test getting all available models."""
        models = get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "gpt2-large" in models
        assert "gpt2-large-conversational" in models
        assert "llama-3.2-1b" in models
        
        # Test that it returns a copy
        models["test"] = "test"
        original_models = get_available_models()
        assert "test" not in original_models
    
    def test_get_models_within_memory_limit(self):
        """Test filtering models by memory limit."""
        # Test with 2.5GB limit - should include small and medium models
        small_models = get_models_within_memory_limit(2.5)
        assert "gpt2-large" in small_models  # 1.5GB
        assert "gpt2-large-conversational" in small_models  # 1.5GB
        assert "llama-3.2-1b" in small_models  # 2.3GB
        assert "llama-3.2-3b" in small_models  # 2.1GB
        assert "qwen2.5-3b" in small_models  # 1.9GB
        assert "phi-3-mini" not in small_models  # 7.1GB
        
        # Test with large limit - should include all models
        all_models = get_models_within_memory_limit(20.0)
        assert len(all_models) == len(AVAILABLE_MODELS)
        
        # Test with very small limit - should include only GPT-2 (0.5GB)
        tiny_models = get_models_within_memory_limit(1.0)
        assert len(tiny_models) == 1  # Only gpt2 (0.5GB) under 1GB limit
        assert "gpt2" in tiny_models


class TestGenerationConfig:
    """Test cases for generation configuration."""
    
    def test_get_generation_config(self):
        """Test generation configuration creation."""
        model_config = get_model_config("gpt2-large")
        gen_config = get_generation_config(model_config)
        
        expected_keys = [
            "max_new_tokens", "temperature", "top_p", "do_sample",
            "pad_token_id", "eos_token_id", "repetition_penalty", "no_repeat_ngram_size"
        ]
        
        for key in expected_keys:
            assert key in gen_config
        
        assert gen_config["max_new_tokens"] == model_config.max_new_tokens
        assert gen_config["temperature"] == model_config.temperature
        assert gen_config["top_p"] == model_config.top_p
        assert gen_config["do_sample"] is True
        
    def test_get_generation_config_conversational(self):
        """Test generation configuration for conversational GPT-2 model."""
        model_config = get_model_config("gpt2-large-conversational")
        gen_config = get_generation_config(model_config)
        
        # Should include model-specific settings
        assert gen_config["top_k"] == 23
        assert gen_config["repetition_penalty"] == 1.176
        assert gen_config["temperature"] == 0.3
        assert gen_config["top_p"] == 0.7


class TestPromptFormatting:
    """Test cases for prompt formatting."""
    
    def test_format_prompt_for_model_simple(self):
        """Test basic prompt formatting."""
        config = get_model_config("gpt2-large")
        prompt = format_prompt_for_model("Hello, world!", config)
        
        # GPT-2 uses simple template
        assert prompt == "Hello, world!"
    
    def test_format_prompt_for_model_llama(self):
        """Test Llama prompt formatting."""
        config = get_model_config("llama-3.2-1b")
        prompt = format_prompt_for_model("Hello, world!", config)
        
        assert prompt == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello, world!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    def test_format_prompt_for_model_with_history(self):
        """Test prompt formatting with conversation history."""
        config = get_model_config("gpt2-large")
        history = "User: Hi\nAssistant: Hello!"
        prompt = format_prompt_for_model("How are you?", config, history)
        
        expected = "User: Hi\nAssistant: Hello!\n\nUser: How are you?\nAssistant:"
        assert prompt == expected
    
    def test_format_prompt_for_model_phi(self):
        """Test Phi model prompt formatting."""
        config = get_model_config("phi-3-mini")
        prompt = format_prompt_for_model("Test question", config)
        
        assert prompt == "<|user|>\nTest question<|end|>\n<|assistant|>\n"
    
    def test_format_prompt_for_model_conversational_gpt2(self):
        """Test conversational GPT-2 prompt formatting."""
        config = get_model_config("gpt2-large-conversational")
        prompt = format_prompt_for_model("Hello there!", config)
        
        assert prompt == "<|USER|> Hello there! <|ASSISTANT|> "
        
    def test_format_prompt_for_model_conversational_with_history(self):
        """Test conversational GPT-2 formatting with history."""
        config = get_model_config("gpt2-large-conversational")
        history = "User: Hi\nAssistant: Hello!"
        prompt = format_prompt_for_model("How are you?", config, history)
        
        expected = "<|USER|> User: Hi\nAssistant: Hello!\n\nUser: How are you?\nAssistant: <|ASSISTANT|> "
        assert prompt == expected


class TestSystemMessages:
    """Test cases for system message handling."""
    
    def test_get_system_message_no_system_prompt(self):
        """Test system message for models that don't use system prompts."""
        config = get_model_config("gpt2-large")
        message = get_system_message(config)
        
        assert message is None
    
    def test_get_system_message_with_system_prompt(self):
        """Test system message for models that use system prompts."""
        # Create a mock config that requires system prompt
        config = ModelConfig(
            model_path="test",
            model_family="test",
            prompt_template="test",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            estimated_memory_gb=1.0,
            requires_system_prompt=True
        )
        
        message = get_system_message(config)
        assert message is not None
        assert "helpful AI assistant" in message


class TestOptimizationConfig:
    """Test cases for optimization configuration."""
    
    def test_get_optimization_config_known_family(self):
        """Test optimization config for known model family."""
        config = get_model_config("llama-3.2-1b")
        opt_config = get_optimization_config(config)
        
        assert opt_config["use_cache"] is True
        assert opt_config["gradient_checkpointing"] is True
    
    def test_get_optimization_config_unknown_family(self):
        """Test optimization config for unknown model family."""
        # Create mock config with unknown family
        config = ModelConfig(
            model_path="test",
            model_family="unknown",
            prompt_template="test",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            estimated_memory_gb=1.0
        )
        
        opt_config = get_optimization_config(config)
        
        # Should return default config
        assert opt_config["use_cache"] is True
        assert opt_config["gradient_checkpointing"] is True
    
    def test_get_optimization_config_phi(self):
        """Test optimization config for Phi model (requires trust_remote_code)."""
        config = get_model_config("phi-3-mini")
        opt_config = get_optimization_config(config)
        
        assert opt_config["trust_remote_code"] is True


class TestModelConfigConsistency:
    """Test cases for model configuration consistency."""
    
    def test_all_models_have_required_fields(self):
        """Test that all models have required configuration fields."""
        for model_name, config in AVAILABLE_MODELS.items():
            # Check required fields
            assert hasattr(config, 'model_path')
            assert hasattr(config, 'model_family')
            assert hasattr(config, 'prompt_template')
            assert hasattr(config, 'max_new_tokens')
            assert hasattr(config, 'temperature')
            assert hasattr(config, 'top_p')
            assert hasattr(config, 'estimated_memory_gb')
            
            # Check field types
            assert isinstance(config.model_path, str)
            assert isinstance(config.model_family, str)
            assert isinstance(config.max_new_tokens, int)
            assert isinstance(config.temperature, float)
            assert isinstance(config.top_p, float)
            assert isinstance(config.estimated_memory_gb, float)
            
            # Check reasonable ranges
            assert 0 < config.temperature <= 1.0
            assert 0 < config.top_p <= 1.0
            assert config.max_new_tokens > 0
            assert config.estimated_memory_gb > 0
    
    def test_prompt_templates_valid(self):
        """Test that all prompt templates contain {instruction} placeholder."""
        for model_name, config in AVAILABLE_MODELS.items():
            assert "{instruction}" in config.prompt_template, f"Model {model_name} missing {{instruction}} in template"
    
    def test_memory_estimates_reasonable(self):
        """Test that memory estimates are reasonable."""
        for model_name, config in AVAILABLE_MODELS.items():
            # Memory should be between 0.5GB and 20GB (reasonable range)
            assert 0.5 <= config.estimated_memory_gb <= 20.0, f"Model {model_name} has unreasonable memory estimate"


if __name__ == "__main__":
    pytest.main([__file__])