"""Shared model runtime for loading, unloading, and memory management."""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.model_configs import ModelConfig, get_model_config, get_optimization_config
from ..utils.memory_utils import MemoryMonitor, get_flash_attention_config, optimize_for_rtx3060

logger = logging.getLogger(__name__)


class ModelRuntime:
    """Shared runtime abstraction for model lifecycle management."""

    def __init__(self, device: str = "cuda", memory_limit_gb: float = 8.0):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.memory_monitor = MemoryMonitor(gpu_memory_limit_gb=memory_limit_gb)
        self.current_model: Optional[Any] = None
        self.current_tokenizer: Optional[Any] = None
        self.current_model_name: Optional[str] = None

        if self.device == "cuda":
            optimize_for_rtx3060()

        logger.info("ModelRuntime initialized on %s", self.device)

    def unload_model(self) -> None:
        """Unload currently active model and tokenizer."""
        if self.current_model is not None:
            logger.info("Cleaning up model: %s", self.current_model_name)
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.memory_monitor.cleanup_gpu_memory()

    def _get_model_loading_config(self, model_config: ModelConfig) -> Dict[str, Any]:
        if model_config.model_family == "gemma3":
            config: Dict[str, Any] = {
                "trust_remote_code": True,
                "dtype": torch.bfloat16,
            }
            if self.device == "cuda":
                config["device_map"] = "cuda"
            return config

        config = self.memory_monitor.get_memory_optimization_config()
        config.update(get_flash_attention_config())
        config.update(get_optimization_config(model_config))
        if model_config.load_in_4bit:
            config["load_in_4bit"] = True
            config["dtype"] = torch.float16

        # Large models on high-VRAM single-GPU systems can get killed during
        # HF accelerate auto-sharding due to extra host-memory pressure while
        # assigning shards. Prefer direct CUDA placement in that case.
        if self.device == "cuda" and model_config.estimated_memory_gb >= 24:
            config["device_map"] = "cuda"
            config.pop("max_memory", None)
            config["dtype"] = torch.bfloat16
            logger.info(
                "Using direct CUDA placement for large model '%s' (estimated %.1fGB)",
                model_config.model_path,
                model_config.estimated_memory_gb,
            )
        return config

    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load model/tokenizer into runtime and return both."""
        if self.current_model_name == model_name:
            logger.info("Model %s already loaded", model_name)
            return self.current_model, self.current_tokenizer

        model_config = get_model_config(model_name)
        if not self.memory_monitor.can_load_model(model_name):
            raise RuntimeError(
                f"Model {model_name} estimated memory ({model_config.estimated_memory_gb:.1f}GB) "
                f"exceeds limit ({self.memory_monitor.gpu_memory_limit:.1f}GB)"
            )

        self.unload_model()
        self.memory_monitor.log_memory_usage("Before model loading", logger)
        logger.info("Loading model: %s (%s)", model_name, model_config.model_path)

        try:
            loading_config = self._get_model_loading_config(model_config)
            tokenizer = self._load_tokenizer(model_name, model_config, loading_config)
            model = self._load_model(model_config, loading_config)

            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name

            self.memory_monitor.check_memory_limit(f"loading {model_name}")
            self.memory_monitor.log_memory_usage("After model loading", logger)
            logger.info("Successfully loaded %s", model_name)
            return model, tokenizer
        except Exception:
            self.unload_model()
            raise

    def _load_tokenizer(self, model_name: str, model_config: ModelConfig, loading_config: Dict[str, Any]) -> Any:
        if model_config.model_family == "gemma3":
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                if hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_path,
                trust_remote_code=loading_config.get("trust_remote_code", False),
                padding_side="left",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        if model_config.chat_template_name == "gpt2_conversational":
            special_tokens = ["<|USER|>", "<|ASSISTANT|>"]
            vocab = tokenizer.get_vocab()
            new_tokens = [token for token in special_tokens if token not in vocab]
            if new_tokens:
                tokenizer.add_tokens(new_tokens)

        logger.info("Tokenizer loaded, vocab size: %s", len(tokenizer))
        return tokenizer

    def _load_model(self, model_config: ModelConfig, loading_config: Dict[str, Any]) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if model_config.model_family == "gemma3":
                model = AutoModelForCausalLM.from_pretrained(model_config.model_path, **loading_config)
            else:
                model_kwargs = {
                    "dtype": loading_config.get("dtype", loading_config.get("torch_dtype")),
                    "attn_implementation": loading_config.get("attn_implementation", "eager"),
                    "low_cpu_mem_usage": loading_config["low_cpu_mem_usage"],
                    "trust_remote_code": loading_config.get("trust_remote_code", False),
                    "use_cache": loading_config.get("use_cache", True),
                }
                if loading_config.get("load_in_4bit"):
                    model_kwargs["load_in_4bit"] = True
                if "device_map" in loading_config:
                    model_kwargs["device_map"] = loading_config["device_map"]
                model = AutoModelForCausalLM.from_pretrained(model_config.model_path, **model_kwargs)

        if model_config.model_family != "gemma3" and "device_map" not in loading_config:
            try:
                has_meta_tensors = any(p.is_meta for p in model.parameters())
                if has_meta_tensors:
                    model = model.to_empty(device=self.device)
                else:
                    model = model.to(self.device)
            except Exception:
                model = model.to_empty(device=self.device)

        model.eval()
        return model

    def get_model_info(self) -> Dict[str, Any]:
        if self.current_model is None:
            return {"status": "no_model_loaded"}

        config = get_model_config(self.current_model_name)
        memory_stats = self.memory_monitor.get_comprehensive_stats()
        return {
            "model_name": self.current_model_name,
            "model_path": config.model_path,
            "model_family": config.model_family,
            "estimated_memory_gb": config.estimated_memory_gb,
            "actual_memory_gb": memory_stats.gpu_allocated_gb,
            "device": self.device,
            "vocab_size": len(self.current_tokenizer) if self.current_tokenizer else None,
            "memory_stats": memory_stats,
        }
