"""Perplexity evaluator using sliding-window negative log-likelihood."""

import json
import logging
import math
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.model_configs import get_model_config
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Perplexity evaluation pipeline with model/dataset/chunk parameter sweeps."""

    def __init__(
        self,
        model_names: List[str],
        cache_dir: str = "data",
        memory_limit_gb: float = 6.0,
        max_questions: Optional[int] = None,
        output_dir: str = "results",
        datasets: Optional[List[str]] = None,
        block_size: int = 1024,
        stride_ratios: Optional[List[float]] = None,
        device: str = "auto",
        max_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        use_all_tokens: bool = False,
        handle_residue: bool = True,
    ):
        self.model_names = model_names
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_limit_gb = memory_limit_gb
        self.max_questions = max_questions
        self.output_dir = output_dir
        self.datasets = datasets or ["wikitext2"]
        self.block_size = max(2, block_size)
        self.stride_ratios = stride_ratios or [0.5, 1.0]
        self.device = self._resolve_device(device)
        self.max_samples = max_samples if max_samples is not None else max_questions
        self.max_tokens = max_tokens
        self.use_all_tokens = use_all_tokens
        self.handle_residue = handle_residue
        self.memory_monitor = MemoryMonitor(gpu_memory_limit_gb=memory_limit_gb)
        self._progress = {
            "models_completed": 0,
            "total_models": len(model_names),
            "total_tokens_evaluated": 0,
            "peak_memory_gb": 0.0,
        }

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run full perplexity evaluation."""
        all_results: List[Dict[str, Any]] = []
        start_time = datetime.now()

        for model_name in self.model_names:
            logger.info("Perplexity evaluating model: %s", model_name)
            model, tokenizer = self._load_model_and_tokenizer(model_name)
            max_length = self._get_model_max_length(model, tokenizer)
            adjusted_block_size = min(self.block_size, max_length) if max_length else self.block_size

            for dataset_name in self.datasets:
                text = self._load_dataset_text(dataset_name)
                for stride_ratio in self.stride_ratios:
                    stride_ratio = max(0.1, min(1.0, stride_ratio))
                    result = self._evaluate_with_backoff(
                        text=text,
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        initial_block_size=adjusted_block_size,
                        stride_ratio=stride_ratio,
                    )
                    all_results.append(result)
                    self._progress["total_tokens_evaluated"] += result["num_tokens"]

            self._unload_model(model, tokenizer)
            self._progress["models_completed"] += 1
            self._progress["peak_memory_gb"] = max(
                self._progress["peak_memory_gb"],
                self.memory_monitor.peak_memory,
            )

        duration_seconds = int((datetime.now() - start_time).total_seconds())
        finite_ppls = [r["perplexity"] for r in all_results if math.isfinite(r["perplexity"])]
        avg_ppl = (sum(finite_ppls) / len(finite_ppls)) if finite_ppls else float("inf")

        summary_report = self._build_summary(all_results, avg_ppl, duration_seconds)
        return {
            "evaluation_mode": "perplexity",
            "model_names": self.model_names,
            "datasets": self.datasets,
            "results": all_results,
            "summary": {
                "total_time_seconds": duration_seconds,
                "total_models": len(self.model_names),
                "total_datasets": len(self.datasets),
                "total_runs": len(all_results),
                "average_perplexity": round(avg_ppl, 4) if math.isfinite(avg_ppl) else float("inf"),
            },
            "summary_report": summary_report,
        }

    def export_results(self, results: Dict[str, Any], output_dir: str = "results") -> None:
        """Export perplexity results in JSON format."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"perplexity_results_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("Perplexity results exported to %s", json_path)

    def get_evaluation_progress(self) -> Dict[str, Any]:
        """Return current evaluation progress."""
        return self._progress.copy()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model_and_tokenizer(self, model_name: str) -> Tuple[Any, Any]:
        model_cfg = get_model_config(model_name)
        model_path = model_cfg.model_path
        logger.info("Loading perplexity model with direct loader: %s (%s)", model_name, model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "cuda"

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()
        self.memory_monitor.log_memory_usage(f"Loaded {model_name}", logger)
        return model, tokenizer

    def _unload_model(self, model: Any, tokenizer: Any) -> None:
        del tokenizer
        del model
        self.memory_monitor.cleanup_gpu_memory()

    def _get_model_max_length(self, model: Any, tokenizer: Any) -> Optional[int]:
        max_length = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if isinstance(max_length, int) and max_length > 0:
            return max_length
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and tokenizer_max > 0 and tokenizer_max < 10_000_000:
            return tokenizer_max
        return None

    def _load_dataset_text(self, dataset_name: str) -> str:
        dataset_key = dataset_name.lower()
        if dataset_key == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [x for x in dataset["text"] if str(x).strip()]
            return self._truncate_text("\n\n".join(texts))
        if dataset_key == "wikitext103":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            texts = [x for x in dataset["text"] if str(x).strip()]
            return self._truncate_text("\n\n".join(texts))
        if dataset_key == "ptb":
            return self._truncate_text(self._load_ptb_text())
        if dataset_key == "shakespeare":
            return self._truncate_text(self._load_shakespeare_text())
        raise ValueError(
            f"Unsupported perplexity dataset '{dataset_name}'. "
            f"Supported: wikitext2, wikitext103, ptb, shakespeare"
        )

    def _truncate_text(self, text: str) -> str:
        if not self.use_all_tokens and self.max_samples is not None and self.max_samples > 0:
            lines = text.splitlines()
            text = "\n".join(lines[: self.max_samples])
        if self.max_tokens is not None and self.max_tokens > 0:
            approx_chars = self.max_tokens * 4
            text = text[:approx_chars]
        return text

    def _load_ptb_text(self) -> str:
        ptb_file = self.cache_dir / "ptb_test.txt"
        if not ptb_file.exists():
            url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
            try:
                urllib.request.urlretrieve(url, str(ptb_file))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download PTB test set and no local cache at {ptb_file}: {e}"
                ) from e
        return ptb_file.read_text(encoding="utf-8")

    def _load_shakespeare_text(self) -> str:
        shakespeare_file = self.cache_dir / "shakespeare.txt"
        if shakespeare_file.exists():
            return shakespeare_file.read_text(encoding="utf-8")

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            shakespeare_file.write_text(response.text, encoding="utf-8")
            return response.text
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Shakespeare text and no local cache at {shakespeare_file}: {e}"
            ) from e

    def _tokenize_and_chunk(
        self, text: str, tokenizer: Any, block_size: int, stride_ratio: float
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
        # Disable max-length warning here: we deliberately chunk long sequences ourselves.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Token indices sequence length is longer than the specified maximum sequence length",
            )
            tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]  # type: ignore[index]
        seq_len = tokens.size(1)
        stride = max(1, int(block_size * stride_ratio))

        samples: List[torch.Tensor] = []
        begin_locs: List[Tuple[int, int, int]] = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + block_size, seq_len)
            chunk = tokens[:, begin_loc:end_loc]
            if chunk.size(1) < 2:
                continue
            samples.append(chunk)
            begin_locs.append((begin_loc, end_loc, prev_end_loc))
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if self.handle_residue and begin_locs:
            last_end = begin_locs[-1][1]
            if last_end < seq_len:
                residue_chunk = tokens[:, last_end:seq_len]
                if residue_chunk.size(1) >= 2:
                    samples.append(residue_chunk)
                    begin_locs.append((last_end, seq_len, last_end))

        return samples, begin_locs

    def _evaluate_text(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
        model_name: str,
        dataset_name: str,
        block_size: int,
        stride_ratio: float,
    ) -> Dict[str, Any]:
        samples, begin_locs = self._tokenize_and_chunk(text, tokenizer, block_size, stride_ratio)
        if not samples:
            raise RuntimeError(f"No valid token chunks created for dataset '{dataset_name}'")

        nll_sum = 0.0
        n_tokens = 0
        model_device = next(model.parameters()).device

        for input_ids, (begin_loc, end_loc, prev_end_loc) in zip(samples, begin_locs):
            input_ids = input_ids.to(device=model_device, dtype=torch.long)
            if input_ids.size(1) < 2:
                continue

            trg_len = end_loc - prev_end_loc
            target_ids = input_ids.clone()
            if trg_len < input_ids.size(1):
                target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            if num_loss_tokens > 0:
                nll_sum += float(neg_log_likelihood.item()) * num_loss_tokens
                n_tokens += int(num_loss_tokens)

        avg_nll = nll_sum / n_tokens if n_tokens > 0 else float("inf")
        perplexity = math.exp(avg_nll) if n_tokens > 0 and math.isfinite(avg_nll) else float("inf")

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "block_size": block_size,
            "stride_ratio": stride_ratio,
            "avg_nll": round(avg_nll, 4) if math.isfinite(avg_nll) else float("inf"),
            "perplexity": round(perplexity, 4) if math.isfinite(perplexity) else float("inf"),
            "num_tokens": n_tokens,
        }

    def _evaluate_with_backoff(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
        model_name: str,
        dataset_name: str,
        initial_block_size: int,
        stride_ratio: float,
    ) -> Dict[str, Any]:
        block_size = max(64, initial_block_size)
        min_block_size = 128 if self.device == "cuda" else 64
        last_error: Optional[Exception] = None

        while block_size >= min_block_size:
            try:
                if self.device == "cuda":
                    self.memory_monitor.cleanup_gpu_memory()
                if block_size != initial_block_size:
                    logger.warning(
                        "Retrying perplexity eval with smaller block_size=%s (model=%s dataset=%s stride_ratio=%.2f)",
                        block_size,
                        model_name,
                        dataset_name,
                        stride_ratio,
                    )
                return self._evaluate_text(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    block_size=block_size,
                    stride_ratio=stride_ratio,
                )
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                logger.warning(
                    "CUDA OOM at block_size=%s (model=%s dataset=%s). Reducing block size.",
                    block_size,
                    model_name,
                    dataset_name,
                )
                self.memory_monitor.cleanup_gpu_memory()
                block_size //= 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    last_error = e
                    logger.warning(
                        "Runtime OOM at block_size=%s (model=%s dataset=%s). Reducing block size.",
                        block_size,
                        model_name,
                        dataset_name,
                    )
                    self.memory_monitor.cleanup_gpu_memory()
                    block_size //= 2
                else:
                    raise

        raise RuntimeError(
            f"Perplexity evaluation failed after OOM backoff for model={model_name}, "
            f"dataset={dataset_name}, initial_block_size={initial_block_size}. "
            f"Try setting --max-tokens (e.g. 20000) and a smaller --block-size (e.g. 1024)."
        ) from last_error

    def _build_summary(self, all_results: List[Dict[str, Any]], avg_ppl: float, duration_seconds: int) -> str:
        lines = [
            "PERPLEXITY EVALUATION RESULTS",
            "=" * 50,
            f"Total runs: {len(all_results)}",
            f"Average perplexity: {avg_ppl:.4f}" if math.isfinite(avg_ppl) else "Average perplexity: inf",
            f"Total time: {duration_seconds}s",
            "",
        ]

        for result in all_results:
            lines.append(
                f"{result['model_name']} | {result['dataset_name']} | "
                f"ppl={result['perplexity']:.4f} | "
                f"tokens={result['num_tokens']} | "
                f"block={result['block_size']} stride_ratio={result['stride_ratio']:.2f}"
            )

        return "\n".join(lines)
