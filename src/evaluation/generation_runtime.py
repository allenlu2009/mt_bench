"""MT-bench response generation routines using an already-loaded model."""

import logging
from typing import Any, Dict, Optional

import torch

from ..models.model_configs import (
    get_family_behavior,
    get_generation_behavior,
    get_generation_config,
    get_model_config,
)

logger = logging.getLogger(__name__)


def generate_response_with_loaded_model(
    *,
    prompt: str,
    current_model_name: str,
    current_model: Any,
    current_tokenizer: Any,
    memory_monitor: Any,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a response from already-loaded model/tokenizer state."""
    generation_kwargs = generation_kwargs or {}
    config = get_model_config(current_model_name)
    gen_config = get_generation_config(config)
    gen_config.update(generation_kwargs)

    if gen_config["pad_token_id"] is None:
        gen_config["pad_token_id"] = (
            current_tokenizer.pad_token_id
            if current_tokenizer.pad_token_id is not None
            else current_tokenizer.eos_token_id
        )
    if gen_config["eos_token_id"] is None:
        gen_config["eos_token_id"] = current_tokenizer.eos_token_id

    vocab_size = len(current_tokenizer)
    for key in ["pad_token_id", "eos_token_id", "bos_token_id"]:
        if key in gen_config and gen_config[key] is not None and gen_config[key] >= vocab_size:
            logger.warning("%s (%s) >= vocab_size (%s), using eos_token_id", key, gen_config[key], vocab_size)
            gen_config[key] = current_tokenizer.eos_token_id

    memory_monitor.log_memory_usage("Before generation", logger)

    try:
        current_config = get_model_config(current_model_name)
        family_behavior = get_family_behavior(current_config)
        generation_behavior = get_generation_behavior(current_config)

        if family_behavior.use_chat_template_generation:
            messages = [{"role": "user", "content": prompt}]
            try:
                inputs = current_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.warning("Chat template failed for gemma3 (%s), using fallback", e)
                formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                inputs = current_tokenizer(formatted_prompt, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = current_model.generate(**inputs, max_new_tokens=256)

            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            for pattern in ["<pad>", "<mask>"]:
                response = response.replace(pattern, "")
            response = response.strip()

        elif generation_behavior.use_legacy_encode_generation:
            input_ids = current_tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            model_device = next(current_model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = torch.ones_like(input_ids).to(model_device)

            gen_config_copy = gen_config.copy()
            if "max_new_tokens" in gen_config_copy:
                max_length = len(input_ids[0]) + gen_config_copy.pop("max_new_tokens")
                gen_config_copy["max_length"] = min(max_length, 1024)

            with torch.no_grad():
                outputs = current_model.generate(input_ids, attention_mask=attention_mask, **gen_config_copy)

            input_length = input_ids.shape[1]
            if generation_behavior.decode_with_assistant_marker:
                full_output = current_tokenizer.decode(outputs[0], skip_special_tokens=False)
                marker = generation_behavior.assistant_marker
                if marker in full_output:
                    response = full_output.split(marker)[-1].strip()
                else:
                    response_tokens = outputs[0][input_length:]
                    response = current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            else:
                response_tokens = outputs[0][input_length:]
                response = current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        else:
            inputs = current_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            input_ids = inputs["input_ids"]
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.error(
                    "Input contains token IDs >= vocab_size: max_id=%s, vocab_size=%s",
                    max_token_id,
                    vocab_size,
                )
                inputs["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)
                logger.warning("Clamped input token IDs to valid range")

            model_device = next(current_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = current_model.generate(**inputs, **gen_config)

            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = current_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        memory_monitor.log_memory_usage("After generation", logger)
        return response

    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM during generation: %s", str(e))
        memory_monitor.cleanup_gpu_memory()
        raise RuntimeError(f"GPU out of memory during generation: {str(e)}")
    except Exception as e:
        logger.error("Generation failed: %s", str(e))
        raise

