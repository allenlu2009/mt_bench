"""Shared response-generation routines for MT-bench evaluators."""

import time
from typing import Any

from ..utils.data_loader import MTBenchQuestion


async def generate_question_session(
    *,
    model_name: str,
    question: MTBenchQuestion,
    model_config: Any,
    conversation_handler: Any,
    model_manager: Any,
    memory_monitor: Any,
    logger: Any,
    turn1_only: bool = False,
) -> Any:
    """
    Generate a full conversation session for one MT-bench question.

    Returns a ConversationSession from conversation_handler.
    """
    session_id = conversation_handler.start_conversation(question, model_name)
    try:
        turns_to_process = [1] if turn1_only else [1, 2]
        for turn_number in turns_to_process:
            tokenizer = model_manager.get_current_tokenizer()
            prompt = conversation_handler.format_turn_prompt(
                session_id, turn_number, question, model_config, tokenizer
            )

            start_time = time.time()
            response = model_manager.generate_response(prompt, model_name)
            generation_time = time.time() - start_time
            memory_after = memory_monitor.get_gpu_memory_usage()
            memory_monitor.log_memory_usage(f"Generated response for Q{question.question_id}", logger)

            conversation_handler.add_turn_response(
                session_id=session_id,
                turn_number=turn_number,
                question=question,
                response=response,
                generation_time=generation_time,
                memory_used_gb=memory_after,
            )

        return conversation_handler.end_conversation(session_id)
    except Exception:
        conversation_handler.cleanup_session(session_id)
        raise

