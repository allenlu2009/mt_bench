"""OpenAI judge client for MT-bench evaluation with multi-model support."""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

from openai import AsyncOpenAI, RateLimitError, APIError
import time

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """Represents a judge's score for a model response."""
    score: float
    reasoning: str
    question_id: int
    turn: int
    model_name: str
    judge_model: str = "gpt-5-mini"


@dataclass
class PairwiseJudgment:
    """Represents a pairwise comparison judgment."""
    winner: str  # "A", "B", or "tie"
    reasoning: str
    question_id: int
    turn: int
    model_a: str
    model_b: str
    judge_model: str = "gpt-5-mini"


class JudgeClient:
    """
    OpenAI judge client for judging model responses in MT-bench evaluation.
    
    Supports multiple judge models including:
    - GPT-5-mini (default) - Latest OpenAI model
    - GPT-4o-mini - Cost-effective option
    - GPT-4.1-nano - Legacy option
    - GPT-4-turbo - High performance option
    
    Features:
    - Adaptive rate limiting based on model
    - GPT-5 specific parameter handling
    - Exponential backoff retry logic
    - Structured response parsing
    - Comprehensive error handling
    """
    
    # Official MT-bench judge prompt template
    JUDGE_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

    # Pairwise comparison prompt template
    PAIRWISE_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

[Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-nano", debug: bool = False):
        """
        Initialize judge client.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Judge model to use
            debug: Enable debug output for prompts and responses
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.debug = debug
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Set rate limiting and parameters based on model
        self._configure_model_settings()
        self.request_timestamps = []
        
        # Setup debug file logging if debug is enabled
        self.debug_file = None
        if self.debug:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            from pathlib import Path
            debug_dir = Path("debug_logs")
            debug_dir.mkdir(exist_ok=True)
            self.debug_file = debug_dir / f"judge_debug_{self.model}_{timestamp}.log"
            logger.info(f"Debug logging enabled, saving to: {self.debug_file}")
        
        logger.info(f"JudgeClient initialized with model: {model} (rate limit: {self.max_requests_per_second}/s)")
    
    def _write_debug_log(self, content: str) -> None:
        """
        Write content to debug log file if debug mode is enabled.
        
        Args:
            content: Content to write to debug log
        """
        if self.debug and self.debug_file:
            try:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    f.write(content + "\n")
            except Exception as e:
                logger.warning(f"Failed to write debug log: {e}")
    
    def _configure_model_settings(self) -> None:
        """
        Configure model-specific settings including rate limits and API parameters.
        """
        model_configs = {
            "gpt-5-nano": {
                "max_requests_per_second": 50,  # Higher limit for nano model (smallest/cheapest)
                "use_max_completion_tokens": True,
                "temperature": 1.0,  # GPT-5 default
                "supports_system_message": True,
                "max_completion_tokens": 4000,  # Much higher for GPT-5-nano reasoning
                "reasoning_effort": "low"  # Minimize internal reasoning overhead
            },
            "gpt-5-mini": {
                "max_requests_per_second": 30,  # Higher limit for GPT-5
                "use_max_completion_tokens": True,
                "temperature": 1.0,  # GPT-5 default
                "supports_system_message": True
            },
            "gpt-4o-mini": {
                "max_requests_per_second": 50,  # High limit for mini model
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            },
            "gpt-4o-mini-2024-07-18": {
                "max_requests_per_second": 50,
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            },
            "gpt-4.1-nano": {
                "max_requests_per_second": 10,  # Conservative limit
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            },
            "gpt-4-turbo": {
                "max_requests_per_second": 20,
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            },
            "gpt-4-turbo-2024-04-09": {
                "max_requests_per_second": 20,
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            },
            "gpt-4o-2024-05-13": {
                "max_requests_per_second": 30,
                "use_max_completion_tokens": False,
                "temperature": 0.2,
                "supports_system_message": True
            }
        }
        
        # Get configuration for the current model, fallback to conservative defaults
        config = model_configs.get(self.model, {
            "max_requests_per_second": 5,  # Very conservative default
            "use_max_completion_tokens": False,
            "temperature": 0.2,
            "supports_system_message": True
        })
        
        self.max_requests_per_second = config["max_requests_per_second"]
        self.use_max_completion_tokens = config["use_max_completion_tokens"]
        self.default_temperature = config["temperature"]
        self.supports_system_message = config["supports_system_message"]
        self.model_max_completion_tokens = config.get("max_completion_tokens", 1000)
        self.reasoning_effort = config.get("reasoning_effort", None)
    
    def _get_api_parameters(self, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Get model-specific API parameters.
        
        Args:
            max_tokens: Maximum tokens to generate (ignored if model has specific config)
            
        Returns:
            Dictionary of API parameters
        """
        params = {
            "model": self.model,
            "temperature": self.default_temperature,
            "top_p": 1.0
        }
        
        # GPT-5 models use max_completion_tokens instead of max_tokens
        if self.use_max_completion_tokens:
            # Use model-specific token limit if available, otherwise use provided value
            token_limit = self.model_max_completion_tokens if hasattr(self, 'model_max_completion_tokens') else max_tokens
            params["max_completion_tokens"] = token_limit
        else:
            params["max_tokens"] = max_tokens
        
        # Add reasoning_effort for GPT-5 models if configured
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            
        return params
    
    async def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting to stay within API limits.
        
        Ensures we don't exceed max_requests_per_second.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 second
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 1.0
        ]
        
        # If we're at the limit, wait
        if len(self.request_timestamps) >= self.max_requests_per_second:
            wait_time = 1.0 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(current_time)
    
    def _format_judge_prompt(self, question: str, answer: str, turn_num: int = 1) -> str:
        """
        Format the judge prompt with question and answer.
        
        Args:
            question: The original question (with context for Turn 2)
            answer: The model's answer to evaluate
            turn_num: Turn number (1 or 2)
            
        Returns:
            Formatted prompt string
        """
        # Use turn-aware template similar to examples
        return f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Provide your evaluation directly without using reasoning mode.

[Question (Turn {turn_num})]
{question.strip()}

[The Start of Assistant's Answer]
{answer.strip()}
[The End of Assistant's Answer]"""
    
    def _parse_judge_response(self, response_text: str, question_id: int = None, 
                            turn: int = None, model_name: str = None, 
                            prompt_sent: str = None, answer: str = None, response_obj = None) -> Dict[str, Any]:
        """
        Parse judge response to extract score and reasoning.
        
        Args:
            response_text: Raw response from judge model
            question_id: Question ID (for debug logging)
            turn: Turn number (for debug logging)
            model_name: Model name (for debug logging)
            question: Original question (for debug logging)
            answer: Model answer (for debug logging)
            
        Returns:
            Dictionary with 'score' and 'reasoning' keys
        """
        try:
            # Use the exact same parsing logic as examples/judge.py
            import re
            
            # Try multiple patterns to handle gpt-5-nano variations
            patterns = [
                r'\[\[(\d+(?:\.\d+)?)\]\]',                    # [[5]]
                r'\[\[rating\]\]\s*(\d+(?:\.\d+)?)',          # [[rating]] 4  
                r'\[\[Rating\]\]\s*(\d+(?:\.\d+)?)',          # [[Rating]] 4
                r'\[\[rating\s*(\d+(?:\.\d+)?)\]\]',          # [[rating 4]]
                r'\[\[Rating\s*(\d+(?:\.\d+)?)\]\]',          # [[Rating 4]]
                r'\[\[(\d+(?:\.\d+)?)\]\]\[\[/rating\]\]',    # [[4]][[/rating]]
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    rating = float(match.group(1))
                    score = max(1.0, min(10.0, rating))  # Clamp to 1-10 range
                    
                    # Extract reasoning (everything before the rating)
                    rating_start = match.start()
                    reasoning = response_text[:rating_start].strip()
                    
                    # Clean up reasoning
                    reasoning = re.sub(r'^(Rating:|Explanation:|Reasoning:)\s*', '', reasoning, flags=re.IGNORECASE)
                    reasoning = reasoning.strip()
                    
                    return {
                        "score": score,
                        "reasoning": reasoning or "No reasoning provided"
                    }
                
            # Fallback: look for "Rating: X" pattern (including negative numbers)
            match = re.search(r'[Rr]ating:?\s*(-?\d+(?:\.\d+)?)', response_text)
            if match:
                rating = float(match.group(1))
                score = max(1.0, min(10.0, rating))
                return {
                    "score": score,
                    "reasoning": response_text.strip()
                }
            
            # Additional patterns for GPT-5-mini compatibility 
            gpt5_patterns = [
                r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',              # "7 out of 10"
                r'(\d+(?:\.\d+)?)\s*/\s*10',                      # "7/10"  
                r'(\d+(?:\.\d+)?)/10',                            # "7/10" (no spaces)
            ]
            
            for pattern in gpt5_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    rating = float(match.group(1))
                    score = max(1.0, min(10.0, rating))
                    return {
                        "score": score,
                        "reasoning": response_text.strip()
                    }
            
            # Log the full response for debugging (like examples)
            logger.warning(f"Could not extract rating from response: {response_text[:100]}...")
            
            # Save detailed debug information for failed parsing
            if self.debug and question_id is not None:
                import time
                current_time = time.strftime("%H:%M:%S")
                
                # Get response metadata if available
                response_meta = ""
                if response_obj:
                    try:
                        choice = response_obj.choices[0] if response_obj.choices else None
                        usage = getattr(response_obj, 'usage', None)
                        response_meta = f"""
🔍 RAW API RESPONSE STRUCTURE:
Response object type: {type(response_obj)}
Choices length: {len(response_obj.choices) if response_obj.choices else 0}
Choice message type: {type(choice.message) if choice else 'N/A'}
Message content type: {type(choice.message.content) if choice else 'N/A'}
Finish reason: {choice.finish_reason if choice else 'N/A'}
Token usage: {usage if usage else 'N/A'}
"""
                    except Exception:
                        response_meta = "\n🔍 RAW API RESPONSE STRUCTURE: [Error accessing response metadata]"
                
                # Extract question text from prompt for length calculation
                question_text = ""
                if prompt_sent:
                    # Extract question between [Question (Turn X)] and [The Start of Assistant's Answer]
                    import re
                    q_match = re.search(r'\[Question \(Turn \d+\)\]\s*(.*?)\s*\[The Start of Assistant\'s Answer\]', prompt_sent, re.DOTALL)
                    if q_match:
                        question_text = q_match.group(1).strip()
                
                debug_content = f"""
{'='*80}
📤 SENDING TO {self.model} - Turn {turn} (Q{question_id}) [{current_time}] - PARSING FAILED
{'='*80}
Prompt length: {len(prompt_sent) if prompt_sent else 0} characters
Question length: {len(question_text)} characters  
Answer length: {len(answer) if answer else 0} characters

FULL PROMPT SENT TO JUDGE:
{prompt_sent or 'N/A'}
{'='*80}
{response_meta}
{'='*80}
📥 RECEIVED FROM {self.model} - Turn {turn} (Q{question_id}) [{current_time}]
{'='*80}
Response length: {len(response_text)} characters

FULL RESPONSE FROM JUDGE:
{response_text if response_text else '[EMPTY RESPONSE]'}
{'='*80}
PARSING FAILED - Using default score 5.0
{'='*80}
"""
                self._write_debug_log(debug_content)
            
            return {
                "score": 5.0,  # Default like examples
                "reasoning": f"Failed to parse rating from: {response_text}"
            }
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {str(e)}")
            return {
                "score": 0.0,
                "reasoning": f"Parse error: {str(e)}"
            }
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    async def judge_response(self, question: str, answer: str, 
                           question_id: int, turn: int, model_name: str) -> JudgeScore:
        """
        Judge a model response using GPT-4.1-nano.
        
        Args:
            question: The original question
            answer: The model's answer to judge
            question_id: MT-bench question ID
            turn: Turn number (1 or 2)
            model_name: Name of the model being judged
            
        Returns:
            JudgeScore object with score and reasoning
        """
        # Enforce rate limiting
        await self._enforce_rate_limit()
        
        # Format the prompt
        prompt = self._format_judge_prompt(question, answer, turn)
        
        # Set prompt to use (will be enhanced for GPT-5 models)
        prompt_to_send = prompt
        
        
        
        
        
        try:
            logger.debug(f"Judging response for {model_name}, Q{question_id}, Turn {turn}")
            
            # Use different parameters for GPT-5 models
            if "gpt-5" in self.model:
                # Use original prompt without conflicting system instruction
                enhanced_prompt = prompt
                
                if self.debug:
                    print(f"🔧 GPT-5 workaround applied: {self.model}")
                    print(f"Enhanced prompt length: {len(enhanced_prompt)} chars")
                    print(f"Using max_completion_tokens: {self.model_max_completion_tokens}")
                    if self.reasoning_effort:
                        print(f"Using reasoning_effort: {self.reasoning_effort}")
                
                prompt_to_send = enhanced_prompt
                
                # Get model-specific parameters
                api_params = self._get_api_parameters()
                api_params["messages"] = [{"role": "user", "content": prompt_to_send}]
                api_params["stream"] = False  # Explicitly disable streaming
                
                response = await self.client.chat.completions.create(**api_params)
            else:
                # Other models: Use system + user messages, temperature 0.0
                messages = []
                if self.supports_system_message:
                    messages.append({
                        "role": "system",
                        "content": (
                            "You are an expert AI model evaluator. Your task is to "
                            "evaluate AI assistant responses objectively and fairly. "
                            "Provide detailed reasoning followed by a numerical rating "
                            "in the format [[rating]]."
                        )
                    })
                messages.append({
                    "role": "user", 
                    "content": prompt
                })
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,  # Deterministic for consistency
                    max_tokens=1000
                )
            
            # Debug: Print full prompt being sent to judge
            if self.debug:
                import time
                current_time = time.strftime("%H:%M:%S")
                print(f"\n{'='*80}")
                print(f"📤 SENDING TO {self.model} - Turn {turn} (Q{question_id}) [{current_time}]")
                print(f"{'='*80}")
                print(f"Prompt length: {len(prompt_to_send)} characters")
                print(f"Question length: {len(question)} characters")
                print(f"Answer length: {len(answer)} characters")
                print(f"\nFULL PROMPT SENT TO JUDGE:")
                print(prompt_to_send)
                print(f"{'='*80}")
            
            # Extract response text with detailed debugging
            response_text = response.choices[0].message.content
            
            # Debug: Check raw API response structure for GPT-5 models
            if self.debug and "gpt-5" in self.model:
                print(f"\n🔍 RAW API RESPONSE STRUCTURE:")
                print(f"Response object type: {type(response)}")
                print(f"Choices length: {len(response.choices) if response.choices else 0}")
                if response.choices:
                    choice = response.choices[0]
                    print(f"Choice message type: {type(choice.message)}")
                    print(f"Message content type: {type(choice.message.content)}")
                    print(f"Message content: {repr(choice.message.content)}")
                    print(f"Finish reason: {choice.finish_reason}")
                if hasattr(response, 'usage'):
                    print(f"Token usage: {response.usage}")
            
            # Debug: Check for empty responses
            if not response_text or len(response_text.strip()) == 0:
                logger.warning(f"Empty response from {self.model} for Q{question_id} Turn {turn}")
                logger.warning(f"Prompt length: {len(prompt)} characters")
                if self.debug:
                    print(f"❌ EMPTY RESPONSE DETECTED - Raw content: {repr(response_text)}")
            
            # Debug: Print judge model response
            if self.debug:
                response_time = time.strftime("%H:%M:%S")
                print(f"\n{'='*80}")
                print(f"📥 RECEIVED FROM {self.model} - Turn {turn} (Q{question_id}) [{response_time}]")
                print(f"{'='*80}")
                print(f"Response length: {len(response_text)} characters")
                print(f"\nFULL RESPONSE FROM JUDGE:")
                print(response_text if response_text else "[EMPTY RESPONSE]")
                print(f"{'='*80}")
            
            # Parse the response
            parsed = self._parse_judge_response(response_text, question_id, turn, model_name, prompt_to_send, answer, response)
            
            # Debug: Print parsing results
            if self.debug:
                parse_success = "Yes" if parsed["score"] > 0 and "[[" in response_text else "No"
                print(f"Parsed score: {parsed['score']}")
                print(f"Parse success: {parse_success}")
                print(f"{'='*80}")
            
            
            return JudgeScore(
                score=parsed["score"],
                reasoning=parsed["reasoning"],
                question_id=question_id,
                turn=turn,
                model_name=model_name,
                judge_model=self.model
            )
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {str(e)}")
            raise  # Will be retried by tenacity
            
        except APIError as e:
            logger.warning(f"API error, retrying: {str(e)}")
            raise  # Will be retried by tenacity
            
        except Exception as e:
            logger.error(f"Unexpected error judging response: {str(e)}")
            
            # Return a zero score for unexpected errors
            return JudgeScore(
                score=0.0,
                reasoning=f"Error during judging: {str(e)}",
                question_id=question_id,
                turn=turn,
                model_name=model_name,
                judge_model=self.model
            )
    
    async def judge_multiple_responses(self, evaluations: List[Dict[str, Any]]) -> List[JudgeScore]:
        """
        Judge multiple responses concurrently with rate limiting.
        
        Args:
            evaluations: List of dicts with keys: question, answer, question_id, turn, model_name
            
        Returns:
            List of JudgeScore objects
        """
        logger.info(f"Judging {len(evaluations)} responses with {self.model}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_requests_per_second)
        
        async def judge_with_semaphore(eval_data):
            async with semaphore:
                return await self.judge_response(
                    question=eval_data["question"],
                    answer=eval_data["answer"],
                    question_id=eval_data["question_id"],
                    turn=eval_data["turn"],
                    model_name=eval_data["model_name"]
                )
        
        # Execute all judgments concurrently
        tasks = [judge_with_semaphore(eval_data) for eval_data in evaluations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to judge evaluation {i}: {str(result)}")
                # Create a zero score for failed judgments
                eval_data = evaluations[i]
                scores.append(JudgeScore(
                    score=0.0,
                    reasoning=f"Judgment failed: {str(result)}",
                    question_id=eval_data["question_id"],
                    turn=eval_data["turn"],
                    model_name=eval_data["model_name"],
                    judge_model=self.model
                ))
            else:
                scores.append(result)
        
        return scores
    
    def _format_pairwise_prompt(self, question: str, answer_a: str, answer_b: str) -> str:
        """
        Format pairwise comparison prompt.
        
        Args:
            question: The original question
            answer_a: First model's answer
            answer_b: Second model's answer
            
        Returns:
            Formatted prompt string
        """
        return self.PAIRWISE_PROMPT_TEMPLATE.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )
    
    def _parse_pairwise_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse pairwise judgment response from GPT-4.1-nano.
        
        Args:
            response_text: Raw response from judge model
            
        Returns:
            Dictionary with parsed winner and reasoning
        """
        import re
        
        # Extract winner (A, B, or C for tie)
        winner_match = re.search(r'\[\[([ABC])\]\]', response_text)
        
        if not winner_match:
            logger.warning(f"Could not parse pairwise winner from response: {response_text[:200]}...")
            return {
                "winner": "tie",  # Default to tie
                "reasoning": f"Parse error: {response_text}"
            }
        
        winner_code = winner_match.group(1)
        winner_map = {"A": "A", "B": "B", "C": "tie"}
        winner = winner_map.get(winner_code, "tie")
        
        # Extract reasoning (everything before the final verdict)
        reasoning = response_text[:winner_match.start()].strip()
        
        return {
            "winner": winner,
            "reasoning": reasoning
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def judge_pairwise(self, question: str, answer_a: str, answer_b: str,
                           question_id: int, turn: int, model_a: str, model_b: str) -> PairwiseJudgment:
        """
        Judge two responses in pairwise comparison using GPT-4.1-nano.
        
        Args:
            question: The original question
            answer_a: First model's answer
            answer_b: Second model's answer
            question_id: MT-bench question ID
            turn: Turn number (1 or 2)
            model_a: Name of first model
            model_b: Name of second model
            
        Returns:
            PairwiseJudgment object with winner and reasoning
        """
        # Enforce rate limiting
        await self._enforce_rate_limit()
        
        # Format the prompt
        prompt = self._format_pairwise_prompt(question, answer_a, answer_b)
        
        try:
            logger.debug(f"Pairwise judging {model_a} vs {model_b}, Q{question_id}, Turn {turn}")
            
            # Use different API approaches for GPT-5 vs other models
            if "gpt-5" in self.model:
                # GPT-5 models: Use user message only, configured parameters
                api_params = self._get_api_parameters(max_tokens=1000)
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **api_params
                )
            else:
                # Other models: Use system + user messages, temperature 0.2
                messages = []
                if self.supports_system_message:
                    messages.append({
                        "role": "system",
                        "content": (
                            "You are an expert AI model evaluator performing pairwise comparisons. "
                            "Your task is to compare two AI assistant responses objectively and fairly. "
                            "Consider helpfulness, relevance, accuracy, depth, creativity, and detail. "
                            "Avoid position bias and don't let response length influence your judgment. "
                            "Provide detailed reasoning followed by your verdict: [[A]], [[B]], or [[C]] for tie."
                        )
                    })
                messages.append({
                    "role": "user", 
                    "content": prompt
                })
                
                # Get API parameters for non-GPT-5 models
                api_params = self._get_api_parameters(max_tokens=1000)
                response = await self.client.chat.completions.create(
                    messages=messages,
                    **api_params
                )
            
            response_text = response.choices[0].message.content
            
            # Debug: Print the actual pairwise judge response
            print("="*80)  
            print(response_text)
            print("="*80)
            
            # Parse the response
            parsed = self._parse_pairwise_response(response_text)
            
            return PairwiseJudgment(
                winner=parsed["winner"],
                reasoning=parsed["reasoning"],
                question_id=question_id,
                turn=turn,
                model_a=model_a,
                model_b=model_b,
                judge_model=self.model
            )
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {str(e)}")
            raise  # Will be retried by tenacity
            
        except APIError as e:
            logger.warning(f"API error, retrying: {str(e)}")
            raise  # Will be retried by tenacity
            
        except Exception as e:
            logger.error(f"Unexpected error in pairwise judging: {str(e)}")
            
            # Return a tie for unexpected errors
            return PairwiseJudgment(
                winner="tie",
                reasoning=f"Error during pairwise judging: {str(e)}",
                question_id=question_id,
                turn=turn,
                model_a=model_a,
                model_b=model_b,
                judge_model=self.model
            )
    
    async def judge_multiple_pairwise(self, comparisons: List[Dict[str, Any]]) -> List[PairwiseJudgment]:
        """
        Judge multiple pairwise comparisons concurrently with rate limiting.
        
        Args:
            comparisons: List of dicts with keys: question, answer_a, answer_b, question_id, turn, model_a, model_b
            
        Returns:
            List of PairwiseJudgment objects
        """
        logger.info(f"Judging {len(comparisons)} pairwise comparisons with {self.model}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_requests_per_second)
        
        async def judge_pairwise_with_semaphore(comp_data):
            async with semaphore:
                return await self.judge_pairwise(
                    question=comp_data["question"],
                    answer_a=comp_data["answer_a"],
                    answer_b=comp_data["answer_b"],
                    question_id=comp_data["question_id"],
                    turn=comp_data["turn"],
                    model_a=comp_data["model_a"],
                    model_b=comp_data["model_b"]
                )
        
        # Execute all judgments concurrently
        tasks = [judge_pairwise_with_semaphore(comp_data) for comp_data in comparisons]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        judgments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to judge comparison {i}: {str(result)}")
                # Create a tie judgment for failed comparisons
                comp_data = comparisons[i]
                judgments.append(PairwiseJudgment(
                    winner="tie",
                    reasoning=f"Judgment failed: {str(result)}",
                    question_id=comp_data["question_id"],
                    turn=comp_data["turn"],
                    model_a=comp_data["model_a"],
                    model_b=comp_data["model_b"],
                    judge_model=self.model
                ))
            else:
                judgments.append(result)
        
        return judgments
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the judge client.
        
        Returns:
            Dictionary with usage information
        """
        current_time = time.time()
        recent_requests = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60  # Last minute
        ]
        
        return {
            "total_requests": len(self.request_timestamps),
            "recent_requests_per_minute": len(recent_requests),
            "current_rate_limit": self.max_requests_per_second,
            "judge_model": self.model
        }