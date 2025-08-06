"""OpenAI GPT-4.1-nano judge client for MT-bench evaluation."""

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
    judge_model: str = "gpt-4.1-nano"


class JudgeClient:
    """
    OpenAI GPT-4.1-nano client for judging model responses in MT-bench evaluation.
    
    Features:
    - Rate limiting compliance (max 10 req/sec for GPT-4.1-nano)
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

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano"):
        """
        Initialize judge client.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Judge model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Rate limiting: GPT-4.1-nano allows max 10 requests per second
        self.max_requests_per_second = 10
        self.request_timestamps = []
        
        logger.info(f"JudgeClient initialized with model: {model}")
    
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
    
    def _format_judge_prompt(self, question: str, answer: str) -> str:
        """
        Format the judge prompt with question and answer.
        
        Args:
            question: The original question
            answer: The model's answer to evaluate
            
        Returns:
            Formatted prompt string
        """
        return self.JUDGE_PROMPT_TEMPLATE.format(
            question=question.strip(),
            answer=answer.strip()
        )
    
    def _parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse judge response to extract score and reasoning.
        
        Args:
            response_text: Raw response from judge model
            
        Returns:
            Dictionary with 'score' and 'reasoning' keys
        """
        try:
            # Look for rating pattern [[X]] or Rating: [[X]]
            import re
            
            # Try to find rating in [[rating]] format
            rating_match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response_text)
            
            if rating_match:
                score = float(rating_match.group(1))
                
                # Extract reasoning (everything before the rating)
                rating_start = rating_match.start()
                reasoning = response_text[:rating_start].strip()
                
                # Clean up reasoning
                reasoning = re.sub(r'^(Rating:|Explanation:|Reasoning:)\s*', '', reasoning, flags=re.IGNORECASE)
                reasoning = reasoning.strip()
                
                return {
                    "score": score,
                    "reasoning": reasoning or "No reasoning provided"
                }
            
            # Fallback: try to extract any number between 1-10
            number_matches = re.findall(r'\b([1-9]|10)\b', response_text)
            if number_matches:
                score = float(number_matches[-1])  # Take the last number found
                return {
                    "score": score,
                    "reasoning": response_text.strip()
                }
            
            logger.warning(f"Could not parse rating from response: {response_text[:100]}...")
            return {
                "score": 0.0,
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
        prompt = self._format_judge_prompt(question, answer)
        
        try:
            logger.debug(f"Judging response for {model_name}, Q{question_id}, Turn {turn}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI model evaluator. Your task is to "
                            "evaluate AI assistant responses objectively and fairly. "
                            "Provide detailed reasoning followed by a numerical rating "
                            "in the format [[rating]]."
                        )
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistent judging
                max_tokens=1000,
                top_p=1.0
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the response
            parsed = self._parse_judge_response(response_text)
            
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