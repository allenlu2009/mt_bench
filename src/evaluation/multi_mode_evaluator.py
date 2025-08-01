"""Multi-mode MT-bench evaluator supporting both single and pairwise evaluation."""

import asyncio
import logging
from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .mtbench_evaluator import MTBenchEvaluator
from .judge_client import JudgeClient, PairwiseJudgment
from ..utils.response_manager import ResponseManager
from ..utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class MultiModeEvaluator:
    """
    MT-bench evaluator supporting multiple evaluation modes.
    
    Modes:
    - single: Traditional absolute scoring (1-10 scale)
    - pairwise: Head-to-head comparison between model pairs
    - both: Both single and pairwise evaluation
    """
    
    def __init__(self,
                 model_names: List[str],
                 openai_api_key: str,
                 judge_model: str = "gpt-4.1-nano",
                 cache_dir: str = "data",
                 response_cache_dir: str = "cached_responses",
                 memory_limit_gb: float = 6.0,
                 max_questions: Optional[int] = None,
                 disable_response_cache: bool = False):
        """
        Initialize multi-mode evaluator.
        
        Args:
            model_names: List of model names to evaluate
            openai_api_key: OpenAI API key for judge
            judge_model: Judge model to use
            cache_dir: Directory for caching MT-bench data
            response_cache_dir: Directory for caching model responses
            memory_limit_gb: GPU memory limit
            max_questions: Maximum questions to evaluate (for testing)
            disable_response_cache: Disable response caching
        """
        self.model_names = model_names
        self.openai_api_key = openai_api_key
        self.judge_model = judge_model
        self.max_questions = max_questions
        self.memory_limit_gb = memory_limit_gb
        
        # Initialize components
        self.single_evaluator = MTBenchEvaluator(
            model_names=model_names,
            openai_api_key=openai_api_key,
            judge_model=judge_model,
            cache_dir=cache_dir,
            memory_limit_gb=memory_limit_gb,
            max_questions=max_questions
        )
        
        self.response_manager = ResponseManager(
            cache_dir=response_cache_dir,
            use_disk_cache=not disable_response_cache
        )
        
        self.data_loader = DataLoader(cache_dir)
        self.judge_client = JudgeClient(openai_api_key, judge_model)
        
        # Track evaluation progress
        self.progress = {
            "models_completed": 0,
            "total_models": len(model_names),
            "single_scores": 0,
            "pairwise_judgments": 0,
            "peak_memory_gb": 0.0
        }
        
        logger.info(f"MultiModeEvaluator initialized for {len(model_names)} models")
    
    async def run_single_evaluation(self) -> Dict[str, Any]:
        """
        Run single model evaluation (absolute scoring).
        
        Returns:
            Single evaluation results
        """
        logger.info("Starting single model evaluation")
        results = await self.single_evaluator.run_full_evaluation()
        
        # Update progress
        self.progress["single_scores"] = sum(
            len(model_result.get("scores", []))
            for model_result in results.get("model_results", [])
        )
        
        return results
    
    async def run_pairwise_evaluation(self) -> Dict[str, Any]:
        """
        Run pairwise comparison evaluation.
        
        Returns:
            Pairwise evaluation results
        """
        logger.info("Starting pairwise comparison evaluation")
        
        if len(self.model_names) < 2:
            raise ValueError("Pairwise evaluation requires at least 2 models")
        
        # Load questions
        questions = self.data_loader.load_questions()
        if self.max_questions:
            questions = questions[:self.max_questions]
        
        # Generate responses for all models (using response manager)
        await self._generate_all_responses(questions)
        
        # Perform pairwise comparisons
        pairwise_results = await self._run_pairwise_comparisons(questions)
        
        return pairwise_results
    
    async def run_both_evaluation(self) -> Dict[str, Any]:
        """
        Run both single and pairwise evaluation.
        
        Returns:
            Combined evaluation results
        """
        logger.info("Starting combined single + pairwise evaluation")
        
        # Run single evaluation first
        single_results = await self.run_single_evaluation()
        
        # Run pairwise evaluation (will reuse cached responses)
        pairwise_results = await self.run_pairwise_evaluation()
        
        # Combine results
        combined_results = {
            "evaluation_mode": "both",
            "model_names": self.model_names,
            "single_evaluation": single_results,
            "pairwise_evaluation": pairwise_results,
            "summary_report": self._generate_combined_summary(single_results, pairwise_results)
        }
        
        return combined_results
    
    async def _generate_all_responses(self, questions: List[Dict[str, Any]]) -> None:
        """Generate responses for all models, using caching."""
        logger.info(f"Generating responses for {len(self.model_names)} models")
        
        for model_name in self.model_names:
            # Check if responses are already cached
            generation_config = self.single_evaluator.model_manager.get_generation_config()
            
            if self.response_manager.has_cached_responses(model_name, generation_config):
                logger.info(f"Using cached responses for {model_name}")
                continue
            
            # Generate new responses
            logger.info(f"Generating fresh responses for {model_name}")
            
            # Load model
            self.single_evaluator.model_manager.load_model(model_name)
            
            # Generate responses
            model_responses = {}
            for question in questions:
                question_id = str(question["question_id"])
                conversation = await self.single_evaluator.conversation_handler.run_conversation(
                    question, model_name
                )
                
                # Convert to cache format
                turns = []
                for turn_idx, turn in enumerate(conversation["turns"]):
                    turns.append({
                        "question": turn["question"],
                        "response": turn["response"],
                        "metadata": {
                            "model_name": model_name,
                            "generation_time": turn.get("generation_time", 0.0)
                        }
                    })
                
                model_responses[question_id] = turns
            
            # Cache responses
            self.response_manager.cache_responses(model_name, model_responses, generation_config)
            
            # Cleanup model
            self.single_evaluator.model_manager.cleanup()
            
            self.progress["models_completed"] += 1
    
    async def _run_pairwise_comparisons(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all pairwise comparisons between models."""
        model_pairs = list(combinations(self.model_names, 2))
        logger.info(f"Running {len(model_pairs)} pairwise comparisons")
        
        # Prepare comparison data
        comparisons = []
        generation_config = self.single_evaluator.model_manager.get_generation_config()
        
        for model_a, model_b in model_pairs:
            for question in questions:
                question_id = str(question["question_id"])
                
                # Get responses for both turns
                for turn in [1, 2]:
                    response_a = self.response_manager.get_response_for_comparison(
                        model_a, question_id, turn, generation_config
                    )
                    response_b = self.response_manager.get_response_for_comparison(
                        model_b, question_id, turn, generation_config
                    )
                    
                    if response_a and response_b:
                        turn_question = question["turns"][turn - 1]
                        comparisons.append({
                            "question": turn_question,
                            "answer_a": response_a,
                            "answer_b": response_b,
                            "question_id": question["question_id"],
                            "turn": turn,
                            "model_a": model_a,
                            "model_b": model_b
                        })
        
        # Run judgments
        judgments = await self.judge_client.judge_multiple_pairwise(comparisons)
        
        # Process results
        pairwise_results = self._process_pairwise_results(judgments, model_pairs)
        
        self.progress["pairwise_judgments"] = len(judgments)
        
        return pairwise_results
    
    def _process_pairwise_results(self, 
                                judgments: List[PairwiseJudgment], 
                                model_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Process pairwise judgment results into structured format."""
        
        # Initialize win matrix
        win_matrix = {}
        for model in self.model_names:
            win_matrix[model] = {other: {"wins": 0, "losses": 0, "ties": 0} 
                               for other in self.model_names if other != model}
        
        # Count wins/losses/ties
        for judgment in judgments:
            model_a, model_b = judgment.model_a, judgment.model_b
            
            if judgment.winner == "A":
                win_matrix[model_a][model_b]["wins"] += 1
                win_matrix[model_b][model_a]["losses"] += 1
            elif judgment.winner == "B":
                win_matrix[model_b][model_a]["wins"] += 1
                win_matrix[model_a][model_b]["losses"] += 1
            else:  # tie
                win_matrix[model_a][model_b]["ties"] += 1
                win_matrix[model_b][model_a]["ties"] += 1
        
        # Calculate win rates
        win_rates = {}
        for model in self.model_names:
            total_comparisons = 0
            total_wins = 0
            
            for opponent in self.model_names:
                if opponent != model:
                    stats = win_matrix[model][opponent]
                    total_comparisons += stats["wins"] + stats["losses"] + stats["ties"]
                    total_wins += stats["wins"]
            
            win_rates[model] = total_wins / total_comparisons if total_comparisons > 0 else 0.0
        
        # Create rankings
        ranking = sorted(self.model_names, key=lambda m: win_rates[m], reverse=True)
        
        return {
            "evaluation_mode": "pairwise",
            "model_names": self.model_names,
            "model_pairs": model_pairs,
            "win_matrix": win_matrix,
            "win_rates": win_rates,
            "ranking": ranking,
            "detailed_judgments": [
                {
                    "question_id": j.question_id,
                    "turn": j.turn,
                    "model_a": j.model_a,
                    "model_b": j.model_b,
                    "winner": j.winner,
                    "reasoning": j.reasoning
                }
                for j in judgments
            ],
            "summary_report": self._generate_pairwise_summary(win_rates, ranking, len(judgments))
        }
    
    def _generate_pairwise_summary(self, win_rates: Dict[str, float], 
                                 ranking: List[str], total_judgments: int) -> str:
        """Generate pairwise evaluation summary report."""
        lines = [
            "PAIRWISE COMPARISON RESULTS",
            "=" * 50,
            "",
            "RANKING BY WIN RATE:",
        ]
        
        for i, model in enumerate(ranking, 1):
            win_rate = win_rates[model] * 100
            lines.append(f"{i}. {model} - {win_rate:.1f}% win rate")
        
        lines.extend([
            "",
            f"Total pairwise judgments: {total_judgments}",
            f"Model pairs compared: {len(list(combinations(self.model_names, 2)))}"
        ])
        
        return "\n".join(lines)
    
    def _generate_combined_summary(self, single_results: Dict[str, Any], 
                                 pairwise_results: Dict[str, Any]) -> str:
        """Generate combined summary report."""
        lines = [
            "COMBINED EVALUATION RESULTS",
            "=" * 50,
            "",
            "ABSOLUTE SCORING (Single Model Evaluation):"
        ]
        
        # Add single evaluation summary
        if "summary_report" in single_results:
            lines.extend(single_results["summary_report"].split("\n")[2:])  # Skip header
        
        lines.extend([
            "",
            "PAIRWISE COMPARISON RESULTS:"
        ])
        
        # Add pairwise summary
        if "summary_report" in pairwise_results:
            lines.extend(pairwise_results["summary_report"].split("\n")[2:])  # Skip header
        
        return "\n".join(lines)
    
    def export_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Export evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Use single evaluator's export method if available
        if "single_evaluation" in results:
            self.single_evaluator.export_results(results["single_evaluation"], output_dir)
        elif results.get("evaluation_mode") == "single":
            self.single_evaluator.export_results(results, output_dir)
        
        # Export pairwise results if available
        if "pairwise_evaluation" in results or results.get("evaluation_mode") == "pairwise":
            self._export_pairwise_results(results, output_dir)
    
    def _export_pairwise_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Export pairwise comparison results."""
        import json
        from datetime import datetime
        
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine which results to export
        pairwise_data = results.get("pairwise_evaluation", results)
        
        # Export pairwise results as JSON
        pairwise_file = output_path / f"pairwise_results_{timestamp}.json"
        with open(pairwise_file, 'w', encoding='utf-8') as f:
            json.dump(pairwise_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pairwise results exported to {pairwise_file}")
    
    def get_evaluation_progress(self) -> Dict[str, Any]:
        """Get current evaluation progress."""
        return self.progress.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get response cache statistics."""
        return self.response_manager.get_cache_stats()