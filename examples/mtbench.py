"""
Complete MT-bench evaluation system for phi models.
"""

import json
import time
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from phi_evaluator import PhiEvaluator
from judge import MTBenchJudge
from mtbench_data import MTBENCH_QUESTIONS, get_all_categories
from models import MTBenchResponse, MTBenchResult, PairwiseComparison, EvaluationSummary
from config import Config
from memory_utils import aggressive_memory_cleanup, print_memory_status


class MTBenchEvaluator:
    """
    Complete MT-bench evaluation system.
    """
    
    def __init__(self, judge_model: str = None):
        """Initialize the MT-bench evaluator."""
        self.judge = MTBenchJudge(judge_model=judge_model)
        self.results: Dict[str, List[MTBenchResult]] = {}
        self.pairwise_results: List[PairwiseComparison] = []
        self.lock = threading.Lock()
        
    def generate_model_responses(self, model_name: str, questions_to_use: List = None) -> List[MTBenchResponse]:
        """
        Generate responses for MT-bench questions using a phi model.
        
        Args:
            model_name: Name of the phi model to evaluate
            questions_to_use: List of questions to evaluate (defaults to all MTBENCH_QUESTIONS)
            
        Returns:
            List of MTBenchResponse objects
        """
        if questions_to_use is None:
            questions_to_use = MTBENCH_QUESTIONS
            
        print(f"Generating MT-bench responses for {model_name}")
        print("-" * 50)
        
        model_config = Config.PHI_MODELS[model_name]
        responses = []
        
        with PhiEvaluator(model_config) as evaluator:
            evaluator.load_model()
            
            for question in questions_to_use:
                print(f"Processing question {question.question_id} ({question.category})")
                
                question_responses = []
                total_time = 0
                
                # Generate response for each turn with proper conversation context
                for i, turn in enumerate(question.turns):
                    # Build conversation messages up to current turn
                    conversation_messages = []
                    
                    # Add all previous turns
                    for j in range(i):
                        conversation_messages.append({"role": "user", "content": question.turns[j]})
                        conversation_messages.append({"role": "assistant", "content": question_responses[j]})
                    
                    # Add current turn
                    conversation_messages.append({"role": "user", "content": turn})
                    
                    try:
                        # Generate response using multi-turn conversation method
                        response, gen_time = evaluator.generate_response_multiturn(conversation_messages)
                        question_responses.append(response)
                        total_time += gen_time
                        
                        print(f"  Turn {i+1}: Generated in {gen_time:.2f}s")
                        
                    except Exception as e:
                        print(f"  Error in turn {i+1}: {e}")
                        question_responses.append(f"Error: {str(e)}")
                        
                # Create response object
                mtbench_response = MTBenchResponse(
                    question_id=question.question_id,
                    model_name=model_name,
                    turns=question_responses,
                    generation_time=total_time
                )
                
                responses.append(mtbench_response)
                print(f"  Total time: {total_time:.2f}s")
                
                # Print detailed question and response information for debugging
                print(f"\n📝 Question {question.question_id} Details:")
                print(f"   Category: {question.category}")
                print(f"   Total Turns: {len(question.turns)}")
                print("   " + "="*80)
                
                # Print all turns (questions and responses) with full details
                for i, turn_question in enumerate(question.turns, 1):
                    print(f"\n   🔵 TURN {i} QUESTION:")
                    print(f"   {turn_question}")
                    
                    if i <= len(question_responses):
                        response = question_responses[i-1]
                        print(f"\n   🟢 TURN {i} RESPONSE:")
                        print(f"   {response}")
                        print(f"   📏 Response Length: {len(response)} characters")
                    else:
                        print(f"\n   ❌ TURN {i} RESPONSE: NOT GENERATED")
                    print("   " + "-"*60)
                
                print()
                
        return responses
        
    def evaluate_individual_model(self, model_name: str, responses: List[MTBenchResponse]) -> List[MTBenchResult]:
        """
        Evaluate individual model responses using the judge.
        
        Args:
            model_name: Name of the model
            responses: List of model responses
            
        Returns:
            List of MTBenchResult objects
        """
        print(f"Evaluating {model_name} responses with judge")
        print("-" * 50)
        
        results = []
        
        for response in responses:
            # Get the original question
            question = next(q for q in MTBENCH_QUESTIONS if q.question_id == response.question_id)
            
            print(f"Judging question {response.question_id} ({question.category})")
            
            # Judge the response
            result = self.judge.evaluate_mtbench_result(
                question=question.turns[0],
                responses=response.turns,
                model_name=model_name,
                question_id=response.question_id,
                category=question.category,
                questions=question.turns
            )
            
            results.append(result)
            
            print(f"  Turn 1 score: {result.turn_1_score.score:.1f}")
            if result.turn_2_score:
                print(f"  Turn 2 score: {result.turn_2_score.score:.1f}")
            print(f"  Average: {result.average_score:.1f}")
            print()
            
        return results
        
    def evaluate_pairwise_comparison(self, model_a: str, responses_a: List[MTBenchResponse],
                                   model_b: str, responses_b: List[MTBenchResponse]) -> List[PairwiseComparison]:
        """
        Perform pairwise comparison between two models.
        
        Args:
            model_a: Name of first model
            responses_a: Responses from first model
            model_b: Name of second model
            responses_b: Responses from second model
            
        Returns:
            List of PairwiseComparison objects
        """
        print(f"Pairwise comparison: {model_a} vs {model_b}")
        print("-" * 50)
        
        comparisons = []
        
        for resp_a, resp_b in zip(responses_a, responses_b):
            assert resp_a.question_id == resp_b.question_id, "Question IDs must match"
            
            # Get the original question
            question = next(q for q in MTBENCH_QUESTIONS if q.question_id == resp_a.question_id)
            
            print(f"Comparing question {resp_a.question_id} ({question.category})")
            
            # Perform pairwise comparison
            comparison = self.judge.compare_models_pairwise(
                question=question.turns[0],
                responses_a=resp_a.turns,
                responses_b=resp_b.turns,
                model_a=model_a,
                model_b=model_b,
                question_id=resp_a.question_id,
                category=question.category
            )
            
            comparisons.append(comparison)
            
            print(f"  Winner: {comparison.winner}")
            print()
            
        return comparisons
        
    def run_complete_evaluation(self, max_questions: int = None, selected_models: dict = None) -> EvaluationSummary:
        """
        Run complete MT-bench evaluation for selected models.
        
        Args:
            max_questions: Limit number of questions for debugging (None for all questions)
            selected_models: Dictionary of models to evaluate (defaults to all models)
        
        Returns:
            EvaluationSummary with all results
        """
        # Select questions to evaluate
        questions_to_use = MTBENCH_QUESTIONS
        if max_questions:
            questions_to_use = MTBENCH_QUESTIONS[:max_questions]
            print("=== Starting MT-bench Evaluation (DEBUG MODE) ===")
            print(f"Limited to {max_questions} question(s) for debugging")
        else:
            print("=== Starting Complete MT-bench Evaluation ===")
            
        # Use selected models or default to all models
        models_to_evaluate = selected_models if selected_models is not None else Config.PHI_MODELS
        
        print(f"Timestamp: {datetime.now()}")
        print(f"Models to evaluate: {list(models_to_evaluate.keys())}")
        print(f"Questions to evaluate: {len(questions_to_use)}")
        print(f"Categories: {get_all_categories()}")
        print()
        
        # Step 1: Generate responses for all models (sequential loading for memory management)
        all_responses = {}
        model_names = list(models_to_evaluate.keys())
        
        for i, model_name in enumerate(model_names):
            print(f"Processing model {i+1}/{len(model_names)}: {model_name}")
            print("⚠️  Loading model sequentially to avoid OOM on RTX 3060")
            responses = self.generate_model_responses(model_name, questions_to_use)
            all_responses[model_name] = responses
            
            # Force aggressive memory cleanup between models (except after last model)
            if i < len(model_names) - 1:  # Don't clean after last model
                print(f"🧹 Performing aggressive memory cleanup after {model_name}...")
                aggressive_memory_cleanup(model_name)
                
                # Additional cleanup verification
                import torch
                if torch.cuda.is_available():
                    print_memory_status(f"Memory Status After {model_name} Cleanup")
            print()
            
        # Step 2: Individual evaluation
        for model_name, responses in all_responses.items():
            results = self.evaluate_individual_model(model_name, responses)
            self.results[model_name] = results
            
        # Step 3: Pairwise comparison
        model_names = list(models_to_evaluate.keys())
        if len(model_names) == 2:
            model_a, model_b = model_names
            comparisons = self.evaluate_pairwise_comparison(
                model_a, all_responses[model_a],
                model_b, all_responses[model_b]
            )
            self.pairwise_results = comparisons
            
        # Step 4: Generate summary
        summary = self._generate_summary()
        
        print("=== Evaluation Complete ===")
        self._print_summary(summary)
        
        return summary
        
    def _generate_summary(self) -> EvaluationSummary:
        """
        Generate evaluation summary from results.
        
        Returns:
            EvaluationSummary object
        """
        # Calculate individual average scores
        individual_scores = {}
        for model_name, results in self.results.items():
            if results:
                avg_score = sum(r.average_score for r in results) / len(results)
                individual_scores[model_name] = avg_score
                
        # Calculate pairwise win/loss/tie counts
        pairwise_results = {}
        if self.pairwise_results:
            model_names = list(self.results.keys())  # Use models that were actually evaluated
            for model in model_names:
                pairwise_results[model] = {"wins": 0, "losses": 0, "ties": 0}
                
            for comparison in self.pairwise_results:
                if comparison.winner == comparison.model_a:
                    pairwise_results[comparison.model_a]["wins"] += 1
                    pairwise_results[comparison.model_b]["losses"] += 1
                elif comparison.winner == comparison.model_b:
                    pairwise_results[comparison.model_b]["wins"] += 1
                    pairwise_results[comparison.model_a]["losses"] += 1
                else:  # tie
                    pairwise_results[comparison.model_a]["ties"] += 1
                    pairwise_results[comparison.model_b]["ties"] += 1
                    
        return EvaluationSummary(
            models_evaluated=list(Config.PHI_MODELS.keys()),
            total_questions=len([r for results in self.results.values() for r in results]),
            categories=list(set(r.category for results in self.results.values() for r in results)),
            individual_scores=individual_scores,
            pairwise_results=pairwise_results,
            judge_model=self.judge.judge_model
        )
        
    def _print_summary(self, summary: EvaluationSummary) -> None:
        """
        Print formatted evaluation summary.
        
        Args:
            summary: EvaluationSummary to print
        """
        print("\n" + "="*60)
        print("MT-BENCH EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Evaluation Date: {summary.timestamp}")
        print(f"Judge Model: {summary.judge_model}")
        print(f"Total Questions: {summary.total_questions}")
        print(f"Categories: {', '.join(summary.categories)}")
        print()
        
        print("INDIVIDUAL MODEL SCORES:")
        print("-" * 30)
        for model, score in summary.individual_scores.items():
            print(f"{model}: {score:.2f}/10.0")
        print()
        
        if summary.pairwise_results:
            print("PAIRWISE COMPARISON RESULTS:")
            print("-" * 30)
            for model, results in summary.pairwise_results.items():
                wins = results["wins"]
                losses = results["losses"] 
                ties = results["ties"]
                total = wins + losses + ties
                win_rate = (wins / total * 100) if total > 0 else 0
                print(f"{model}: {wins}W-{losses}L-{ties}T ({win_rate:.1f}% win rate)")
        
        print("\nDetailed results available in self.results and self.pairwise_results")


def run_mtbench_evaluation(max_questions: int = None, selected_models: dict = None, judge_model: str = None):
    """
    Main function to run MT-bench evaluation.
    
    Args:
        max_questions: Limit number of questions for debugging (None for all questions)
        selected_models: Dictionary of models to evaluate (defaults to all models)
        judge_model: Judge model to use for evaluation
    """
    # Validate configuration
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and ensure all required variables are set.")
        return None
        
    # Run evaluation
    evaluator = MTBenchEvaluator(judge_model=judge_model)
    summary = evaluator.run_complete_evaluation(max_questions=max_questions, selected_models=selected_models)
    
    return evaluator, summary


if __name__ == "__main__":
    run_mtbench_evaluation()