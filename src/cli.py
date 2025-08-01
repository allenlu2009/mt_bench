"""Command-line interface for MT-bench evaluation system."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .evaluation.mtbench_evaluator import MTBenchEvaluator
from .evaluation.multi_mode_evaluator import MultiModeEvaluator
from .models.model_configs import get_available_models, get_models_within_memory_limit
from .utils.memory_utils import optimize_for_rtx3060


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mtbench_evaluation.log')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


def validate_models(model_names: List[str], memory_limit_gb: float) -> List[str]:
    """
    Validate and filter model names.
    
    Args:
        model_names: List of model names to validate
        memory_limit_gb: Memory limit for filtering
        
    Returns:
        List of valid model names
        
    Raises:
        ValueError: If no valid models found
    """
    available_models = get_available_models()
    memory_compatible_models = get_models_within_memory_limit(memory_limit_gb)
    
    valid_models = []
    for model_name in model_names:
        if model_name not in available_models:
            print(f"Warning: Model '{model_name}' not available. Available models:")
            for name in available_models.keys():
                print(f"  - {name}")
            continue
        
        if model_name not in memory_compatible_models:
            config = available_models[model_name]
            print(f"Warning: Model '{model_name}' requires {config.estimated_memory_gb:.1f}GB "
                  f"but limit is {memory_limit_gb:.1f}GB. Skipping.")
            continue
        
        valid_models.append(model_name)
    
    if not valid_models:
        raise ValueError("No valid models found after filtering")
    
    return valid_models


def print_available_models() -> None:
    """Print all available models with their specifications."""
    models = get_available_models()
    
    print("Available Models:")
    print("-" * 80)
    print(f"{'Model Name':<20} {'Family':<10} {'Memory (GB)':<12} {'Model Path'}")
    print("-" * 80)
    
    for name, config in models.items():
        print(f"{name:<20} {config.model_family:<10} {config.estimated_memory_gb:<12.1f} {config.model_path}")
    
    print(f"\nTotal models: {len(models)}")


def print_qa_examples(results: dict, max_questions: int) -> None:
    """
    Print examples of questions, answers, and judgments.
    
    Args:
        results: Complete evaluation results
        max_questions: Maximum number of questions evaluated
    """
    if "model_results" not in results:
        return
        
    print("\n" + "="*80)
    print("DETAILED Q&A EXAMPLES")
    print("="*80)
    
    for model_result in results["model_results"]:
        model_name = model_result["model_name"] 
        print(f"\nModel: {model_name}")
        print("-" * 50)
        
        if "detailed_responses" not in model_result:
            print("No detailed responses available")
            continue
            
        responses = model_result["detailed_responses"]
        
        if max_questions == 1:
            # Show only the first question
            if responses:
                show_qa_detail(responses[0], 1)
        else:
            # Show first and last questions
            if len(responses) >= 1:
                print(f"\n📝 FIRST QUESTION (Q{responses[0].get('question_id', '?')}):")
                show_qa_detail(responses[0], 1)
                
            if len(responses) >= 2:
                print(f"\n📝 LAST QUESTION (Q{responses[-1].get('question_id', '?')}):")
                show_qa_detail(responses[-1], len(responses))


def show_qa_detail(response_data: dict, question_num: int) -> None:
    """Show detailed Q&A for a single question."""
    
    # Question
    question_text = response_data.get('question', 'Question text not available')
    print(f"\n🔸 Question: {question_text}")
    
    # Category
    category = response_data.get('category', 'Unknown')
    print(f"🔸 Category: {category}")
    
    # Model responses
    conversation = response_data.get('conversation', [])
    if conversation:
        print("\n💬 Conversation:")
        for i, turn in enumerate(conversation, 1):
            user_msg = turn.get('user_message', 'No user message')
            assistant_msg = turn.get('content', 'No response')
            
            # Truncate very long messages
            if len(user_msg) > 200:
                user_msg = user_msg[:200] + "..."
            if len(assistant_msg) > 300:
                assistant_msg = assistant_msg[:300] + "..."
                
            print(f"  Turn {i}:")
            print(f"    User: {user_msg}")
            print(f"    Assistant: {assistant_msg}")
    
    # Judgments
    scores = response_data.get('scores', [])
    if scores:
        print("\n⚖️  GPT-4.1-nano Judgments:")
        for i, score_data in enumerate(scores, 1):
            score = score_data.get('score', 'N/A')
            reasoning = score_data.get('reasoning', 'No reasoning provided')
            # Truncate long reasoning
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            print(f"  Turn {i} Score: {score}/10")
            print(f"  Reasoning: {reasoning}")
    
    print()


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="MT-bench Evaluation System - Evaluate language models using MT-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model (absolute scoring)
  python -m src.cli --models gpt2-large
  
  # Evaluate multiple models (absolute scoring)
  python -m src.cli --models gpt2-large llama-3.2-1b phi-3-mini
  
  # Pairwise comparison between two models
  python -m src.cli --mode pairwise --models gpt2-large llama-3.2-1b
  
  # Both absolute scoring and pairwise comparison
  python -m src.cli --mode both --models gpt2-large llama-3.2-1b phi-3-mini
  
  # Quick test with limited questions
  python -m src.cli --models gpt2-large --max-questions 5
  
  # Adjust memory limit for different GPUs
  python -m src.cli --models llama-3.2-3b --memory-limit 8.0
  
  # List available models
  python -m src.cli --list-models
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to evaluate (use --list-models to see available models)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "pairwise", "both"],
        default="single",
        help="Evaluation mode: single (absolute scoring), pairwise (head-to-head comparison), or both (default: single)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for judge (uses OPENAI_API_KEY env var if not provided)"
    )
    
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-nano",
        help="Judge model to use (default: gpt-4.1-nano)"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=6.0,
        help="GPU memory limit in GB (default: 6.0 for RTX 3060)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="data",
        help="Directory for caching data (default: data)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results)"
    )
    
    parser.add_argument(
        "--response-cache-dir",
        default="cached_responses",
        help="Directory for caching model responses (default: cached_responses)"
    )
    
    parser.add_argument(
        "--disable-response-cache",
        action="store_true",
        help="Disable response caching (always generate fresh responses)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of MT-bench data"
    )
    
    return parser


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # List models and exit if requested
    if args.list_models:
        print_available_models()
        return
    
    # Validate required arguments
    if not args.models:
        parser.error("--models is required (use --list-models to see available models)")
    
    # Validate evaluation mode requirements
    if args.mode in ["pairwise", "both"] and len(args.models) < 2:
        parser.error("Pairwise comparison requires at least 2 models")
    
    # Check for API key
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-api-key")
    
    try:
        # Apply GPU optimizations
        optimize_for_rtx3060()
        logger.info("Applied RTX 3060 optimizations")
        
        # Validate models
        valid_models = validate_models(args.models, args.memory_limit)
        logger.info(f"Validated models: {valid_models}")
        
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Initialize appropriate evaluator based on mode
        if args.mode == "single":
            # Use original single-mode evaluator
            evaluator = MTBenchEvaluator(
                model_names=valid_models,
                openai_api_key=api_key,
                judge_model=args.judge_model,
                cache_dir=args.cache_dir,
                memory_limit_gb=args.memory_limit,
                max_questions=args.max_questions
            )
            
            logger.info(f"Starting single-mode MT-bench evaluation for {len(valid_models)} models")
            results = await evaluator.run_full_evaluation()
            
        else:
            # Use multi-mode evaluator for pairwise and both modes
            evaluator = MultiModeEvaluator(
                model_names=valid_models,
                openai_api_key=api_key,
                judge_model=args.judge_model,
                cache_dir=args.cache_dir,
                response_cache_dir=args.response_cache_dir,
                memory_limit_gb=args.memory_limit,
                max_questions=args.max_questions,
                disable_response_cache=args.disable_response_cache
            )
            
            logger.info(f"Starting {args.mode}-mode MT-bench evaluation for {len(valid_models)} models")
            
            if args.max_questions:
                logger.info(f"Limited to {args.max_questions} questions for testing")
            
            # Run appropriate evaluation mode
            if args.mode == "pairwise":
                results = await evaluator.run_pairwise_evaluation()
            elif args.mode == "both":
                results = await evaluator.run_both_evaluation()
            else:
                raise ValueError(f"Unknown evaluation mode: {args.mode}")
        
        # Export results
        evaluator.export_results(results, args.output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        if "summary_report" in results:
            print(results["summary_report"])
        
        print(f"\nResults saved to: {args.output_dir}")
        
        # Print progress stats
        progress = evaluator.get_evaluation_progress()
        print(f"\nEvaluation Statistics:")
        print(f"  Models evaluated: {progress['models_completed']}/{progress['total_models']}")
        print(f"  Total responses judged: {progress['total_scores']}")
        print(f"  Peak memory usage: {progress['peak_memory_gb']:.2f}GB")
        
        # Print detailed Q&A examples
        print_qa_examples(results, args.max_questions or 80)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\nEvaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


def cli_entry_point() -> None:
    """Entry point for setuptools console script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())