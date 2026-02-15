"""Argument parser construction for benchmark CLI."""

import argparse

EPILOG = """
Examples:
  # Default project is mt-bench (same as --project mt-bench)
  python -m src.cli --models gpt2-large
  
  # Explicitly select project
  python -m src.cli --project mt-bench --models gpt2-large
  python -m src.cli --project perplexity --models gpt2-large
  
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
  
  # Use different judge models
  python -m src.cli --models gpt2-large --judge-model gpt-5-mini
  python -m src.cli --models gpt2-large --judge-model gpt-4o-mini
  python -m src.cli --models gpt2-large --judge-model gpt-4.1-nano
  
  # Adjust memory limit for different GPUs
  python -m src.cli --models llama-3.2-3b --memory-limit 8.0
  
  # Evaluate only first turn (single Q&A) for faster testing
  python -m src.cli --models gemma3-270m --turn1 --max-questions 5
  
  # Evaluate all models in a family
  python -m src.cli --family gemma3 --max-questions 2
  python -m src.cli --family llama --turn1
  
  # List available models
  python -m src.cli --list-models
"""


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation System - MT-bench (default) with future-ready project routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    parser.add_argument("--project", choices=["mt-bench", "perplexity"], default="mt-bench")
    parser.add_argument("--models", nargs="+", help="Model names to evaluate")
    parser.add_argument("--family", help="Evaluate all models in a specific family")
    parser.add_argument("--mode", choices=["single", "pairwise", "both"], default="single")
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")
    parser.add_argument("--openai-api-key", help="OpenAI API key for judge")
    parser.add_argument(
        "--judge-model",
        choices=[
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4.1-nano",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
        ],
        default="gpt-5-nano",
    )
    parser.add_argument("--memory-limit", type=float, default=8.0)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--turn1", action="store_true")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--response-cache-dir", default="cached_responses")
    parser.add_argument("--perplexity-datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--stride-ratios", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--use-all-tokens", action="store_true")
    parser.add_argument("--no-residue", action="store_true")
    parser.add_argument("--disable-response-cache", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug-judge", action="store_true")
    parser.add_argument("--low-score", type=float, default=2.0)
    parser.add_argument("--force-download", action="store_true")
    return parser

