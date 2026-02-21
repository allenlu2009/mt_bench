"""Argument parser construction for benchmark CLI."""

import argparse
from typing import List

EPILOG = """
Examples:
  # Default project remains mt-bench
  python -m src.cli --models gpt2-large

  # New subcommand style (preferred)
  python -m src.cli mt-bench --models gpt2-large
  python -m src.cli perplexity --models qwen2.5-3b --perplexity-datasets wikitext2

  # Backward-compatible style still works
  python -m src.cli --project mt-bench --models gpt2-large
  python -m src.cli --project perplexity --models qwen2.5-3b
"""


def normalize_cli_argv(argv: List[str]) -> List[str]:
    """
    Normalize legacy CLI invocations to subcommand form.

    Compatibility behavior:
    - If no subcommand is provided, defaults to `mt-bench`.
    - If `--project <name>` is provided, inserts that subcommand and removes the flag pair.
    """
    if not argv:
        return ["mt-bench"]
    if len(argv) == 1 and argv[0] in {"-h", "--help"}:
        return argv

    if argv[0] in {"mt-bench", "perplexity"}:
        return argv

    project = "mt-bench"
    normalized: List[str] = []
    skip_next = False
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--project":
            if i + 1 < len(argv):
                project = argv[i + 1]
                skip_next = True
            continue
        normalized.append(token)

    return [project] + normalized


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", help="Model names to evaluate")
    parser.add_argument("--family", help="Evaluate all models in a specific family")
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")
    parser.add_argument("--memory-limit", type=float, default=8.0)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--force-download", action="store_true")


def _add_mtbench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["single", "pairwise", "both"], default="single")
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
    parser.add_argument("--turn1", action="store_true")
    parser.add_argument("--response-cache-dir", default="cached_responses")
    parser.add_argument("--disable-response-cache", action="store_true")
    parser.add_argument("--debug-judge", action="store_true")
    parser.add_argument("--low-score", type=float, default=2.0)


def _add_perplexity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["single"], default="single")
    parser.add_argument("--perplexity-datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--stride-ratios", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--use-all-tokens", action="store_true")
    parser.add_argument("--no-residue", action="store_true")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser with project subcommands."""
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    subparsers = parser.add_subparsers(dest="project", metavar="project")

    mtbench_parser = subparsers.add_parser("mt-bench", help="Run MT-bench evaluation")
    _add_common_args(mtbench_parser)
    _add_mtbench_args(mtbench_parser)
    mtbench_parser.set_defaults(project="mt-bench")

    perplexity_parser = subparsers.add_parser("perplexity", help="Run perplexity evaluation")
    _add_common_args(perplexity_parser)
    _add_perplexity_args(perplexity_parser)
    perplexity_parser.set_defaults(project="perplexity")

    return parser
