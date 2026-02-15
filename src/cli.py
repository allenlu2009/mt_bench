"""Command-line interface entrypoint for benchmark evaluation."""

import asyncio
import logging
import sys
from pathlib import Path

from .cli_parts.common import (
    print_available_models,
    require_mtbench_api_key,
    resolve_model_names,
    setup_logging,
    validate_models,
)
from .cli_parts.output import print_completion_summary, print_qa_examples
from .cli_parts.parser import create_parser
from .evaluation.mtbench_evaluator import MTBenchEvaluator
from .evaluation.multi_mode_evaluator import MultiModeEvaluator
from .evaluation.perplexity_evaluator import PerplexityEvaluator
from .utils.memory_utils import optimize_for_rtx3060


async def _run_mtbench(args, valid_models, api_key, logger):
    if args.mode == "single":
        evaluator = MTBenchEvaluator(
            model_names=valid_models,
            openai_api_key=api_key,
            judge_model=args.judge_model,
            cache_dir=args.cache_dir,
            memory_limit_gb=args.memory_limit,
            max_questions=args.max_questions,
            debug_judge=args.debug_judge,
            low_score_threshold=args.low_score,
            turn1_only=args.turn1,
        )
        logger.info("Starting single-mode MT-bench evaluation for %s models", len(valid_models))
        results = await evaluator.run_full_evaluation()
        return evaluator, results

    evaluator = MultiModeEvaluator(
        model_names=valid_models,
        openai_api_key=api_key,
        judge_model=args.judge_model,
        cache_dir=args.cache_dir,
        response_cache_dir=args.response_cache_dir,
        memory_limit_gb=args.memory_limit,
        max_questions=args.max_questions,
        disable_response_cache=args.disable_response_cache,
        debug_judge=args.debug_judge,
        low_score_threshold=args.low_score,
    )
    logger.info("Starting %s-mode MT-bench evaluation for %s models", args.mode, len(valid_models))
    if args.max_questions:
        logger.info("Limited to %s questions for testing", args.max_questions)

    if args.mode == "pairwise":
        results = await evaluator.run_pairwise_evaluation()
    elif args.mode == "both":
        results = await evaluator.run_both_evaluation()
    else:
        raise ValueError(f"Unknown evaluation mode: {args.mode}")
    return evaluator, results


async def _run_perplexity(args, valid_models, logger):
    if args.mode != "single":
        raise ValueError("Perplexity project currently supports only --mode single")

    evaluator = PerplexityEvaluator(
        model_names=valid_models,
        cache_dir=args.cache_dir,
        memory_limit_gb=args.memory_limit,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        datasets=args.perplexity_datasets,
        block_size=args.block_size,
        stride_ratios=args.stride_ratios,
        device=args.device,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        use_all_tokens=args.use_all_tokens,
        handle_residue=not args.no_residue,
    )
    logger.info("Starting perplexity evaluation for %s models", len(valid_models))
    results = await evaluator.run_full_evaluation()
    return evaluator, results


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.list_models:
        print_available_models()
        return

    model_names = resolve_model_names(args, parser)
    if args.mode in ["pairwise", "both"] and len(model_names) < 2:
        parser.error("Pairwise comparison requires at least 2 models")

    api_key = require_mtbench_api_key(args, parser)

    try:
        optimize_for_rtx3060()
        logger.info("Applied RTX 3060 optimizations")
        valid_models = validate_models(model_names, args.memory_limit)
        logger.info("Validated models: %s", valid_models)
        Path(args.output_dir).mkdir(exist_ok=True)

        if args.project == "mt-bench":
            evaluator, results = await _run_mtbench(args, valid_models, api_key, logger)
        else:
            try:
                evaluator, results = await _run_perplexity(args, valid_models, logger)
            except ValueError as e:
                parser.error(str(e))
                return

        evaluator.export_results(results, args.output_dir)
        progress = evaluator.get_evaluation_progress()
        print_completion_summary(results, progress, args.output_dir)

        if args.project == "mt-bench":
            print_qa_examples(results, args.max_questions or 80)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e))
        print(f"\nError: {str(e)}")
        sys.exit(1)


def cli_entry_point() -> None:
    """Entry point for setuptools console script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())

