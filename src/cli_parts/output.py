"""CLI output helpers for summaries and MT-bench detailed examples."""

from typing import Any, Dict


def show_qa_detail(response_data: Dict[str, Any], judge_model: str = "gpt-5-mini") -> None:
    """Show detailed Q&A for a single question."""
    question_text = response_data.get("question", "Question text not available")
    print(f"\n🔸 Question: {question_text}")
    print(f"🔸 Category: {response_data.get('category', 'Unknown')}")

    conversation = response_data.get("conversation", [])
    if conversation:
        print("\n💬 Conversation:")
        for i, turn in enumerate(conversation, 1):
            user_msg = turn.get("user_message", "No user message")
            assistant_msg = turn.get("content", "No response")
            if len(user_msg) > 200:
                user_msg = user_msg[:200] + "..."
            if len(assistant_msg) > 300:
                assistant_msg = assistant_msg[:300] + "..."
            print(f"  Turn {i}:")
            print(f"    User: {user_msg}")
            print(f"    Assistant: {assistant_msg}")

    scores = response_data.get("scores", [])
    if scores:
        print(f"\n⚖️  {judge_model} Judgments:")
        for i, score_data in enumerate(scores, 1):
            score = score_data.get("score", "N/A")
            reasoning = score_data.get("reasoning", "No reasoning provided")
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            print(f"  Turn {i} Score: {score}/10")
            print(f"  Reasoning: {reasoning}")
    print()


def print_qa_examples(results: Dict[str, Any], max_questions: int) -> None:
    """Print examples of questions, answers, and judgments for MT-bench."""
    if "model_results" not in results:
        return

    print("\n" + "=" * 80)
    print("DETAILED Q&A EXAMPLES")
    print("=" * 80)
    judge_model = results.get("metadata", {}).get("judge_model", "gpt-5-mini")

    for model_result in results["model_results"]:
        model_name = model_result["model_name"]
        print(f"\nModel: {model_name}")
        print("-" * 50)
        responses = model_result.get("detailed_responses", [])
        if not responses:
            print("No detailed responses available")
            continue

        if max_questions == 1:
            show_qa_detail(responses[0], judge_model)
            continue

        print(f"\n📝 FIRST QUESTION (Q{responses[0].get('question_id', '?')}):")
        show_qa_detail(responses[0], judge_model)

        if len(responses) >= 2:
            print(f"\n📝 LAST QUESTION (Q{responses[-1].get('question_id', '?')}):")
            show_qa_detail(responses[-1], judge_model)


def print_completion_summary(results: Dict[str, Any], progress: Dict[str, Any], output_dir: str) -> None:
    """Print standard evaluation completion summary."""
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    if "summary_report" in results:
        print(results["summary_report"])

    print(f"\nResults saved to: {output_dir}")
    print("\nEvaluation Statistics:")
    print(f"  Models evaluated: {progress['models_completed']}/{progress['total_models']}")

    if "total_scores" in progress:
        print(f"  Total responses judged: {progress['total_scores']}")
    elif "total_tokens_evaluated" in progress:
        print(f"  Total tokens evaluated: {progress['total_tokens_evaluated']}")
    else:
        total_judgments = progress.get("single_scores", 0) + progress.get("pairwise_judgments", 0)
        print(f"  Total judgments made: {total_judgments}")
        if progress.get("pairwise_judgments", 0) > 0:
            print(f"    - Pairwise comparisons: {progress['pairwise_judgments']}")
        if progress.get("single_scores", 0) > 0:
            print(f"    - Single evaluations: {progress['single_scores']}")

    print(f"  Peak memory usage: {progress['peak_memory_gb']:.2f}GB")

