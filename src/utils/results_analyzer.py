"""Results analysis and visualization utilities for MT-bench evaluation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import logging
from dataclasses import asdict

from ..evaluation.judge_client import JudgeScore
from ..evaluation.conversation_handler import ConversationSession

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Analyzes and processes MT-bench evaluation results.
    
    Provides statistical analysis, visualization helpers, and result export functions.
    """
    
    def __init__(self):
        """Initialize results analyzer."""
        logger.info("ResultsAnalyzer initialized")
    
    def aggregate_model_results(self, scores: List[JudgeScore], 
                               sessions: List[ConversationSession],
                               questions: List['MTBenchQuestion'] = None) -> Dict[str, Any]:
        """
        Aggregate results for a single model.
        
        Args:
            scores: List of judge scores for the model
            sessions: List of conversation sessions for the model
            questions: List of original MT-bench questions (optional)
            
        Returns:
            Dictionary with aggregated model results
        """
        if not scores:
            return {"error": "No scores provided"}
        
        model_name = scores[0].model_name
        
        # Basic statistics
        all_scores = [score.score for score in scores]
        
        # Category-wise analysis
        category_stats = self._analyze_by_category(scores)
        
        # Turn-wise analysis
        turn_stats = self._analyze_by_turn(scores)
        
        # Performance metrics
        performance_stats = self._calculate_performance_metrics(sessions)
        
        return {
            "model_name": model_name,
            "total_questions": len(set(score.question_id for score in scores)),
            "total_responses": len(scores),
            "overall_score": {
                "mean": np.mean(all_scores),
                "std": np.std(all_scores),
                "median": np.median(all_scores),
                "min": np.min(all_scores),
                "max": np.max(all_scores),
                "percentile_25": np.percentile(all_scores, 25),
                "percentile_75": np.percentile(all_scores, 75)
            },
            "category_breakdown": category_stats,
            "turn_breakdown": turn_stats,
            "performance_metrics": performance_stats,
            "score_distribution": self._get_score_distribution(all_scores),
            "detailed_responses": self._extract_detailed_responses(scores, sessions, questions)
        }
    
    def _analyze_by_category(self, scores: List[JudgeScore]) -> Dict[str, Dict[str, float]]:
        """
        Analyze scores by MT-bench category.
        
        Args:
            scores: List of judge scores
            
        Returns:
            Dictionary with category-wise statistics
        """
        # Group scores by category (need to get category from question_id)
        # For now, we'll use a simple mapping based on question_id ranges
        category_mapping = self._get_category_mapping()
        
        category_scores = {}
        for score in scores:
            category = category_mapping.get(score.question_id, "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score.score)
        
        # Calculate statistics for each category
        category_stats = {}
        for category, cat_scores in category_scores.items():
            category_stats[category] = {
                "mean": np.mean(cat_scores),
                "std": np.std(cat_scores),
                "count": len(cat_scores),
                "min": np.min(cat_scores),
                "max": np.max(cat_scores)
            }
        
        return category_stats
    
    def _analyze_by_turn(self, scores: List[JudgeScore]) -> Dict[str, Dict[str, float]]:
        """
        Analyze scores by conversation turn.
        
        Args:
            scores: List of judge scores
            
        Returns:
            Dictionary with turn-wise statistics
        """
        turn_scores = {1: [], 2: []}
        
        for score in scores:
            if score.turn in turn_scores:
                turn_scores[score.turn].append(score.score)
        
        turn_stats = {}
        for turn, t_scores in turn_scores.items():
            if t_scores:
                turn_stats[f"turn_{turn}"] = {
                    "mean": np.mean(t_scores),
                    "std": np.std(t_scores),
                    "count": len(t_scores),
                    "min": np.min(t_scores),
                    "max": np.max(t_scores)
                }
        
        # Calculate turn-to-turn consistency
        if len(turn_scores[1]) > 0 and len(turn_scores[2]) > 0:
            turn_stats["consistency"] = {
                "turn1_vs_turn2_correlation": np.corrcoef(
                    turn_scores[1][:min(len(turn_scores[1]), len(turn_scores[2]))],
                    turn_scores[2][:min(len(turn_scores[1]), len(turn_scores[2]))]
                )[0, 1] if min(len(turn_scores[1]), len(turn_scores[2])) > 1 else 0.0,
                "mean_difference": np.mean(turn_scores[2]) - np.mean(turn_scores[1])
            }
        
        return turn_stats
    
    def _calculate_performance_metrics(self, sessions: List[ConversationSession]) -> Dict[str, float]:
        """
        Calculate performance metrics from conversation sessions.
        
        Args:
            sessions: List of conversation sessions
            
        Returns:
            Dictionary with performance metrics
        """
        if not sessions:
            return {}
        
        total_times = [session.total_time for session in sessions]
        peak_memories = [session.peak_memory_gb for session in sessions]
        
        # Calculate average turn times
        all_turn_times = []
        for session in sessions:
            for turn in session.turns:
                all_turn_times.append(turn.generation_time)
        
        return {
            "average_total_time": np.mean(total_times),
            "average_turn_time": np.mean(all_turn_times) if all_turn_times else 0.0,
            "peak_memory_usage_gb": np.max(peak_memories),
            "average_memory_usage_gb": np.mean(peak_memories),
            "total_conversations": len(sessions),
            "total_turns": sum(len(session.turns) for session in sessions)
        }
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """
        Get score distribution histogram.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary with score range counts
        """
        bins = [0, 2, 4, 6, 8, 10]
        labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
        
        hist, _ = np.histogram(scores, bins=bins)
        
        return dict(zip(labels, hist.tolist()))
    
    def _get_category_mapping(self) -> Dict[int, str]:
        """
        Get mapping from question ID to category.
        
        This is a simplified mapping based on typical MT-bench question ranges.
        In a real implementation, this should come from the actual dataset.
        
        Returns:
            Dictionary mapping question_id to category
        """
        # Based on typical MT-bench structure
        category_ranges = {
            "writing": range(81, 91),
            "roleplay": range(91, 101), 
            "reasoning": range(101, 111),
            "math": range(111, 121),
            "coding": range(121, 131),
            "extraction": range(131, 141),
            "stem": range(141, 151),
            "humanities": range(151, 161)
        }
        
        mapping = {}
        for category, id_range in category_ranges.items():
            for qid in id_range:
                mapping[qid] = category
        
        return mapping
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results across multiple models.
        
        Args:
            model_results: List of aggregated model results
            
        Returns:
            Dictionary with comparative analysis
        """
        if len(model_results) == 0:
            return {"error": "No models to analyze", "model_count": 0}
        
        model_names = [result["model_name"] for result in model_results]
        overall_scores = [result["overall_score"]["mean"] for result in model_results]
        
        # Handle single model case
        if len(model_results) == 1:
            return {
                "model_count": 1,
                "overall_ranking": [{"model": model_names[0], "score": overall_scores[0]}],
                "score_statistics": {
                    "highest_score": overall_scores[0],
                    "lowest_score": overall_scores[0],
                    "score_range": 0.0,
                    "average_score": overall_scores[0],
                },
                "single_model": True,
                "note": "Single model evaluation - no comparison available"
            }
        
        # Overall ranking for multiple models
        ranking = sorted(zip(model_names, overall_scores), 
                        key=lambda x: x[1], reverse=True)
        
        # Category-wise comparison
        category_comparison = self._compare_by_category(model_results)
        
        # Performance comparison
        performance_comparison = self._compare_performance(model_results)
        
        return {
            "model_count": len(model_results),
            "overall_ranking": [{"model": name, "score": score} for name, score in ranking],
            "score_statistics": {
                "highest_score": max(overall_scores),
                "lowest_score": min(overall_scores),
                "score_range": max(overall_scores) - min(overall_scores),
                "average_score": np.mean(overall_scores),
                "score_std": np.std(overall_scores)
            },
            "category_comparison": category_comparison,
            "performance_comparison": performance_comparison
        }
    
    def _compare_by_category(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare models by category performance.
        
        Args:
            model_results: List of model results
            
        Returns:
            Dictionary with category comparison
        """
        # Get all categories
        all_categories = set()
        for result in model_results:
            all_categories.update(result.get("category_breakdown", {}).keys())
        
        category_comparison = {}
        for category in all_categories:
            category_scores = {}
            for result in model_results:
                model_name = result["model_name"]
                cat_stats = result.get("category_breakdown", {}).get(category, {})
                if "mean" in cat_stats:
                    category_scores[model_name] = cat_stats["mean"]
            
            if category_scores:
                best_model = max(category_scores.items(), key=lambda x: x[1])
                category_comparison[category] = {
                    "scores": category_scores,
                    "best_model": best_model[0],
                    "best_score": best_model[1],
                    "score_range": max(category_scores.values()) - min(category_scores.values())
                }
        
        return category_comparison
    
    def _compare_performance(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare models by performance metrics.
        
        Args:
            model_results: List of model results
            
        Returns:
            Dictionary with performance comparison
        """
        performance_metrics = {}
        
        for result in model_results:
            model_name = result["model_name"]
            perf = result.get("performance_metrics", {})
            
            performance_metrics[model_name] = {
                "avg_turn_time": perf.get("average_turn_time", 0.0),
                "peak_memory_gb": perf.get("peak_memory_usage_gb", 0.0),
                "avg_memory_gb": perf.get("average_memory_usage_gb", 0.0)
            }
        
        # Find fastest and most memory efficient
        if performance_metrics:
            fastest_model = min(performance_metrics.items(), 
                              key=lambda x: x[1]["avg_turn_time"])
            most_efficient_model = min(performance_metrics.items(), 
                                     key=lambda x: x[1]["peak_memory_gb"])
            
            return {
                "metrics_by_model": performance_metrics,
                "fastest_model": {
                    "model": fastest_model[0],
                    "avg_turn_time": fastest_model[1]["avg_turn_time"]
                },
                "most_memory_efficient": {
                    "model": most_efficient_model[0],
                    "peak_memory_gb": most_efficient_model[1]["peak_memory_gb"]
                }
            }
        
        return {}
    
    def export_results(self, results: Dict[str, Any], output_path: str, 
                      format: str = "json") -> None:
        """
        Export results to file.
        
        Args:
            results: Results dictionary to export
            output_path: Path to output file
            format: Export format ('json', 'csv', or 'xlsx')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == "csv":
            # Convert to DataFrame for CSV export
            df = self._results_to_dataframe(results)
            df.to_csv(output_path, index=False)
        
        elif format.lower() == "xlsx":
            # Convert to DataFrame for Excel export
            df = self._results_to_dataframe(results)
            df.to_excel(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for CSV/Excel export.
        
        Args:
            results: Results dictionary
            
        Returns:
            pandas DataFrame
        """
        # This is a simplified version - would need more complex logic
        # for full hierarchical data flattening
        if "model_results" in results:
            rows = []
            for model_result in results["model_results"]:
                row = {
                    "model_name": model_result["model_name"],
                    "overall_score": model_result["overall_score"]["mean"],
                    "score_std": model_result["overall_score"]["std"],
                    "total_questions": model_result["total_questions"]
                }
                
                # Add category scores
                for category, stats in model_result.get("category_breakdown", {}).items():
                    row[f"category_{category}_score"] = stats.get("mean", 0.0)
                
                rows.append(row)
            
            return pd.DataFrame(rows)
        
        # Fallback: flatten the dictionary
        return pd.json_normalize(results)
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            results: Complete evaluation results
            
        Returns:
            Summary report as string
        """
        report_lines = []
        report_lines.append("MT-BENCH EVALUATION SUMMARY REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        if "comparison" in results:
            comparison = results["comparison"]
            model_count = comparison.get('model_count', 0)
            report_lines.append(f"Models Evaluated: {model_count}")
            report_lines.append("")
            
            # Handle error case
            if "error" in comparison:
                report_lines.append(f"Note: {comparison['error']}")
                report_lines.append("")
            elif model_count > 0:
                # Overall ranking
                if comparison.get("single_model"):
                    report_lines.append("SINGLE MODEL EVALUATION:")
                    report_lines.append(f"Model: {comparison['overall_ranking'][0]['model']}")
                    report_lines.append(f"Score: {comparison['overall_ranking'][0]['score']:.2f}")
                else:
                    report_lines.append("OVERALL RANKING:")
                    for i, rank in enumerate(comparison["overall_ranking"], 1):
                        report_lines.append(f"{i}. {rank['model']} - {rank['score']:.2f}")
                report_lines.append("")
            
            # Best performers by category
            if "category_comparison" in comparison:
                report_lines.append("CATEGORY LEADERS:")
                for category, cat_data in comparison["category_comparison"].items():
                    best_model = cat_data["best_model"]
                    best_score = cat_data["best_score"]
                    report_lines.append(f"{category}: {best_model} ({best_score:.2f})")
                report_lines.append("")
            
            # Performance metrics
            if "performance_comparison" in comparison:
                perf_comp = comparison["performance_comparison"]
                if "fastest_model" in perf_comp:
                    fastest = perf_comp["fastest_model"]
                    report_lines.append(f"Fastest Model: {fastest['model']} ({fastest['avg_turn_time']:.2f}s/turn)")
                
                if "most_memory_efficient" in perf_comp:
                    efficient = perf_comp["most_memory_efficient"]
                    report_lines.append(f"Most Memory Efficient: {efficient['model']} ({efficient['peak_memory_gb']:.2f}GB)")
        
        return "\n".join(report_lines)
    
    def _extract_detailed_responses(self, scores: List['JudgeScore'], sessions: List['ConversationSession'], questions: List['MTBenchQuestion'] = None) -> List[Dict[str, Any]]:
        """
        Extract detailed conversation data for display.
        
        Args:
            scores: List of judge scores
            sessions: List of conversation sessions
            questions: List of original MT-bench questions
            
        Returns:
            List of detailed response data
        """
        from ..evaluation.judge_client import JudgeScore
        from ..evaluation.conversation_handler import ConversationSession
        
        # Create a mapping from question_id to session, scores, and questions
        session_map = {session.question_id: session for session in sessions}
        question_map = {q.question_id: q for q in questions} if questions else {}
        
        # Group scores by question_id
        scores_by_question = {}
        for score in scores:
            if score.question_id not in scores_by_question:
                scores_by_question[score.question_id] = []
            scores_by_question[score.question_id].append(score)
        
        detailed_responses = []
        
        for question_id, question_scores in scores_by_question.items():
            session = session_map.get(question_id)
            original_question = question_map.get(question_id)
            if not session:
                continue
                
            # Get the original question data
            question_data = {
                "question_id": question_id,
                "question": original_question.turns[0] if original_question else f"Question {question_id}",
                "category": session.category,
                "conversation": [],
                "scores": []
            }
            
            # Add conversation turns
            for turn in session.turns:
                question_data["conversation"].append({
                    "role": "assistant",
                    "content": turn.assistant_response,
                    "turn_number": turn.turn_number,
                    "user_message": turn.user_message
                })
            
            # Add scores and reasoning
            for score in question_scores:
                question_data["scores"].append({
                    "turn": score.turn,
                    "score": score.score,
                    "reasoning": score.reasoning
                })
            
            detailed_responses.append(question_data)
        
        return detailed_responses