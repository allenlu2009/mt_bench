"""Main MT-bench evaluator integrating all components."""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from tqdm.asyncio import tqdm

from ..models.model_manager import ModelManager
from ..models.model_configs import get_model_config, format_prompt_for_model
from ..utils.data_loader import DataLoader, MTBenchQuestion
from ..utils.memory_utils import MemoryMonitor
from ..utils.results_analyzer import ResultsAnalyzer
from .judge_client import JudgeClient, JudgeScore
from .conversation_handler import ConversationHandler, ConversationSession

logger = logging.getLogger(__name__)


class MTBenchEvaluator:
    """
    Main MT-bench evaluator class.
    
    Orchestrates the complete evaluation pipeline:
    1. Load models with memory optimization
    2. Generate responses for MT-bench questions
    3. Judge responses using GPT-4.1-nano
    4. Analyze and export results
    """
    
    def __init__(self, 
                 model_names: List[str],
                 openai_api_key: Optional[str] = None,
                 judge_model: str = "gpt-5-nano",
                 cache_dir: str = "data",
                 memory_limit_gb: float = 6.0,
                 max_questions: Optional[int] = None,
                 debug_judge: bool = False,
                 turn1_only: bool = False):
        """
        Initialize MT-bench evaluator.
        
        Args:
            model_names: List of model names to evaluate
            openai_api_key: OpenAI API key for judge
            judge_model: Judge model to use
            cache_dir: Directory for data caching
            memory_limit_gb: GPU memory limit
            max_questions: Limit number of questions (for testing)
            debug_judge: Enable debug output for judge prompts and responses
            turn1_only: If True, evaluate only the first turn (single Q&A)
        """
        self.model_names = model_names
        self.max_questions = max_questions
        self.turn1_only = turn1_only
        
        # Initialize components
        self.model_manager = ModelManager(memory_limit_gb=memory_limit_gb)
        self.data_loader = DataLoader(cache_dir=cache_dir)
        self.judge_client = JudgeClient(api_key=openai_api_key, model=judge_model, debug=debug_judge)
        self.conversation_handler = ConversationHandler()
        self.results_analyzer = ResultsAnalyzer()
        self.memory_monitor = MemoryMonitor(memory_limit_gb)
        
        # Results storage
        self.all_scores: List[JudgeScore] = []
        self.all_sessions: List[ConversationSession] = []
        self.evaluation_metadata = {
            "start_time": None,
            "end_time": None,
            "models_evaluated": model_names,
            "judge_model": judge_model,
            "total_questions": 0,
            "memory_limit_gb": memory_limit_gb,
            "turn1_only": turn1_only
        }
        
        logger.info(f"MTBenchEvaluator initialized for models: {model_names}")
    
    def _select_questions_by_category(self, questions: List[MTBenchQuestion], 
                                    max_questions: int) -> List[MTBenchQuestion]:
        """
        Select questions prioritizing one per category for balanced evaluation.
        
        Args:
            questions: All available questions
            max_questions: Maximum number of questions to select
            
        Returns:
            Selected questions with balanced category representation
        """
        if max_questions >= len(questions):
            return questions
        
        # Group questions by category
        categories = {}
        for q in questions:
            if q.category not in categories:
                categories[q.category] = []
            categories[q.category].append(q)
        
        selected = []
        category_names = sorted(categories.keys())
        category_index = {cat: 0 for cat in category_names}
        
        # Single pass: cycle through categories and take next available question
        while len(selected) < max_questions:
            questions_added_this_round = 0
            for category in category_names:
                if (len(selected) < max_questions and 
                    category_index[category] < len(categories[category])):
                    selected.append(categories[category][category_index[category]])
                    category_index[category] += 1
                    questions_added_this_round += 1
            
            # Break if no more questions available in any category
            if questions_added_this_round == 0:
                break
        
        # Sort by question_id to maintain consistent order
        selected.sort(key=lambda q: q.question_id)
        
        # Log category distribution
        selected_categories = {}
        for q in selected:
            selected_categories[q.category] = selected_categories.get(q.category, 0) + 1
        
        logger.info(f"Selected questions by category: {dict(sorted(selected_categories.items()))}")
        return selected
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete MT-bench evaluation for all specified models.
        
        Returns:
            Complete evaluation results dictionary
        """
        logger.info("Starting full MT-bench evaluation")
        self.evaluation_metadata["start_time"] = time.time()
        
        try:
            # Load MT-bench questions
            questions = self.data_loader.load_mtbench_questions()
            
            # Limit questions if specified (prefer one per category)
            if self.max_questions:
                questions = self._select_questions_by_category(questions, self.max_questions)
                logger.info(f"Limited to {len(questions)} questions for testing")
            
            self.evaluation_metadata["total_questions"] = len(questions)
            
            # Evaluate each model
            model_results = []
            for model_name in self.model_names:
                logger.info(f"Evaluating model: {model_name}")
                
                model_result = await self.evaluate_single_model(model_name, questions)
                model_results.append(model_result)
                
                # Cleanup memory between models
                self.model_manager.cleanup()
                self.memory_monitor.cleanup_gpu_memory()
            
            # Aggregate results
            final_results = await self._aggregate_all_results(model_results)
            
            self.evaluation_metadata["end_time"] = time.time()
            final_results["metadata"] = self.evaluation_metadata
            
            logger.info("Full MT-bench evaluation completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            self.evaluation_metadata["end_time"] = time.time()
            self.evaluation_metadata["error"] = str(e)
            raise
        finally:
            # Final cleanup
            self.conversation_handler.cleanup_all_sessions()
            self.model_manager.cleanup()
    
    async def evaluate_single_model(self, model_name: str, 
                                   questions: List[MTBenchQuestion]) -> Dict[str, Any]:
        """
        Evaluate a single model on all questions.
        
        Args:
            model_name: Name of the model to evaluate
            questions: List of MT-bench questions
            
        Returns:
            Model evaluation results
        """
        logger.info(f"Starting evaluation of {model_name}")
        start_time = time.time()
        
        try:
            # Load model
            model, tokenizer = self.model_manager.load_model(model_name)
            model_config = get_model_config(model_name)
            
            self.memory_monitor.log_memory_usage(f"Loaded {model_name}", logger)
            self.memory_monitor.check_memory_usage(f"Loaded {model_name}")
            
            # Generate responses for all questions
            sessions = await self._generate_model_responses(
                model_name, questions, model_config
            )
            
            # Judge all responses
            scores = await self._judge_model_responses(model_name, sessions, questions)
            
            # Store results
            self.all_scores.extend(scores)
            self.all_sessions.extend(sessions)
            
            # Analyze results for this model
            model_result = self.results_analyzer.aggregate_model_results(scores, sessions, questions)
            
            evaluation_time = time.time() - start_time
            model_result["evaluation_time_seconds"] = evaluation_time
            
            logger.info(f"Completed evaluation of {model_name} in {evaluation_time:.1f}s")
            logger.info(f"Average score: {model_result['overall_score']['mean']:.2f}")
            
            return model_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {str(e)}")
            raise
    
    async def _generate_model_responses(self, model_name: str, 
                                      questions: List[MTBenchQuestion],
                                      model_config) -> List[ConversationSession]:
        """
        Generate model responses for all questions.
        
        Args:
            model_name: Name of the model
            questions: List of questions
            model_config: Model configuration
            
        Returns:
            List of conversation sessions
        """
        sessions = []
        
        # Use tqdm for progress tracking
        pbar = tqdm(questions, desc=f"Generating responses ({model_name})")
        
        for question in pbar:
            try:
                session = await self._process_single_question(
                    model_name, question, model_config
                )
                sessions.append(session)
                
                # Update progress bar with memory info
                memory_usage = self.memory_monitor.get_gpu_memory_usage()
                pbar.set_postfix(memory=f"{memory_usage:.1f}GB")
                
            except Exception as e:
                logger.error(f"Failed to process question {question.question_id}: {str(e)}")
                # Continue with other questions
                continue
        
        return sessions
    
    async def _process_single_question(self, model_name: str, 
                                     question: MTBenchQuestion,
                                     model_config) -> ConversationSession:
        """
        Process a single MT-bench question with multi-turn conversation.
        
        Args:
            model_name: Name of the model
            question: MT-bench question
            model_config: Model configuration
            
        Returns:
            Complete conversation session
        """
        # Start conversation
        session_id = self.conversation_handler.start_conversation(question, model_name)
        
        try:
            # Process turns (1 or 2 depending on turn1_only flag)
            turns_to_process = [1] if self.turn1_only else [1, 2]
            for turn_number in turns_to_process:
                # Get current tokenizer for chat template support
                tokenizer = self.model_manager.get_current_tokenizer()
                
                # Format prompt for this turn (with chat template support)
                prompt = self.conversation_handler.format_turn_prompt(
                    session_id, turn_number, question, model_config, tokenizer
                )
                
                # Generate response
                start_time = time.time()
                memory_before = self.memory_monitor.get_gpu_memory_usage()
                
                response = self.model_manager.generate_response(prompt, model_name)
                
                generation_time = time.time() - start_time
                memory_after = self.memory_monitor.get_gpu_memory_usage()
                self.memory_monitor.check_memory_usage(f"Generated response for Q{question.question_id}")
                
                # Add turn to conversation
                self.conversation_handler.add_turn_response(
                    session_id=session_id,
                    turn_number=turn_number,
                    question=question,
                    response=response,
                    generation_time=generation_time,
                    memory_used_gb=memory_after
                )
            
            # End conversation and return session
            session = self.conversation_handler.end_conversation(session_id)
            
            # Validate conversation format
            if not self.conversation_handler.validate_conversation_format(session, question, self.turn1_only):
                logger.warning(f"Invalid conversation format for question {question.question_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error processing question {question.question_id}: {str(e)}")
            # Cleanup session on error
            self.conversation_handler.cleanup_session(session_id)
            raise
    
    async def _judge_model_responses(self, model_name: str, 
                                   sessions: List[ConversationSession],
                                   questions: List[MTBenchQuestion]) -> List[JudgeScore]:
        """
        Judge all model responses using GPT-4.1-nano.
        
        Args:
            model_name: Name of the model
            sessions: List of conversation sessions
            questions: Original questions
            
        Returns:
            List of judge scores
        """
        logger.info(f"Judging responses for {model_name}")
        
        # Prepare evaluation data for judge
        evaluations = []
        question_lookup = {q.question_id: q for q in questions}
        
        for session in sessions:
            question = question_lookup.get(session.question_id)
            if not question:
                logger.warning(f"Question {session.question_id} not found")
                continue
            
            # Create lookup for turns in this session
            turn_lookup = {turn.turn_number: turn for turn in session.turns}
            
            for turn in session.turns:
                # For Turn 2, create contextual question including Turn 1
                if turn.turn_number == 2 and 1 in turn_lookup:
                    turn_1 = turn_lookup[1]
                    # Truncate Turn 1 response if it's very long to avoid token limits
                    turn_1_response = turn_1.assistant_response
                    if len(turn_1_response) > 1000:
                        turn_1_response = turn_1_response[:1000] + "..."
                    contextual_question = f"Turn 1 Question: {turn_1.user_message}\n\nTurn 1 Response: {turn_1_response}\n\nTurn 2 Question: {turn.user_message}"
                    question_text = contextual_question
                else:
                    question_text = turn.user_message
                
                evaluations.append({
                    "question": question_text,
                    "answer": turn.assistant_response,
                    "question_id": session.question_id,
                    "turn": turn.turn_number,
                    "model_name": model_name
                })
        
        # Judge all responses
        scores = await self.judge_client.judge_multiple_responses(evaluations)
        
        logger.info(f"Judged {len(scores)} responses for {model_name}")
        return scores
    
    async def _aggregate_all_results(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all models.
        
        Args:
            model_results: List of individual model results
            
        Returns:
            Complete aggregated results
        """
        logger.info("Aggregating results from all models")
        
        # Compare models
        comparison = self.results_analyzer.compare_models(model_results)
        
        # Generate summary report
        results = {
            "model_results": model_results,
            "comparison": comparison,
            "judge_stats": self.judge_client.get_usage_stats()
        }
        
        summary_report = self.results_analyzer.generate_summary_report(results)
        results["summary_report"] = summary_report
        
        logger.info("Results aggregation completed")
        return results
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "results") -> None:
        """
        Export results in multiple formats.
        
        Args:
            results: Complete evaluation results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp for filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        json_path = output_path / f"mtbench_results_{timestamp}.json"
        self.results_analyzer.export_results(results, json_path, "json")
        
        # Export CSV
        csv_path = output_path / f"mtbench_results_{timestamp}.csv"
        try:
            self.results_analyzer.export_results(results, csv_path, "csv")
        except Exception as e:
            logger.warning(f"Failed to export CSV: {str(e)}")
        
        # Save summary report
        summary_path = output_path / f"mtbench_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(results.get("summary_report", "No summary available"))
        
        logger.info(f"Results exported to {output_path}")
    
    def get_evaluation_progress(self) -> Dict[str, Any]:
        """
        Get current evaluation progress.
        
        Returns:
            Dictionary with progress information
        """
        return {
            "models_completed": len(set(score.model_name for score in self.all_scores)),
            "total_models": len(self.model_names),
            "total_scores": len(self.all_scores),
            "total_sessions": len(self.all_sessions),
            "active_conversations": len(self.conversation_handler.get_active_sessions()),
            "memory_usage_gb": self.memory_monitor.get_gpu_memory_usage(),
            "peak_memory_gb": self.memory_monitor.peak_memory
        }