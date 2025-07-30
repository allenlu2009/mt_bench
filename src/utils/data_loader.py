"""MT-bench dataset loading and parsing utilities."""

import json
import requests
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MTBenchQuestion:
    """Represents a single MT-bench question with multi-turn conversation."""
    question_id: int
    category: str
    turns: List[str]
    
    def __post_init__(self):
        """Validate the question data."""
        if len(self.turns) != 2:
            raise ValueError(f"MT-bench questions must have exactly 2 turns, got {len(self.turns)}")


class DataLoader:
    """
    Loads and manages MT-bench dataset.
    
    Handles downloading from the official FastChat repository if not cached locally.
    """
    
    # Official MT-bench dataset URL
    MTBENCH_URL = (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
        "fastchat/llm_judge/data/mt_bench/question.jsonl"
    )
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "mt_bench_questions.jsonl"
        
    def _download_mtbench_data(self) -> None:
        """
        Download MT-bench dataset from official repository.
        
        Raises:
            requests.RequestException: If download fails
        """
        logger.info("Downloading MT-bench dataset from official repository")
        
        try:
            response = requests.get(self.MTBENCH_URL, timeout=30)
            response.raise_for_status()
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            logger.info(f"MT-bench dataset cached to {self.cache_file}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download MT-bench dataset: {str(e)}")
            raise
    
    def load_mtbench_questions(self, force_download: bool = False) -> List[MTBenchQuestion]:
        """
        Load MT-bench questions from cache or download if needed.
        
        Args:
            force_download: Force re-download even if cached file exists
            
        Returns:
            List of MTBenchQuestion objects
            
        Raises:
            ValueError: If dataset format is invalid
            requests.RequestException: If download fails
        """
        # Download if cache doesn't exist or force_download is True
        if not self.cache_file.exists() or force_download:
            self._download_mtbench_data()
        
        questions = []
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Validate required fields
                        required_fields = ['question_id', 'category', 'turns']
                        for field in required_fields:
                            if field not in data:
                                raise ValueError(f"Missing required field: {field}")
                        
                        question = MTBenchQuestion(
                            question_id=data['question_id'],
                            category=data['category'],
                            turns=data['turns']
                        )
                        
                        questions.append(question)
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid line {line_num}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(questions)} MT-bench questions")
            
            # Validate dataset completeness
            self._validate_dataset(questions)
            
            return questions
            
        except FileNotFoundError:
            logger.error(f"Cache file not found: {self.cache_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to load MT-bench questions: {str(e)}")
            raise
    
    def _validate_dataset(self, questions: List[MTBenchQuestion]) -> None:
        """
        Validate the loaded dataset.
        
        Args:
            questions: List of loaded questions
            
        Raises:
            ValueError: If dataset validation fails
        """
        if len(questions) == 0:
            raise ValueError("No valid questions found in dataset")
        
        # Check expected number of questions (MT-bench should have 80)
        if len(questions) != 80:
            logger.warning(f"Expected 80 questions, got {len(questions)}")
        
        # Check categories
        categories = set(q.category for q in questions)
        expected_categories = {
            'writing', 'roleplay', 'reasoning', 'math', 'coding', 
            'extraction', 'stem', 'humanities'
        }
        
        missing_categories = expected_categories - categories
        if missing_categories:
            logger.warning(f"Missing categories: {missing_categories}")
        
        extra_categories = categories - expected_categories
        if extra_categories:
            logger.warning(f"Unexpected categories: {extra_categories}")
        
        # Check question IDs are unique
        question_ids = [q.question_id for q in questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Duplicate question IDs found")
        
        logger.info(f"Dataset validation passed: {len(questions)} questions, {len(categories)} categories")
    
    def get_questions_by_category(self, questions: List[MTBenchQuestion], 
                                 category: str) -> List[MTBenchQuestion]:
        """
        Filter questions by category.
        
        Args:
            questions: List of all questions
            category: Category to filter by
            
        Returns:
            List of questions in the specified category
        """
        return [q for q in questions if q.category == category]
    
    def get_categories(self, questions: List[MTBenchQuestion]) -> List[str]:
        """
        Get all unique categories in the dataset.
        
        Args:
            questions: List of questions
            
        Returns:
            List of unique category names
        """
        return sorted(set(q.category for q in questions))
    
    def get_dataset_stats(self, questions: List[MTBenchQuestion]) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            questions: List of questions
            
        Returns:
            Dictionary with dataset statistics
        """
        categories = self.get_categories(questions)
        category_counts = {}
        
        for category in categories:
            category_questions = self.get_questions_by_category(questions, category)
            category_counts[category] = len(category_questions)
        
        return {
            "total_questions": len(questions),
            "total_categories": len(categories),
            "categories": categories,
            "questions_per_category": category_counts,
            "question_id_range": (
                min(q.question_id for q in questions),
                max(q.question_id for q in questions)
            ) if questions else (None, None)
        }
    
    def sample_questions(self, questions: List[MTBenchQuestion], 
                        n: int, seed: Optional[int] = None) -> List[MTBenchQuestion]:
        """
        Sample a subset of questions for testing.
        
        Args:
            questions: List of all questions
            n: Number of questions to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled questions
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        if n >= len(questions):
            return questions.copy()
        
        return random.sample(questions, n)
    
    def export_questions(self, questions: List[MTBenchQuestion], 
                        output_path: str, format: str = "json") -> None:
        """
        Export questions to file.
        
        Args:
            questions: List of questions to export
            output_path: Path to output file
            format: Export format ('json' or 'jsonl')
        """
        output_path = Path(output_path)
        
        # Convert questions to dictionaries
        questions_data = [
            {
                "question_id": q.question_id,
                "category": q.category,
                "turns": q.turns
            }
            for q in questions
        ]
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(questions_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for question_data in questions_data:
                    f.write(json.dumps(question_data, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(questions)} questions to {output_path}")


def load_mt_bench_data(cache_dir: str = "data", 
                      force_download: bool = False) -> List[MTBenchQuestion]:
    """
    Convenience function to load MT-bench data.
    
    Args:
        cache_dir: Directory to cache data
        force_download: Force re-download of data
        
    Returns:
        List of MTBenchQuestion objects
    """
    loader = DataLoader(cache_dir)
    return loader.load_mtbench_questions(force_download)