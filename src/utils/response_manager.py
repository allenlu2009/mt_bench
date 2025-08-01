"""Response caching and management for MT-bench evaluations."""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Represents a cached model response."""
    question_id: int
    turn: int
    question: str
    response: str
    model_name: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class ResponseSet:
    """Collection of responses for a model."""
    model_name: str
    responses: Dict[str, List[CachedResponse]]  # question_id -> [turn1, turn2]
    generation_config: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "responses": {
                qid: [asdict(resp) for resp in turns] 
                for qid, turns in self.responses.items()
            },
            "generation_config": self.generation_config,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseSet':
        """Create from dictionary loaded from JSON."""
        responses = {}
        for qid, turns in data["responses"].items():
            responses[qid] = [CachedResponse(**resp) for resp in turns]
        
        return cls(
            model_name=data["model_name"],
            responses=responses,
            generation_config=data.get("generation_config", {}),
            created_at=data.get("created_at", "")
        )


class ResponseManager:
    """
    Manages model responses with in-memory caching and optional disk persistence.
    
    Features:
    - In-memory caching for fast access during evaluation
    - Optional disk persistence for reuse across sessions
    - Cache invalidation based on model/config changes
    - Efficient storage and retrieval
    """
    
    def __init__(self, cache_dir: str = "cached_responses", use_disk_cache: bool = True):
        """
        Initialize ResponseManager.
        
        Args:
            cache_dir: Directory for disk cache files
            use_disk_cache: Whether to enable disk persistence
        """
        self.cache_dir = Path(cache_dir)
        self.use_disk_cache = use_disk_cache
        self.memory_cache: Dict[str, ResponseSet] = {}
        
        if self.use_disk_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"ResponseManager initialized with disk cache: {self.cache_dir}")
        else:
            logger.info("ResponseManager initialized with memory-only cache")
    
    def _get_cache_key(self, model_name: str, generation_config: Dict[str, Any]) -> str:
        """
        Generate cache key based on model name and generation config.
        
        Args:
            model_name: Name of the model
            generation_config: Generation configuration parameters
            
        Returns:
            Unique cache key string
        """
        config_str = json.dumps(generation_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{model_name}_{config_hash}"
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def has_cached_responses(self, model_name: str, generation_config: Dict[str, Any]) -> bool:
        """
        Check if responses are cached for given model and config.
        
        Args:
            model_name: Name of the model
            generation_config: Generation configuration
            
        Returns:
            True if responses are cached
        """
        cache_key = self._get_cache_key(model_name, generation_config)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return True
            
        # Check disk cache
        if self.use_disk_cache:
            cache_file = self._get_cache_file_path(cache_key)
            return cache_file.exists()
            
        return False
    
    def get_cached_responses(self, 
                           model_name: str, 
                           generation_config: Dict[str, Any]) -> Optional[ResponseSet]:
        """
        Retrieve cached responses for model and config.
        
        Args:
            model_name: Name of the model
            generation_config: Generation configuration
            
        Returns:
            ResponseSet if cached, None otherwise
        """
        cache_key = self._get_cache_key(model_name, generation_config)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            logger.debug(f"Retrieved responses from memory cache: {cache_key}")
            return self.memory_cache[cache_key]
        
        # Try disk cache
        if self.use_disk_cache:
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    response_set = ResponseSet.from_dict(data)
                    
                    # Cache in memory for future access
                    self.memory_cache[cache_key] = response_set
                    
                    logger.info(f"Loaded responses from disk cache: {cache_key}")
                    return response_set
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    # Remove corrupted cache file
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def cache_responses(self, 
                       model_name: str,
                       responses: Dict[str, List[Dict[str, Any]]], 
                       generation_config: Dict[str, Any]) -> None:
        """
        Cache responses for a model.
        
        Args:
            model_name: Name of the model
            responses: Dictionary mapping question_id to list of turn responses
            generation_config: Generation configuration used
        """
        cache_key = self._get_cache_key(model_name, generation_config)
        
        # Convert to CachedResponse objects
        cached_responses = {}
        for question_id, turns in responses.items():
            cached_turns = []
            for turn_idx, turn_data in enumerate(turns):
                cached_response = CachedResponse(
                    question_id=int(question_id),
                    turn=turn_idx + 1,
                    question=turn_data.get("question", ""),
                    response=turn_data.get("response", ""),
                    model_name=model_name,
                    timestamp=datetime.now().isoformat(),
                    metadata=turn_data.get("metadata", {})
                )
                cached_turns.append(cached_response)
            cached_responses[question_id] = cached_turns
        
        # Create ResponseSet
        response_set = ResponseSet(
            model_name=model_name,
            responses=cached_responses,
            generation_config=generation_config,
            created_at=datetime.now().isoformat()
        )
        
        # Cache in memory
        self.memory_cache[cache_key] = response_set
        logger.debug(f"Cached responses in memory: {cache_key}")
        
        # Optionally save to disk
        if self.use_disk_cache:
            self._save_to_disk(cache_key, response_set)
    
    def _save_to_disk(self, cache_key: str, response_set: ResponseSet) -> None:
        """Save ResponseSet to disk."""
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(response_set.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved responses to disk cache: {cache_file}")
            
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save cache file {cache_file}: {e}")
    
    def get_response_for_comparison(self, 
                                  model_name: str, 
                                  question_id: str, 
                                  turn: int,
                                  generation_config: Dict[str, Any]) -> Optional[str]:
        """
        Get specific response for pairwise comparison.
        
        Args:
            model_name: Name of the model
            question_id: Question identifier
            turn: Turn number (1 or 2)
            generation_config: Generation configuration
            
        Returns:
            Response text if found, None otherwise
        """
        response_set = self.get_cached_responses(model_name, generation_config)
        if not response_set:
            return None
        
        if question_id not in response_set.responses:
            return None
        
        turns = response_set.responses[question_id]
        if turn - 1 >= len(turns):
            return None
        
        return turns[turn - 1].response
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached responses.
        
        Args:
            model_name: If provided, clear only this model's cache. Otherwise, clear all.
        """
        if model_name:
            # Clear specific model from memory
            keys_to_remove = [key for key in self.memory_cache.keys() 
                             if self.memory_cache[key].model_name == model_name]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # Clear from disk
            if self.use_disk_cache:
                for cache_file in self.cache_dir.glob(f"{model_name}_*.json"):
                    cache_file.unlink()
            
            logger.info(f"Cleared cache for model: {model_name}")
        else:
            # Clear all caches
            self.memory_cache.clear()
            
            if self.use_disk_cache:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            logger.info("Cleared all caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        memory_models = list({rs.model_name for rs in self.memory_cache.values()})
        
        disk_files = 0
        disk_size = 0
        if self.use_disk_cache and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.json"))
            disk_files = len(cache_files)
            disk_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "memory_cache_entries": len(self.memory_cache),
            "memory_cached_models": memory_models,
            "disk_cache_enabled": self.use_disk_cache,
            "disk_cache_files": disk_files,
            "disk_cache_size_bytes": disk_size,
            "cache_directory": str(self.cache_dir) if self.use_disk_cache else None
        }
    
    def list_cached_models(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        List all cached models with their configurations.
        
        Returns:
            List of tuples: (model_name, generation_config, created_at)
        """
        models = []
        
        # From memory cache
        for response_set in self.memory_cache.values():
            models.append((
                response_set.model_name,
                response_set.generation_config,
                response_set.created_at
            ))
        
        # From disk cache (if not already in memory)
        if self.use_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    model_name = data.get("model_name", "unknown")
                    # Skip if already in memory
                    if any(m[0] == model_name for m in models):
                        continue
                    
                    models.append((
                        model_name,
                        data.get("generation_config", {}),
                        data.get("created_at", "")
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return sorted(models, key=lambda x: x[2], reverse=True)  # Sort by created_at