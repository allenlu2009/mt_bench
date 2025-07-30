"""Multi-turn conversation handling for MT-bench evaluation."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..utils.data_loader import MTBenchQuestion
from ..models.model_configs import ModelConfig, format_prompt_for_model

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""
    turn_number: int
    user_message: str
    assistant_response: str
    generation_time: float = 0.0
    memory_used_gb: float = 0.0


@dataclass
class ConversationSession:
    """Represents a complete multi-turn conversation session."""
    question_id: int
    category: str
    model_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    total_time: float = 0.0
    peak_memory_gb: float = 0.0
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.total_time += turn.generation_time
        self.peak_memory_gb = max(self.peak_memory_gb, turn.memory_used_gb)
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history up to current point."""
        history_parts = []
        
        for turn in self.turns:
            history_parts.append(f"User: {turn.user_message}")
            history_parts.append(f"Assistant: {turn.assistant_response}")
        
        return "\n".join(history_parts)
    
    def get_context_for_turn(self, turn_number: int) -> str:
        """
        Get conversation context needed for a specific turn.
        
        Args:
            turn_number: Turn number (1 or 2)
            
        Returns:
            Conversation context string
        """
        if turn_number == 1:
            return ""  # First turn has no context
        
        # For turn 2, include the first turn
        if len(self.turns) >= 1:
            first_turn = self.turns[0]
            return f"User: {first_turn.user_message}\nAssistant: {first_turn.assistant_response}\n"
        
        return ""


class ConversationHandler:
    """
    Handles multi-turn conversations for MT-bench evaluation.
    
    Manages conversation state, context building, and prompt formatting
    according to MT-bench protocol requirements.
    """
    
    def __init__(self):
        """Initialize conversation handler."""
        self.active_sessions: Dict[str, ConversationSession] = {}
        logger.info("ConversationHandler initialized")
    
    def start_conversation(self, question: MTBenchQuestion, model_name: str) -> str:
        """
        Start a new conversation session.
        
        Args:
            question: MT-bench question with turns
            model_name: Name of the model being evaluated
            
        Returns:
            Session ID for tracking the conversation
        """
        session_id = f"{model_name}_{question.question_id}"
        
        session = ConversationSession(
            question_id=question.question_id,
            category=question.category,
            model_name=model_name
        )
        
        self.active_sessions[session_id] = session
        
        logger.debug(f"Started conversation session: {session_id}")
        return session_id
    
    def format_turn_prompt(self, session_id: str, turn_number: int, 
                          question: MTBenchQuestion, model_config: ModelConfig) -> str:
        """
        Format prompt for a specific turn with appropriate context.
        
        Args:
            session_id: Conversation session ID
            turn_number: Turn number (1 or 2)
            question: MT-bench question
            model_config: Configuration for the model
            
        Returns:
            Formatted prompt string
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        # Get the current turn's question
        current_question = question.turns[turn_number - 1]
        
        # Get conversation context
        context = session.get_context_for_turn(turn_number)
        
        # Format with model-specific template
        prompt = format_prompt_for_model(
            instruction=current_question,
            model_config=model_config,
            conversation_history=context
        )
        
        logger.debug(f"Formatted prompt for {session_id}, turn {turn_number}")
        return prompt
    
    def add_turn_response(self, session_id: str, turn_number: int, 
                         question: MTBenchQuestion, response: str,
                         generation_time: float = 0.0, memory_used_gb: float = 0.0) -> ConversationTurn:
        """
        Add a model response to the conversation.
        
        Args:
            session_id: Conversation session ID
            turn_number: Turn number (1 or 2)
            question: MT-bench question
            response: Model's response
            generation_time: Time taken for generation
            memory_used_gb: GPU memory used during generation
            
        Returns:
            ConversationTurn object
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        # Get the question for this turn
        user_message = question.turns[turn_number - 1]
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=response,
            generation_time=generation_time,
            memory_used_gb=memory_used_gb
        )
        
        # Add to session
        session.add_turn(turn)
        
        logger.debug(f"Added turn {turn_number} to session {session_id}")
        return turn
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get conversation session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            ConversationSession or None if not found
        """
        return self.active_sessions.get(session_id)
    
    def end_conversation(self, session_id: str) -> ConversationSession:
        """
        End conversation and return the complete session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Complete ConversationSession
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        del self.active_sessions[session_id]
        
        logger.debug(f"Ended conversation session: {session_id}")
        return session
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.active_sessions.keys())
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up a specific session.
        
        Args:
            session_id: Session ID to clean up
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.debug(f"Cleaned up session: {session_id}")
    
    def cleanup_all_sessions(self) -> None:
        """Clean up all active sessions."""
        session_count = len(self.active_sessions)
        self.active_sessions.clear()
        logger.info(f"Cleaned up {session_count} conversation sessions")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a conversation session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with session statistics
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "question_id": session.question_id,
            "category": session.category,
            "model_name": session.model_name,
            "turns_completed": len(session.turns),
            "total_time": session.total_time,
            "peak_memory_gb": session.peak_memory_gb,
            "average_turn_time": (session.total_time / len(session.turns)) if session.turns else 0.0
        }
    
    def validate_conversation_format(self, session: ConversationSession, 
                                   question: MTBenchQuestion) -> bool:
        """
        Validate that conversation follows MT-bench format requirements.
        
        Args:
            session: Conversation session to validate
            question: Original MT-bench question
            
        Returns:
            True if valid, False otherwise
        """
        # Check number of turns
        if len(session.turns) != 2:
            logger.warning(f"Invalid turn count: {len(session.turns)}, expected 2")
            return False
        
        # Check that questions match
        for i, turn in enumerate(session.turns):
            expected_question = question.turns[i]
            if turn.user_message != expected_question:
                logger.warning(f"Turn {i+1} question mismatch")
                return False
        
        # Check that responses are not empty
        for i, turn in enumerate(session.turns):
            if not turn.assistant_response.strip():
                logger.warning(f"Empty response in turn {i+1}")
                return False
        
        return True
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Export session data in a standardized format.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with complete session data
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "question_id": session.question_id,
            "category": session.category,
            "model_name": session.model_name,
            "turns": [
                {
                    "turn_number": turn.turn_number,
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "generation_time": turn.generation_time,
                    "memory_used_gb": turn.memory_used_gb
                }
                for turn in session.turns
            ],
            "total_time": session.total_time,
            "peak_memory_gb": session.peak_memory_gb
        }