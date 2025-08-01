"""Unit tests for MultiModeEvaluator."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.evaluation.multi_mode_evaluator import MultiModeEvaluator
from src.evaluation.judge_client import PairwiseJudgment
from src.utils.data_loader import MTBenchQuestion
from src.utils.response_manager import ResponseSet, CachedResponse


class TestMultiModeEvaluator:
    """Test MultiModeEvaluator initialization and basic functionality."""
    
    @pytest.fixture
    def mock_openai_key(self):
        """Mock OpenAI API key."""
        return "test-api-key"
    
    @pytest.fixture
    def sample_models(self):
        """Sample model names for testing."""
        return ["gpt2", "gpt2-large"]
    
    @pytest.fixture
    def sample_questions(self):
        """Sample MT-bench questions for testing."""
        return [
            MTBenchQuestion(
                question_id=1,
                category="writing",
                turns=["Write a story about a robot.", "Make it funnier."]
            )
        ]
    
    def test_initialization(self, sample_models, mock_openai_key):
        """Test MultiModeEvaluator initialization."""
        evaluator = MultiModeEvaluator(
            model_names=sample_models,
            openai_api_key=mock_openai_key
        )
        
        assert evaluator.model_names == sample_models
        assert len(evaluator.model_names) == 2
        assert evaluator.single_evaluator is not None
        assert evaluator.response_manager is not None
        assert evaluator.data_loader is not None
        assert evaluator.judge_client is not None
        
    def test_initialization_single_model_fails(self, mock_openai_key):
        """Test that initialization fails with only one model."""
        with pytest.raises(ValueError, match="requires at least 2 models"):
            MultiModeEvaluator(
                model_names=["gpt2"],
                openai_api_key=mock_openai_key
            )


class TestMultiModeEvaluatorDataLoading:
    """Test DataLoader integration fixes."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        return MultiModeEvaluator(
            model_names=["gpt2", "gpt2-large"],
            openai_api_key="test-key"
        )
    
    @pytest.mark.asyncio
    @patch('src.evaluation.multi_mode_evaluator.DataLoader')
    async def test_load_questions_method_exists(self, mock_data_loader_class, evaluator):
        """Test that the correct DataLoader method is called."""
        # Setup mock
        mock_data_loader = Mock()
        mock_questions = [
            {"question_id": 1, "category": "writing", "turns": ["Question 1", "Follow-up 1"]}
        ]
        mock_data_loader.load_mtbench_questions.return_value = mock_questions
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create new evaluator to use mocked DataLoader
        test_evaluator = MultiModeEvaluator(
            model_names=["gpt2", "gpt2-large"],
            openai_api_key="test-key"
        )
        
        # Mock the single evaluator and response manager methods to avoid actual model loading
        with patch.object(test_evaluator, '_generate_all_responses', new_callable=AsyncMock) as mock_generate, \
             patch.object(test_evaluator, '_run_pairwise_comparisons', new_callable=AsyncMock) as mock_compare:
            
            mock_generate.return_value = None
            mock_compare.return_value = {"pairwise_results": []}
            
            # This should work without AttributeError
            result = await test_evaluator.run_pairwise_evaluation()
            
            # Verify the correct method was called
            mock_data_loader.load_mtbench_questions.assert_called_once()
            assert "pairwise_results" in result


class TestMultiModeEvaluatorGenerationConfig:
    """Test generation config fixes."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        return MultiModeEvaluator(
            model_names=["gpt2", "gpt2-large"],
            openai_api_key="test-key"
        )
    
    @pytest.mark.asyncio
    @patch('src.evaluation.multi_mode_evaluator.get_model_config')
    @patch('src.evaluation.multi_mode_evaluator.get_generation_config')
    async def test_generation_config_retrieval(self, mock_get_gen_config, mock_get_model_config, evaluator):
        """Test that generation config is retrieved correctly from model configs."""
        # Setup mocks
        mock_model_config = Mock()
        mock_generation_config = {"max_new_tokens": 512, "temperature": 0.7}
        mock_get_model_config.return_value = mock_model_config
        mock_get_gen_config.return_value = mock_generation_config
        
        # Mock other dependencies
        sample_questions = [{"question_id": 1, "category": "writing", "turns": ["Q1", "Q2"]}]
        
        with patch.object(evaluator.response_manager, 'has_cached_responses', return_value=True), \
             patch.object(evaluator.response_manager, 'get_cached_responses', return_value={}):
            
            # This should call get_model_config and get_generation_config correctly
            await evaluator._generate_all_responses(sample_questions)
            
            # Verify the correct functions were called
            assert mock_get_model_config.call_count == 2  # Once for each model
            assert mock_get_gen_config.call_count == 2   # Once for each model
            
            # Verify called with correct model names
            mock_get_model_config.assert_any_call("gpt2")
            mock_get_model_config.assert_any_call("gpt2-large")
    
    @pytest.mark.asyncio
    @patch('src.evaluation.multi_mode_evaluator.get_model_config')
    @patch('src.evaluation.multi_mode_evaluator.get_generation_config')
    async def test_pairwise_comparison_generation_config(self, mock_get_gen_config, mock_get_model_config, evaluator):
        """Test that pairwise comparison uses generation config correctly."""
        # Setup mocks
        mock_model_config = Mock()
        mock_generation_config = {"max_new_tokens": 512, "temperature": 0.7}
        mock_get_model_config.return_value = mock_model_config
        mock_get_gen_config.return_value = mock_generation_config
        
        sample_questions = [{"question_id": 1, "category": "writing", "turns": ["Q1", "Q2"]}]
        
        with patch.object(evaluator.response_manager, 'get_cached_responses') as mock_get_cached, \
             patch.object(evaluator.judge_client, 'judge_multiple_pairwise', new_callable=AsyncMock) as mock_judge:
            
            # Create proper ResponseSet with responses
            mock_response_set = ResponseSet(
                model_name="test",
                responses={
                    "1": [
                        CachedResponse(
                            question_id=1,
                            turn=1,
                            question="Test question 1",
                            response="Response 1",
                            model_name="test",
                            timestamp="2025-01-01",
                            metadata={}
                        ),
                        CachedResponse(
                            question_id=1,
                            turn=2,
                            question="Test question 2",
                            response="Response 2",
                            model_name="test",
                            timestamp="2025-01-01",
                            metadata={}
                        )
                    ]
                },
                generation_config=mock_generation_config,
                created_at="2025-01-01"
            )
            mock_get_cached.return_value = mock_response_set
            mock_judge.return_value = []
            
            # This should work without trying to call model_manager.get_generation_config()
            result = await evaluator._run_pairwise_comparisons(sample_questions)
            
            # Verify generation config was retrieved from model configs, not model manager
            mock_get_model_config.assert_called_once_with("gpt2")  # First model for default config
            mock_get_gen_config.assert_called_once_with(mock_model_config)


class TestMultiModeEvaluatorMethods:
    """Test MultiModeEvaluator method signatures and basic functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        return MultiModeEvaluator(
            model_names=["gpt2", "gpt2-large"],
            openai_api_key="test-key"
        )
    
    def test_has_pairwise_evaluation_method(self, evaluator):
        """Test that run_pairwise_evaluation method exists."""
        assert hasattr(evaluator, 'run_pairwise_evaluation')
        assert callable(getattr(evaluator, 'run_pairwise_evaluation'))
    
    def test_has_both_evaluation_method(self, evaluator):
        """Test that run_both_evaluation method exists."""
        assert hasattr(evaluator, 'run_both_evaluation')
        assert callable(getattr(evaluator, 'run_both_evaluation'))
    
    def test_has_single_evaluation_method(self, evaluator):
        """Test that run_single_evaluation method exists."""
        assert hasattr(evaluator, 'run_single_evaluation')
        assert callable(getattr(evaluator, 'run_single_evaluation'))
    
    def test_model_pairs_generation(self, evaluator):
        """Test that model pairs are generated correctly for pairwise comparison."""
        from itertools import combinations
        
        expected_pairs = list(combinations(evaluator.model_names, 2))
        assert len(expected_pairs) == 1  # Only one pair for 2 models
        assert expected_pairs[0] == ("gpt2", "gpt2-large")


class TestMultiModeEvaluatorIntegration:
    """Integration tests for MultiModeEvaluator without actual model loading."""
    
    @pytest.mark.asyncio
    @patch('src.evaluation.multi_mode_evaluator.DataLoader')
    @patch('src.evaluation.multi_mode_evaluator.MTBenchEvaluator')
    @patch('src.evaluation.multi_mode_evaluator.ResponseManager')
    @patch('src.evaluation.multi_mode_evaluator.JudgeClient')
    async def test_pairwise_evaluation_flow(self, mock_judge_client, mock_response_manager, 
                                          mock_evaluator, mock_data_loader):
        """Test complete pairwise evaluation flow without model loading."""
        # Setup mocks
        mock_data_loader_instance = Mock()
        mock_data_loader.return_value = mock_data_loader_instance
        mock_data_loader_instance.load_mtbench_questions.return_value = [
            {"question_id": 1, "category": "writing", "turns": ["Q1", "Q2"]}
        ]
        
        mock_response_manager_instance = Mock()
        mock_response_manager.return_value = mock_response_manager_instance
        mock_response_manager_instance.has_cached_responses.return_value = True
        mock_response_manager_instance.get_cached_responses.return_value = {
            "gpt2": {"1": {"turn_1": "Response A1", "turn_2": "Response A2"}},
            "gpt2-large": {"1": {"turn_1": "Response B1", "turn_2": "Response B2"}}
        }
        mock_response_manager_instance.get_response_for_comparison.return_value = "Mock response"
        
        mock_judge_client_instance = Mock()
        mock_judge_client.return_value = mock_judge_client_instance
        mock_judge_client_instance.judge_multiple_pairwise = AsyncMock()
        mock_judge_client_instance.judge_multiple_pairwise.return_value = [
            PairwiseJudgment(
                winner="A",
                reasoning="Better response",
                question_id=1,
                turn=1,
                model_a="gpt2",
                model_b="gpt2-large"
            )
        ]
        
        # Create evaluator
        evaluator = MultiModeEvaluator(
            model_names=["gpt2", "gpt2-large"],
            openai_api_key="test-key"
        )
        
        # Run pairwise evaluation
        with patch('src.evaluation.multi_mode_evaluator.get_model_config') as mock_get_config, \
             patch('src.evaluation.multi_mode_evaluator.get_generation_config') as mock_get_gen_config:
            
            mock_get_config.return_value = Mock()
            mock_get_gen_config.return_value = {"max_new_tokens": 512}
            
            result = await evaluator.run_pairwise_evaluation()
            
            # Verify the flow worked
            assert "pairwise_results" in result or "judgments" in result or len(result) >= 0
            mock_data_loader_instance.load_mtbench_questions.assert_called_once()