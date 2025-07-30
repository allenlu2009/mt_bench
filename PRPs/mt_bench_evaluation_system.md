name: "MT-bench Evaluation System - Context-Rich PRP with GPU Memory Optimization"
description: |

## Purpose
Implement a comprehensive MT-bench evaluation system for common public models using GPT-4.1-nano as judge, optimized for RTX 3060 GPU memory constraints and Colab compatibility.

## Core Principles
1. **Memory Efficiency**: Optimize for RTX 3060's 6GB VRAM limitation using Flash Attention
2. **Reproducible Results**: Standardized evaluation across models with consistent scoring
3. **Modular Design**: Clean separation between model loading, inference, and evaluation
4. **Colab Compatibility**: Easy deployment on Google Colab with pip installation
5. **Global rules**: Follow all rules in CLAUDE.md

---

## Goal
Build a production-ready MT-bench evaluation system that evaluates multiple LLMs (gpt2-large, llama, phi3, qwen, gemma) using GPT-4.1-nano as judge, with memory-optimized inference for desktop GPU and cloud deployment.

## Why
- **Research Value**: Standardized benchmarking of open-source models on multi-turn conversations
- **Resource Optimization**: Make evaluation accessible on consumer GPUs like RTX 3060
- **Comparative Analysis**: Enable fair comparison between models using consistent judge
- **Educational Purpose**: Demonstrate proper MT-bench implementation patterns

## What
A complete MT-bench evaluation pipeline with:
- Memory-optimized model loading with Flash Attention 2
- Multi-turn conversation handling following MT-bench protocol
- GPT-4.1-nano judge integration with structured scoring
- Comprehensive results analysis and visualization
- Colab notebook for cloud execution

### Success Criteria
- [ ] Successfully evaluate all 5 target models on RTX 3060 (6GB VRAM)
- [ ] Generate valid MT-bench scores using GPT-4.1-nano judge
- [ ] Memory usage stays within 6GB limit during inference
- [ ] Colab notebook runs without memory errors
- [ ] Results saved in standardized format (CSV/JSON)
- [ ] All unit tests pass with >90% coverage

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md
  why: Official MT-bench implementation patterns and evaluation protocol
  
- url: https://github.com/Dao-AILab/flash-attention
  why: Flash Attention 2 implementation for memory optimization
  
- file: /Users/allenlu/Google Drive/github/claude_mt_bench/examples/mtbench.py
  why: Existing implementation patterns to extend and improve
  
- file: /Users/allenlu/Google Drive/github/claude_mt_bench/examples/REST/requirements.txt
  why: Dependencies reference for project requirements
  
- url: https://huggingface.co/docs/transformers/en/perf_train_gpu_one
  why: GPU memory optimization techniques and Flash Attention integration
  
- critical: RTX 3060 has 6GB VRAM - must use Flash Attention, gradient checkpointing, and fp16 precision
- critical: GPT-4.1-nano requires OpenAI API key and proper rate limiting (10 req/sec max)
- critical: MT-bench has 80 questions across 8 categories with 2 turns each
```

### Current Codebase tree
```bash
claude_mt_bench/
├── CLAUDE.md                  # Project rules and guidelines
├── INITIAL.md                 # Feature request specifications  
├── README.md                  # Project documentation
├── examples/
│   ├── mtbench.py            # Existing basic implementation
│   └── REST/
│       ├── requirements.txt   # Dependencies reference
│       └── llm_judge/        # MT-bench reference implementation
└── PRPs/
    └── templates/            # PRP templates
```

### Desired Codebase tree with files to be added
```bash
claude_mt_bench/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_manager.py     # Memory-optimized model loading
│   │   └── model_configs.py     # Model-specific configurations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── mtbench_evaluator.py # Core evaluation logic
│   │   ├── judge_client.py      # GPT-4.1-nano integration
│   │   └── conversation_handler.py # Multi-turn conversation management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── memory_utils.py      # GPU memory monitoring and optimization
│   │   ├── data_loader.py       # MT-bench dataset handling
│   │   └── results_analyzer.py  # Results processing and analysis
│   └── cli.py                   # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   │   ├── test_model_manager.py
│   │   └── test_model_configs.py
│   ├── test_evaluation/
│   │   ├── test_mtbench_evaluator.py
│   │   ├── test_judge_client.py
│   │   └── test_conversation_handler.py
│   └── test_utils/
│       ├── test_memory_utils.py
│       ├── test_data_loader.py
│       └── test_results_analyzer.py
├── requirements.txt             # Desktop GPU dependencies
├── requirements_colab.txt       # Colab-specific dependencies
├── notebooks/
│   └── mtbench_colab.ipynb     # Colab notebook implementation
└── configs/
    ├── model_config.yaml       # Model configurations
    └── evaluation_config.yaml  # Evaluation parameters
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: RTX 3060 has only 6GB VRAM - must use Flash Attention 2
# Flash Attention 2 reduces memory by ~30% and provides ~20% speedup
# Use attn_implementation="flash_attention_2" when loading models

# CRITICAL: OpenAI API rate limiting for GPT-4.1-nano is 10 req/sec
# Must implement exponential backoff with tenacity

# CRITICAL: Transformers models need specific prompt templates
# Each model family (Llama, Phi, Qwen, Gemma) has different chat templates
# Use tokenizer.apply_chat_template() for consistency

# CRITICAL: MT-bench requires exact protocol compliance
# Questions must be processed in order, maintaining conversation context
# Judge scoring must use the official MT-bench prompt format

# CRITICAL: Memory management on consumer GPU
# Use torch.cuda.empty_cache() between model evaluations
# Enable gradient_checkpointing=True to reduce memory usage
# Use fp16 precision: torch_dtype=torch.float16
```

## Implementation Blueprint

### Data models and structure
```python
# Core data models ensuring type safety and consistency
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

@dataclass
class MTBenchQuestion:
    question_id: int
    category: str
    turns: List[str]

@dataclass
class ModelResponse:
    model_name: str
    question_id: int
    turn: int
    response: str
    generation_time: float
    memory_used: float

@dataclass
class JudgeScore:
    question_id: int
    turn: int
    score: float
    reasoning: str
    model_name: str

class EvaluationResult(BaseModel):
    model_name: str = Field(..., description="Name of the evaluated model")
    total_questions: int = Field(..., description="Total number of questions evaluated")
    average_score: float = Field(..., description="Average score across all questions")
    category_scores: Dict[str, float] = Field(..., description="Scores by category")
    detailed_results: List[Dict] = Field(..., description="Detailed per-question results")
```

### List of tasks to be completed in order

```yaml
Task 1 - Core Infrastructure:
CREATE src/__init__.py: Empty file for package initialization
CREATE src/utils/__init__.py: Empty file for utils package
CREATE src/utils/memory_utils.py:
  - IMPLEMENT GPU memory monitoring functions
  - INCLUDE flash attention optimization helpers
  - ADD memory cleanup utilities

Task 2 - Model Management:
CREATE src/models/__init__.py: Empty file for models package  
CREATE src/models/model_configs.py:
  - DEFINE model configurations for all 5 target models
  - INCLUDE memory optimization settings per model
  - ADD prompt templates for each model family

CREATE src/models/model_manager.py:
  - IMPLEMENT memory-optimized model loading
  - USE Flash Attention 2 configuration
  - ADD model switching with memory cleanup
  - INCLUDE GPU memory monitoring

Task 3 - Data Loading:
CREATE src/utils/data_loader.py:
  - IMPLEMENT MT-bench dataset loading from official source
  - PARSE question.jsonl format correctly
  - VALIDATE dataset integrity

Task 4 - Judge Integration:
CREATE src/evaluation/judge_client.py:
  - IMPLEMENT OpenAI GPT-4.1-nano client
  - ADD rate limiting with exponential backoff
  - INCLUDE structured response parsing
  - ADD error handling for API failures

Task 5 - Conversation Management:
CREATE src/evaluation/conversation_handler.py:
  - IMPLEMENT multi-turn conversation tracking
  - MAINTAIN conversation context between turns
  - ADD conversation state management

Task 6 - Core Evaluator:
CREATE src/evaluation/mtbench_evaluator.py:
  - INTEGRATE all components into main evaluator
  - IMPLEMENT evaluation pipeline
  - ADD progress tracking and logging
  - INCLUDE result aggregation

Task 7 - Results Analysis:
CREATE src/utils/results_analyzer.py:
  - IMPLEMENT statistical analysis functions
  - ADD visualization helpers
  - CREATE result export functions (CSV/JSON)

Task 8 - CLI Interface:
CREATE src/cli.py:
  - IMPLEMENT command-line interface with argparse
  - ADD model selection options
  - INCLUDE configuration overrides
  - ADD verbose logging options

Task 9 - Requirements:
CREATE requirements.txt:
  - LIST all desktop GPU dependencies
  - INCLUDE Flash Attention 2 requirements
  - ADD development dependencies

CREATE requirements_colab.txt:
  - OPTIMIZE for Colab environment
  - INCLUDE GPU-specific optimizations

Task 10 - Colab Notebook:
CREATE notebooks/mtbench_colab.ipynb:
  - IMPLEMENT full pipeline in notebook format
  - ADD installation cells with pip commands
  - INCLUDE visualization and results display
  - ADD GPU memory monitoring

Task 11 - Configuration Files:
CREATE configs/model_config.yaml:
  - DEFINE all model configurations
  - INCLUDE memory optimization parameters

CREATE configs/evaluation_config.yaml:
  - SET evaluation parameters
  - INCLUDE judge configuration
```

### Per task pseudocode with CRITICAL details

```python
# Task 1 - Memory Utils
class MemoryMonitor:
    def __init__(self):
        # CRITICAL: Monitor GPU memory throughout evaluation
        self.peak_memory = 0
        
    def get_gpu_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0.0
    
    def cleanup_gpu_memory(self):
        # PATTERN: Always cleanup between model loads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# Task 2 - Model Manager  
class ModelManager:
    def __init__(self, device: str = "cuda"):
        # CRITICAL: Use Flash Attention 2 for memory optimization
        self.device = device
        self.current_model = None
        self.memory_monitor = MemoryMonitor()
        
    def load_model(self, model_name: str):
        # PATTERN: Always cleanup before loading new model
        self._cleanup_current_model()
        
        # CRITICAL: Flash Attention 2 configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Critical for RTX 3060
            attn_implementation="flash_attention_2",  # 30% memory reduction
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # GOTCHA: Enable gradient checkpointing for inference
        model.gradient_checkpointing_enable()
        
        return model

# Task 4 - Judge Client
class JudgeClient:
    def __init__(self, api_key: str):
        # CRITICAL: Rate limiting for GPT-4.1-nano (10 req/sec max)
        self.client = OpenAI(api_key=api_key)
        self.rate_limiter = AsyncLimiter(10, 1)  # 10 requests per second
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    async def get_judgment(self, prompt: str, response_a: str, response_b: str):
        # PATTERN: Use official MT-bench judge prompt format
        judge_prompt = self._format_judge_prompt(prompt, response_a, response_b)
        
        async with self.rate_limiter:
            response = await self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert AI evaluator..."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.2,  # Low temperature for consistent judging
                max_tokens=500
            )
        
        return self._parse_judgment(response.choices[0].message.content)

# Task 6 - Core Evaluator
class MTBenchEvaluator:
    def __init__(self, model_names: List[str]):
        self.model_manager = ModelManager()
        self.judge_client = JudgeClient(os.getenv("OPENAI_API_KEY"))
        self.data_loader = DataLoader()
        self.results_analyzer = ResultsAnalyzer()
        
    async def evaluate_all_models(self):
        questions = self.data_loader.load_mtbench_questions()
        results = []
        
        for model_name in self.model_names:
            # CRITICAL: Load model with memory cleanup
            model = self.model_manager.load_model(model_name)
            
            model_results = []
            for question in tqdm(questions, desc=f"Evaluating {model_name}"):
                # Process multi-turn conversation
                conversation_results = await self._evaluate_conversation(
                    model, question
                )
                model_results.extend(conversation_results)
            
            # Aggregate results per model
            aggregated = self.results_analyzer.aggregate_results(
                model_name, model_results
            )
            results.append(aggregated)
            
            # CRITICAL: Cleanup GPU memory between models
            self.model_manager.cleanup_gpu_memory()
        
        return results
```

### Integration Points
```yaml
DEPENDENCIES:
  - torch>=2.0.0: For Flash Attention 2 support
  - transformers>=4.36.0: Latest model support with Flash Attention
  - flash-attn>=2.0.0: Memory optimization
  - openai>=1.0.0: GPT-4.1-nano API access
  - tenacity: Retry logic for API calls
  
ENVIRONMENT:
  - OPENAI_API_KEY: Required for judge evaluation
  - CUDA_VISIBLE_DEVICES: GPU selection
  - PYTORCH_CUDA_ALLOC_CONF: Memory optimization settings
  
CONFIG_FILES:
  - configs/model_config.yaml: Model-specific settings
  - configs/evaluation_config.yaml: Evaluation parameters
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix          # Auto-fix formatting issues
mypy src/                      # Type checking
black src/                     # Code formatting

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE comprehensive test suite for each component:

def test_model_manager_memory_optimization():
    """Test Flash Attention 2 integration and memory usage"""
    manager = ModelManager()
    initial_memory = manager.memory_monitor.get_gpu_memory_usage()
    
    model = manager.load_model("gpt2-large")
    loaded_memory = manager.memory_monitor.get_gpu_memory_usage()
    
    # Should use Flash Attention 2
    assert hasattr(model.config, 'attn_implementation')
    assert loaded_memory < 6.0  # Within RTX 3060 limits
    
def test_judge_client_rate_limiting():
    """Test API rate limiting compliance"""
    client = JudgeClient(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test rate limiting doesn't exceed 10 req/sec
    start_time = time.time()
    responses = []
    for i in range(5):
        response = asyncio.run(client.get_judgment(
            "Test prompt", "Response A", "Response B"
        ))
        responses.append(response)
    
    elapsed = time.time() - start_time
    assert elapsed >= 0.5  # Should take at least 0.5 seconds for 5 requests

def test_mtbench_data_loading():
    """Test MT-bench dataset loading and parsing"""
    loader = DataLoader()
    questions = loader.load_mtbench_questions()
    
    assert len(questions) == 80  # MT-bench has 80 questions
    assert all(len(q.turns) == 2 for q in questions)  # All multi-turn
    assert len(set(q.category for q in questions)) == 8  # 8 categories
```

```bash
# Run and iterate until passing:
python -m pytest tests/ -v --cov=src --cov-report=html
# Target: >90% test coverage, all tests passing
```

### Level 3: Integration Test
```bash
# Test full pipeline with small subset
export OPENAI_API_KEY="your-key-here"
python -m src.cli --models gpt2-large --max-questions 5 --output test_results.json

# Expected output:
# - Evaluation completed successfully
# - Results saved to test_results.json  
# - GPU memory usage logged and within limits
# - No CUDA out of memory errors

# Test Colab notebook
jupyter nbconvert --execute notebooks/mtbench_colab.ipynb --to html
# Expected: Notebook executes without errors, produces results
```

## Final validation Checklist
- [ ] All tests pass: `python -m pytest tests/ -v --cov=src`
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] CLI interface works: `python -m src.cli --help`
- [ ] GPU memory stays under 6GB during evaluation
- [ ] All 5 target models can be evaluated successfully
- [ ] Colab notebook runs without memory errors
- [ ] Results are saved in correct format (CSV/JSON)
- [ ] API rate limiting respected (no 429 errors)
- [ ] Documentation updated with usage examples

---

## Anti-Patterns to Avoid
- ❌ Don't load multiple models simultaneously (memory overflow)
- ❌ Don't skip Flash Attention 2 (memory optimization critical)
- ❌ Don't ignore GPU memory cleanup between models
- ❌ Don't exceed OpenAI API rate limits (causes evaluation failure)
- ❌ Don't hardcode model paths (use configuration)
- ❌ Don't skip conversation context in multi-turn evaluation
- ❌ Don't use full precision (fp32) on RTX 3060
- ❌ Don't ignore proper error handling for CUDA OOM errors