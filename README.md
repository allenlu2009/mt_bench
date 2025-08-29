# MT-bench Evaluation System

A comprehensive system for evaluating language models using the MT-bench benchmark with GPU memory optimization for RTX 3060 and Google Colab compatibility.

## Features

🚀 **Memory Optimized**: Uses Flash Attention 2 and fp16 precision for RTX 3060 (6GB VRAM)  
⚡ **Fast Evaluation**: Async processing with adaptive rate limiting for multiple judge models
🤖 **Multi-Judge Support**: GPT-5-mini, GPT-4o-mini, GPT-4.1-nano with automatic parameter handling
📊 **Comprehensive Analysis**: Category-wise and turn-wise performance metrics  
🔧 **Easy Deployment**: CLI interface and Google Colab notebook  
🧪 **Well Tested**: Unit tests with >90% coverage  
📈 **Rich Visualizations**: Performance heatmaps and comparison charts  

## Supported Models

- **gpt2-large** (774M params, ~1.5GB) - Fast baseline model
- **llama-3.2-1b** (1B params, ~2.0GB) - Instruction-tuned Llama
- **llama-3.2-3b** (3B params, ~6.0GB) - Larger Llama model  
- **phi-3-mini** (3.8B params, ~2.5GB) - Efficient Microsoft model
- **qwen2.5-3b** (3B params, ~6.0GB) - Alibaba's Qwen model
- **gemma-2b** (2B params, ~4.0GB) - Google's Gemma model

## 🚀 Quick Start

### Desktop GPU (RTX 3060)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/claude_mt_bench.git
cd claude_mt_bench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# 4. Run evaluation (quick test)
python -m src.cli --models gpt2-large llama-3.2-1b --max-questions 5

# 5. Full evaluation
python -m src.cli --models gpt2-large llama-3.2-1b phi-3-mini
```

### Google Colab

1. Open `notebooks/mtbench_colab.ipynb` in Google Colab
2. Select GPU runtime (T4 recommended)
3. Run all cells and follow the interactive setup
4. Enter your OpenAI API key when prompted

## 📚 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [Memory Optimization](#memory-optimization)
- [Results Analysis](#results-analysis)
- [Development](#development)
- [API Reference](#api-reference)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (RTX 3060 or better)
- OpenAI API key for judging

### Desktop Installation

```bash
# Clone repository
git clone https://github.com/your-username/claude_mt_bench.git
cd claude_mt_bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements.txt[dev]
```

### Google Colab Installation

The Colab notebook handles installation automatically. Simply run the first few cells to install dependencies.

## Usage

### Command Line Interface

```bash
# List available models
python -m src.cli --list-models

# Evaluate single model (quick test)
python -m src.cli --models gpt2-large --max-questions 10

# Evaluate multiple models
python -m src.cli --models gpt2-large llama-3.2-1b phi-3-mini

# Adjust memory limit for different GPUs
python -m src.cli --models llama-3.2-3b --memory-limit 8.0

# Full evaluation with custom output
python -m src.cli --models gpt2-large llama-3.2-1b --output-dir my_results

# Use different judge models
python -m src.cli --models gpt2-large --judge-model gpt-5-mini
python -m src.cli --models gpt2-large --judge-model gpt-4o-mini
```

### Jupyter Notebook

```python
from src.evaluation.mtbench_evaluator import MTBenchEvaluator

evaluator = MTBenchEvaluator(
    model_names=['gpt2-large', 'llama-3.2-1b'],
    max_questions=10  # For testing
)

results = await evaluator.run_full_evaluation()
evaluator.export_results(results, "my_results")
```

## System Architecture

```
claude_mt_bench/
├── src/                      # Main source code
│   ├── models/              # Model management
│   │   ├── model_manager.py # Memory-optimized model loading
│   │   └── model_configs.py # Model configurations
│   ├── evaluation/          # Evaluation pipeline
│   │   ├── mtbench_evaluator.py  # Main evaluator
│   │   ├── judge_client.py       # GPT-4.1-nano judge
│   │   └── conversation_handler.py # Multi-turn conversations
│   └── utils/               # Utilities
│       ├── memory_utils.py  # GPU memory monitoring
│       ├── data_loader.py   # MT-bench data loading
│       └── results_analyzer.py # Results analysis
├── tests/                   # Unit tests
├── configs/                 # Configuration files
├── notebooks/               # Jupyter notebooks
│   └── mtbench_colab.ipynb # Google Colab notebook
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Key Components

- **ModelManager**: Handles memory-optimized model loading with Flash Attention 2
- **JudgeClient**: Manages OpenAI API calls with rate limiting  
- **ConversationHandler**: Manages multi-turn conversation state
- **MTBenchEvaluator**: Orchestrates the complete evaluation pipeline
- **MemoryMonitor**: Tracks GPU memory usage and prevents OOM errors

## Configuration

### Model Configuration

Edit `configs/model_config.yaml` to adjust model settings:

```yaml
models:
  gpt2-large:
    estimated_memory_gb: 1.5
    max_new_tokens: 512
    temperature: 0.7
    use_flash_attention: false
    quantization: false
```

### Evaluation Configuration  

Edit `configs/evaluation_config.yaml` for evaluation settings:

```yaml
judge:
  model: "gpt-4.1-nano"
  temperature: 0.2
  rate_limit_requests_per_second: 10

evaluation:
  turns_per_question: 2
  generation_timeout_seconds: 120
  cleanup_between_models: true
```

## Memory Optimization

### RTX 3060 Optimization

The system is specifically optimized for RTX 3060's 6GB VRAM:

- **Flash Attention 2**: Reduces memory usage by ~30%
- **fp16 Precision**: Halves memory requirements
- **Gradient Checkpointing**: Trades compute for memory
- **Model Quantization**: 4-bit quantization for larger models
- **Memory Monitoring**: Prevents OOM errors with cleanup

### Memory Usage by Model

| Model | Memory (fp16) | Memory (4-bit) | RTX 3060 Compatible |
|-------|---------------|----------------|-------------------|
| gpt2-large | 1.5GB | N/A | ✅ |
| llama-3.2-1b | 2.0GB | 1.2GB | ✅ |
| phi-3-mini | 2.5GB | 1.5GB | ✅ |
| qwen2.5-3b | 6.0GB | 3.5GB | ⚠️ (tight fit) |
| gemma-2b | 4.0GB | 2.5GB | ✅ |

### Memory Monitoring

```python
from src.utils.memory_utils import MemoryMonitor

monitor = MemoryMonitor(memory_limit_gb=6.0)
monitor.log_memory_usage("Before model load")

# Memory usage automatically tracked
model = model_manager.load_model("llama-3.2-1b")
monitor.log_memory_usage("After model load")
```

## Results Analysis

### Output Formats

The system generates results in multiple formats:

- **JSON**: Complete detailed results with metadata
- **CSV**: Tabular data for spreadsheet analysis  
- **Summary**: Human-readable performance report
- **Visualizations**: Performance charts and heatmaps (in Colab)

### Sample Results

```json
{
  "model_results": [
    {
      "model_name": "llama-3.2-1b",
      "overall_score": {"mean": 7.2, "std": 1.8},
      "category_breakdown": {
        "writing": {"mean": 7.5},
        "reasoning": {"mean": 6.8},
        "coding": {"mean": 7.1}
      }
    }
  ],
  "comparison": {
    "overall_ranking": [
      {"model": "llama-3.2-1b", "score": 7.2},
      {"model": "gpt2-large", "score": 6.4}
    ]
  }
}
```

### Performance Metrics

- **Response Quality**: Scores from 1-10 by GPT-4.1-nano judge
- **Generation Speed**: Average time per response  
- **Memory Efficiency**: Peak GPU memory usage
- **Category Analysis**: Performance across 8 MT-bench categories
- **Turn Consistency**: Comparison between first and second turns

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test module
python -m pytest tests/test_models/test_model_configs.py -v
```

### Code Quality

```bash
# Format code
black src/

# Lint code  
ruff check src/ --fix

# Type checking
mypy src/
```

### Adding New Models

1. Add model configuration to `src/models/model_configs.py`:

```python
"my-model": ModelConfig(
    model_path="organization/my-model",
    model_family="custom",
    estimated_memory_gb=3.0,
    # ... other settings
)
```

2. Update prompt template if needed
3. Test memory usage with your GPU
4. Add unit tests

## API Reference

### MTBenchEvaluator

Main evaluation class orchestrating the complete pipeline.

```python
evaluator = MTBenchEvaluator(
    model_names=['gpt2-large', 'llama-3.2-1b'],
    openai_api_key="your-api-key",
    judge_model="gpt-5-mini",  # Default judge model
    memory_limit_gb=6.0,
    max_questions=None  # Use all 80 questions
)

# Run evaluation
results = await evaluator.run_full_evaluation()

# Export results
evaluator.export_results(results, "output_dir")
```

### ModelManager

Handles memory-optimized model loading and inference.

```python
from src.models.model_manager import ModelManager

manager = ModelManager(memory_limit_gb=6.0)
model, tokenizer = manager.load_model("llama-3.2-1b")
response = manager.generate_response(prompt)
```

### JudgeClient

Manages OpenAI judge models with adaptive rate limiting and GPT-5 support.

```python
from src.evaluation.judge_client import JudgeClient

# Use different judge models
judge_gpt5 = JudgeClient(api_key="your-key", model="gpt-5-mini")
judge_gpt4o = JudgeClient(api_key="your-key", model="gpt-4o-mini") 
judge_legacy = JudgeClient(api_key="your-key", model="gpt-4.1-nano")

score = await judge_gpt5.judge_response(
    question="What is AI?",
    answer="AI is artificial intelligence...",
    question_id=1,
    turn=1,
    model_name="llama-3.2-1b"
)
```

**Supported Judge Models:**
- `gpt-5-mini` (default) - Latest model with 30 req/s
- `gpt-4o-mini` - Cost-effective with 50 req/s  
- `gpt-4o-mini-2024-07-18` - Specific version
- `gpt-4.1-nano` - Legacy option with 10 req/s
- `gpt-4-turbo` - High performance with 20 req/s
- `gpt-4-turbo-2024-04-09` - Specific version
- `gpt-4o-2024-05-13` - Specific GPT-4o version
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce memory limit
python -m src.cli --models gpt2-large --memory-limit 4.0

# Use smaller models
python -m src.cli --models gpt2-large phi-3-mini
```

**OpenAI API Rate Limits**
```bash
# The system automatically handles rate limiting
# Ensure your API key has sufficient quota
export OPENAI_API_KEY="your-key-with-quota"
```

**Flash Attention Installation Issues**
```bash
# Install without build isolation
pip install flash-attn --no-build-isolation

# Or fall back to standard attention (automatic)
```

### Performance Tips

- **Start Small**: Use `--max-questions 5` for testing
- **Monitor Memory**: Watch GPU memory usage in logs
- **Batch Evaluation**: Evaluate multiple models at once
- **Use SSD**: Store data on fast storage for better performance

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the code style
4. **Add tests** for new functionality
5. **Run tests**: `python -m pytest tests/ -v`
6. **Submit a pull request**

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public methods
- Maintain >90% test coverage
- Use descriptive variable names

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{mtbench_evaluation_system,
  title={MT-bench Evaluation System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/claude_mt_bench}
}
```

## Acknowledgments

- **MT-bench**: Based on the evaluation framework from [lm-sys/FastChat](https://github.com/lm-sys/FastChat)
- **Flash Attention**: Memory optimization from [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **Transformers**: Built on [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Resources

- [MT-bench Paper](https://arxiv.org/abs/2306.05685) - Original MT-bench publication
- [FastChat Repository](https://github.com/lm-sys/FastChat) - Official MT-bench implementation
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Memory-efficient attention mechanism
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model loading and inference
- [OpenAI API](https://platform.openai.com/docs) - Judge API documentation

---

**Built with ❤️ for the AI research community**