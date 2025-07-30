# MT-bench CLI Usage Examples

## 🚀 Getting Started

First, activate your environment and set up your API key:

```bash
# Activate the uv virtual environment
source .venv/bin/activate

# Set your OpenAI API key (required for GPT-4.1-nano judge)
export OPENAI_API_KEY="your-api-key-here"
```

## 📋 Basic Commands

### List Available Models
```bash
# See all available models with their memory requirements
python -m src.cli --list-models
```

**Expected Output:**
```
Available Models:
--------------------------------------------------------------------------------
Model Name           Family     Memory (GB)  Model Path
--------------------------------------------------------------------------------
gpt2-large           gpt2       1.5          gpt2-large
llama-3.2-1b         llama      2.0          meta-llama/Llama-3.2-1B
llama-3.2-3b         llama      6.0          meta-llama/Llama-3.2-3B
phi-3-mini           phi        2.5          microsoft/Phi-3-mini-4k-instruct
qwen2.5-3b           qwen       6.0          Qwen/Qwen2.5-3B-Instruct
gemma-2b             gemma      4.0          google/gemma-2b-it

Total models: 6
```

## 🧪 Testing and Development

### Quick Test (Single Model, Limited Questions)
```bash
# Test with GPT-2 Large (fastest model) and only 5 questions
python -m src.cli --models gpt2-large --max-questions 5 --verbose
```

### Test Multiple Small Models
```bash
# Test 3 smaller models that fit in RTX 3060 memory
python -m src.cli --models gpt2-large llama-3.2-1b phi-3-mini --max-questions 10
```

## 🖥️ RTX 3060 Production Usage

### Single Model Evaluation
```bash
# Evaluate GPT-2 Large (most memory efficient)
python -m src.cli --models gpt2-large

# Evaluate Llama 3.2 1B (good balance of size/performance)
python -m src.cli --models llama-3.2-1b

# Evaluate Phi-3 Mini (optimized 3.8B model)
python -m src.cli --models phi-3-mini
```

### Multiple Model Comparison
```bash
# Compare all models that fit in 6GB (RTX 3060 default)
python -m src.cli --models gpt2-large llama-3.2-1b phi-3-mini

# Include larger models (will be evaluated sequentially)
python -m src.cli --models gpt2-large llama-3.2-1b llama-3.2-3b phi-3-mini qwen2.5-3b
```

## 🔧 Advanced Configuration

### Custom Memory Limits
```bash
# For RTX 3070/3080 (8GB)
python -m src.cli --models llama-3.2-3b qwen2.5-3b --memory-limit 8.0

# For RTX 4090 (24GB) - can run larger models
python -m src.cli --models llama-3.2-3b qwen2.5-3b --memory-limit 20.0

# Conservative memory usage (4GB)
python -m src.cli --models gpt2-large llama-3.2-1b --memory-limit 4.0
```

### Custom Judge Model
```bash
# Use different judge model
python -m src.cli --models gpt2-large --judge-model gpt-4-turbo

# Use GPT-3.5 for faster/cheaper judging
python -m src.cli --models gpt2-large --judge-model gpt-3.5-turbo
```

### Custom Directories
```bash
# Custom cache and output directories
python -m src.cli --models gpt2-large \
    --cache-dir ./my_cache \
    --output-dir ./my_results

# Force re-download of MT-bench data
python -m src.cli --models gpt2-large --force-download
```

## 📊 Complete Production Runs

### Full MT-bench Evaluation (RTX 3060)
```bash
# Complete evaluation of all compatible models (will take several hours)
python -m src.cli \
    --models gpt2-large llama-3.2-1b llama-3.2-3b phi-3-mini qwen2.5-3b \
    --output-dir results/full_evaluation_$(date +%Y%m%d) \
    --verbose
```

### Comprehensive Comparison Study
```bash
# Compare different model families with detailed logging
python -m src.cli \
    --models gpt2-large llama-3.2-1b phi-3-mini qwen2.5-3b \
    --judge-model gpt-4.1-nano \
    --memory-limit 6.0 \
    --output-dir results/model_comparison \
    --cache-dir data/mtbench_cache \
    --verbose
```

## 🚨 Error Handling Examples

### Check Model Availability
```bash
# This will show warning and skip unavailable models
python -m src.cli --models gpt2-large nonexistent-model llama-3.2-1b
```

### Memory Limit Validation
```bash
# This will skip models that exceed memory limit
python -m src.cli --models llama-3.2-3b qwen2.5-3b --memory-limit 4.0
```

### Missing API Key
```bash
# This will show error message about missing API key
unset OPENAI_API_KEY
python -m src.cli --models gpt2-large
```

## 🧩 Integration with Other Tools

### Save Results for Analysis
```bash
# Generate timestamped results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -m src.cli \
    --models gpt2-large llama-3.2-1b \
    --output-dir results/experiment_${TIMESTAMP} \
    --max-questions 20

# Results will be saved as:
# results/experiment_20241129_143022/
#   ├── detailed_results.json
#   ├── summary_results.csv
#   └── evaluation_summary.txt
```

### Batch Processing Script
```bash
#!/bin/bash
# batch_evaluation.sh

# Activate environment
source .venv/bin/activate

# Set API key
export OPENAI_API_KEY="your-key-here"

# Test different memory limits
for memory in 4.0 6.0 8.0; do
    echo "Testing with ${memory}GB memory limit..."
    python -m src.cli \
        --models gpt2-large llama-3.2-1b phi-3-mini \
        --memory-limit ${memory} \
        --max-questions 10 \
        --output-dir results/memory_${memory}gb
done
```

## 📈 Expected Output

### During Execution
```
2024-11-29 14:30:15 - __main__ - INFO - Applied RTX 3060 optimizations
2024-11-29 14:30:15 - __main__ - INFO - Validated models: ['gpt2-large', 'llama-3.2-1b']
2024-11-29 14:30:16 - mtbench_evaluator - INFO - Loading MT-bench dataset...
2024-11-29 14:30:18 - mtbench_evaluator - INFO - Starting evaluation for gpt2-large
[Model Loading] GPU: 1.2GB (20.0%), Cached: 0.1GB, System: 45.2%
[Generation Turn 1] GPU: 1.8GB (30.0%), Cached: 0.2GB, System: 48.1%
...
```

### Final Results
```
================================================================================
EVALUATION COMPLETED SUCCESSFULLY
================================================================================

MT-bench Evaluation Summary
===========================

Model Scores (Average across all categories):
• gpt2-large: 6.2/10
• llama-3.2-1b: 7.1/10

Category Breakdown:
• Writing: gpt2-large=5.8, llama-3.2-1b=7.3
• Roleplay: gpt2-large=6.1, llama-3.2-1b=6.9
• Reasoning: gpt2-large=6.7, llama-3.2-1b=7.4
...

Results saved to: results/

Evaluation Statistics:
  Models evaluated: 2/2
  Total responses judged: 320
  Peak memory usage: 2.1GB
```

## 💡 Tips and Best Practices

1. **Start Small**: Always test with `--max-questions 5` first
2. **Monitor Memory**: Use `--verbose` to track GPU memory usage
3. **Sequential Execution**: Large models are automatically evaluated one at a time
4. **Save Results**: Always specify `--output-dir` for organized results
5. **API Limits**: GPT-4.1-nano judge respects rate limits automatically
6. **Interrupt Safety**: Use Ctrl+C to safely stop evaluation