#!/usr/bin/env python3
"""
Demo script showing MT-bench CLI functionality.

This script demonstrates various CLI commands without actually running evaluations
(which would require OpenAI API key and take significant time).
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and show output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        # Ensure we're in the right directory and environment
        env = os.environ.copy()
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent,
            capture_output=True, 
            text=True,
            env=env
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("❌ ERROR")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
                
    except Exception as e:
        print(f"❌ Exception: {e}")


def main():
    """Main demo function."""
    print("🚀 MT-bench CLI Demonstration")
    print("This demo shows CLI functionality without running full evaluations")
    
    # Check if we're in a virtual environment
    if not os.environ.get('VIRTUAL_ENV') and not sys.prefix != sys.base_prefix:
        print("\n⚠️  Warning: Not in virtual environment!")
        print("Please run: source .venv/bin/activate")
        return
    
    # Demo commands
    commands = [
        {
            "cmd": ["python", "-m", "src.cli", "--help"],
            "desc": "Show CLI help and usage examples"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--list-models"],
            "desc": "List all available models with memory requirements"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--models", "gpt2-large"],
            "desc": "Test model validation (should show warning about invalid model)"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--models", "llama-3.2-3b", "--memory-limit", "4.0"],
            "desc": "Test memory limit validation (should warn about insufficient memory)"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--mode", "pairwise", "--models", "gpt2-large"],
            "desc": "Test pairwise mode validation (should require at least 2 models)"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--mode", "pairwise", "--models", "gpt2-large", "llama-3.2-1b"],
            "desc": "Test pairwise mode with 2 models (should show API key requirement)"
        },
        {
            "cmd": ["python", "-m", "src.cli", "--mode", "both", "--models", "gpt2-large", "llama-3.2-1b", "phi-3-mini"],
            "desc": "Test combined mode with 3 models (should show API key requirement)"
        }
    ]
    
    # Run demo commands
    for cmd_info in commands:
        run_command(cmd_info["cmd"], cmd_info["desc"])
    
    print(f"\n{'='*60}")
    print("📋 To run actual evaluations, you need:")
    print("1. Set OpenAI API key: export OPENAI_API_KEY='your-key-here'")
    print("2. Choose your evaluation mode:")
    print("")
    
    print("🎯 SINGLE MODEL EVALUATION (Traditional Absolute Scoring):")
    print("   # Quick test")
    print("   python -m src.cli --models gpt2-large --max-questions 2")
    print("   # Full evaluation")
    print("   python -m src.cli --models gpt2-large llama-3.2-1b")
    print("")
    
    print("⚔️  PAIRWISE COMPARISON (Head-to-Head Battle):")
    print("   # Compare 2 models directly")
    print("   python -m src.cli --mode pairwise --models gpt2-large llama-3.2-1b --max-questions 2")
    print("   # Tournament style with 3 models")
    print("   python -m src.cli --mode pairwise --models gpt2-large llama-3.2-1b phi-3-mini --max-questions 2")
    print("")
    
    print("🔥 COMBINED EVALUATION (Both Absolute + Pairwise):")
    print("   # Best of both worlds")
    print("   python -m src.cli --mode both --models gpt2-large llama-3.2-1b --max-questions 2")
    print("   # Full tournament")
    print("   python -m src.cli --mode both --models gpt2-large llama-3.2-1b phi-3-mini")
    print("")
    
    print("💾 RESPONSE CACHING OPTIONS:")
    print("   # Use cached responses (default)")
    print("   python -m src.cli --mode pairwise --models model1 model2")
    print("   # Always generate fresh responses")
    print("   python -m src.cli --mode pairwise --models model1 model2 --disable-response-cache")
    print("   # Custom cache directory")
    print("   python -m src.cli --mode both --models model1 model2 --response-cache-dir my_cache")
    print("")
    
    print("📊 MEMORY OPTIMIZATION:")
    print("   # For RTX 3060 (6GB) - default")
    print("   python -m src.cli --mode pairwise --models gpt2-large llama-3.2-1b")
    print("   # For higher-end GPUs")
    print("   python -m src.cli --mode pairwise --models llama-3.2-3b qwen2.5-3b --memory-limit 12.0")
    print("")
    
    print("🏆 EVALUATION MODES EXPLAINED:")
    print("   • single:   Traditional 1-10 scoring for each model independently")
    print("   • pairwise: Direct comparison, determines which model wins each question")
    print("   • both:     Runs both modes, reuses responses for efficiency")
    print("")
    
    print("💡 PRO TIPS:")
    print("   • Pairwise comparison provides more human-aligned rankings (80%+ agreement)")
    print("   • Use --max-questions 2 for quick testing")
    print("   • Response caching makes repeated evaluations much faster")
    print("   • Both modes use same memory-efficient sequential model loading")
    print("="*60)


if __name__ == "__main__":
    main()
