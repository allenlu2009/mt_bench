#!/usr/bin/env python3
"""
Test runner script for MT-bench evaluation system.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --verbose    # Verbose output
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import pytest
        print("✅ pytest available")
        
        # Try importing torch (optional for some tests)
        try:
            import torch
            print("✅ torch available")
        except ImportError:
            print("⚠️  torch not available (some tests may be skipped)")
        
        # Try importing transformers (optional for some tests)    
        try:
            import transformers
            print("✅ transformers available")
        except ImportError:
            print("⚠️  transformers not available (some tests may be skipped)")
            
        return True
    except ImportError as e:
        print(f"❌ Missing pytest: {e}")
        print("💡 Run: uv pip install pytest pytest-asyncio pytest-cov")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run MT-bench tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('tests').exists():
        print("❌ tests/ directory not found. Run from project root.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test selection
    if args.unit:
        cmd.extend(['tests/test_models/'])
        # Only add utils tests if torch is available
        try:
            import torch
            cmd.extend(['tests/test_utils/'])
        except ImportError:
            print("⚠️  Skipping tests/test_utils/ (torch not available)")
    elif args.integration:
        cmd.extend(['tests/test_evaluation/'])
    else:
        # Run all tests but ignore torch-dependent ones if torch not available
        cmd.append('tests/')
        try:
            import torch
        except ImportError:
            cmd.extend(['--ignore=tests/test_utils/test_memory_utils.py'])
            print("⚠️  Ignoring torch-dependent tests")
    
    # Add flags
    if args.verbose:
        cmd.append('-v')
    
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    # Add other useful flags
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker checking
        '--durations=10',  # Show 10 slowest tests
    ])
    
    # Run tests
    success = run_command(cmd, "Running MT-bench tests")
    
    if success:
        print("\n🎉 All tests passed successfully!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print("\n💥 Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()