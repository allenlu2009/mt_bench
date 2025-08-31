#!/bin/bash

# setup-uv.sh - Cross-platform UV configuration script

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        PLATFORM="darwin-arm64"
        PLATFORM_NAME="mac-arm64"
    else
        PLATFORM="darwin-x86_64" 
        PLATFORM_NAME="mac-x86_64"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux-x86_64"
    PLATFORM_NAME="cuda"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows-x86_64"
    PLATFORM_NAME="windows-cuda"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Detected platform: $PLATFORM_NAME"

# Set up global UV cache directory
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    UV_CACHE_DIR="$USERPROFILE/.cache/uv"
    UV_VENV_DIR="$USERPROFILE/.venvs"
else
    UV_CACHE_DIR="$HOME/.cache/uv"
    UV_VENV_DIR="$HOME/.venvs"
fi

# Export UV environment variables
export UV_CACHE_DIR="$UV_CACHE_DIR"
echo "UV_CACHE_DIR=$UV_CACHE_DIR" >> ~/.bashrc  # or ~/.zshrc

# Create venv directory if it doesn't exist
mkdir -p "$UV_VENV_DIR"

# Project name (change this to your project name)
PROJECT_NAME="${1:-$(basename $(pwd))}"
VENV_PATH="$UV_VENV_DIR/$PROJECT_NAME-$PLATFORM_NAME"

echo "Setting up environment: $VENV_PATH"

# Check if project has Python requirements, otherwise default to 3.11
PYTHON_VERSION="3.11"
if [[ -f "pyproject.toml" ]]; then
    # Extract Python requirement from pyproject.toml
    REQUIRED_PYTHON=$(grep -E "requires-python.*=" pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/' | head -1)
    if [[ "$REQUIRED_PYTHON" =~ "3.13" ]]; then
        PYTHON_VERSION="3.13"
    elif [[ "$REQUIRED_PYTHON" =~ "3.12" ]]; then
        PYTHON_VERSION="3.12"
    fi
fi

echo "Using Python version: $PYTHON_VERSION (project requires: ${REQUIRED_PYTHON:-any})"

# Create virtual environment outside project directory
uv venv "$VENV_PATH" --python "$PYTHON_VERSION"

# Platform-specific PyTorch index URLs
case $PLATFORM_NAME in
    "cuda"|"windows-cuda")
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        ;;
    "mac-arm64"|"mac-x86_64")
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        ;;
esac

# Check available UV lock options and generate lock file accordingly
UV_VERSION=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "UV version: $UV_VERSION"

# Check various platform-related flags
if uv lock --help | grep -q "\--python-platform"; then
    echo "Generating lock file with --python-platform for: $PLATFORM"
    UV_EXTRA_INDEX_URL="$TORCH_INDEX" uv lock --python-platform "$PLATFORM" --index-strategy unsafe-best-match
elif uv lock --help | grep -q "\--platform"; then
    echo "Generating lock file with --platform for: $PLATFORM"
    UV_EXTRA_INDEX_URL="$TORCH_INDEX" uv lock --platform "$PLATFORM" --index-strategy unsafe-best-match
else
    echo "Generating standard lock file (platform-specific resolution will happen during sync)"
    echo "Note: Platform-specific packages will be resolved when you run sync on each platform."
    UV_EXTRA_INDEX_URL="$TORCH_INDEX" uv lock --index-strategy unsafe-best-match
fi

# Create activation script
cat > "activate-$PLATFORM_NAME.sh" << EOF
#!/bin/bash
# Activation script for $PLATFORM_NAME

export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_EXTRA_INDEX_URL="$TORCH_INDEX"

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source "$VENV_PATH/Scripts/activate"
else
    source "$VENV_PATH/bin/activate"
fi

echo "Activated environment: $PROJECT_NAME-$PLATFORM_NAME"
echo "UV Cache: $UV_CACHE_DIR"
EOF

chmod +x "activate-$PLATFORM_NAME.sh"

# Create platform-specific sync script
cat > "sync-$PLATFORM_NAME.sh" << EOF
#!/bin/bash
# Sync script for $PLATFORM_NAME

export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_EXTRA_INDEX_URL="$TORCH_INDEX"

# Activate environment and sync
source "./activate-$PLATFORM_NAME.sh"

# Install to the activated global venv (not local .venv)
# Use uv pip install to avoid markupsafe compatibility issues with uv sync
uv pip install -e . --index-strategy unsafe-best-match

echo "Synced packages for $PLATFORM_NAME"
echo "PyTorch index used: $TORCH_INDEX"
EOF

chmod +x "sync-$PLATFORM_NAME.sh"

echo "Setup complete!"
echo "Usage:"
echo "  1. Run sync: ./sync-$PLATFORM_NAME.sh"
echo "  2. Activate: source ./activate-$PLATFORM_NAME.sh"
echo ""
echo "Global cache location: $UV_CACHE_DIR"
echo "Virtual environment: $VENV_PATH"
