#!/bin/bash
# Set up a local Python environment with uv for Apple Silicon GPU acceleration

set -e

# Directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    uv venv "$SCRIPT_DIR/venv"
fi

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv pip install --upgrade pip
uv pip install -r "$SCRIPT_DIR/requirements.txt"

# Generate gRPC code
echo "Generating Python gRPC code..."
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/generate_grpc.sh"

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source $SCRIPT_DIR/venv/bin/activate"
echo ""
echo "To start the neural service, run:"
echo "python $SCRIPT_DIR/neural_service.py"
echo ""
echo "For Apple Silicon GPU acceleration, make sure that:"
echo "1. TensorFlow is installed with Metal support (should be automatic)"
echo "2. Python is running natively (not under Rosetta)" 