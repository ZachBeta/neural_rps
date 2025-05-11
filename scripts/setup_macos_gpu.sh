#!/bin/bash

# setup_macos_gpu.sh - Sets up environment for GPU development on macOS
# Usage: source ./scripts/setup_macos_gpu.sh

echo "Setting up environment for GPU development on macOS..."

# Check if homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Please install it first."
    echo "Visit https://brew.sh/ for installation instructions."
    return 1
fi

# Check if tensorflow is installed
if ! brew list tensorflow &> /dev/null; then
    echo "Installing TensorFlow via Homebrew..."
    brew install tensorflow
else
    echo "TensorFlow is already installed via Homebrew."
fi

# Set environment variables
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/lib
export CPATH=$CPATH:/opt/homebrew/include
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/lib

echo "Environment variables set:"
echo "LIBRARY_PATH=$LIBRARY_PATH"
echo "CPATH=$CPATH" 
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

# Install TensorFlow Go bindings if not already installed
if ! go list -m github.com/tensorflow/tensorflow &> /dev/null; then
    echo "Installing TensorFlow Go bindings v2.12.0..."
    go get github.com/tensorflow/tensorflow/tensorflow/go@v2.12.0+incompatible
else
    echo "TensorFlow Go bindings are already installed."
    echo "Updating to v2.12.0..."
    go get github.com/tensorflow/tensorflow/tensorflow/go@v2.12.0+incompatible
fi

echo ""
echo "Setup complete! You can now build with GPU support:"
echo "  ./scripts/build.sh --gpu"
echo ""
echo "Note: This script must be sourced, not executed, to set environment variables:"
echo "  source ./scripts/setup_macos_gpu.sh" 