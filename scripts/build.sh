#!/bin/bash

# build.sh - Build script for Neural RPS
# Usage: ./scripts/build.sh [--gpu]

# Set default build mode
BUILD_MODE="cpu"
BUILD_TAGS=""
OUTPUT_DIR="bin"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) BUILD_MODE="gpu"; BUILD_TAGS="-tags=gpu"; ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Building in $BUILD_MODE mode..."

# Build all commands
for cmd in $(find cmd -mindepth 1 -maxdepth 1 -type d); do
    CMD_NAME=$(basename $cmd)
    echo "Building $CMD_NAME..."
    go build $BUILD_TAGS -o $OUTPUT_DIR/$CMD_NAME ./cmd/$CMD_NAME
done

echo "Build complete. Binaries are in the $OUTPUT_DIR directory."

# If GPU mode, print additional information
if [ "$BUILD_MODE" = "gpu" ]; then
    echo ""
    echo "GPU build notes:"
    echo "  - Make sure TensorFlow is properly installed"
    echo "  - On Apple Silicon, Metal Performance Shaders will be used"
    echo "  - On NVIDIA GPUs, CUDA and cuDNN must be installed"
fi 