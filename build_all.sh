#!/bin/bash
# Build and test all implementations

echo "===== Building and Testing Neural RPS Implementations ====="

# Build all implementations
echo "Building all implementations..."
make build

# Clean build output
echo ""
echo "===== Testing Implementations ====="

# Test Golang implementation
echo "Testing Golang implementation..."
make test-go

echo ""
echo "===== Running Implementations ====="

# Run each implementation
echo "1. Running Legacy C++ implementation (press Ctrl+C to stop)..."
make run-legacy-cpp

echo ""
echo "2. Running C++ simplified demo..."
make run-cpp

echo ""
echo "Note: You can also run the full C++ neural implementation with:"
echo "make run-cpp-full"

echo ""
echo "3. Running Golang implementation (press Ctrl+C to stop)..."
make run-go

echo ""
echo "4. Running AlphaGo demo (press Ctrl+C to stop)..."
make run-alphago

echo ""
echo "===== All Done! =====" 