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
echo "1. Running C++ implementation (press Ctrl+C to stop)..."
make run-cpp

echo ""
echo "2. Running Golang implementation (press Ctrl+C to stop)..."
make run-go

echo ""
echo "3. Running AlphaGo demo (press Ctrl+C to stop)..."
make run-alphago

echo ""
echo "===== All Done! =====" 