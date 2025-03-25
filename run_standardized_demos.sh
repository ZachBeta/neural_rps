#!/bin/bash

# run_standardized_demos.sh
# Run all implementations with standardized output format

set -e  # Exit on any error

# Ensure we're in the project root
cd "$(dirname "$0")"

echo "==========================================="
echo "Running Neural RPS demos with standardized output format"
echo "==========================================="

# Build all implementations
echo "Building all implementations..."
make build

# Run Legacy C++ implementation
echo ""
echo "Running Legacy C++ implementation..."
cd legacy_cpp_implementation/build && ./src/legacy_neural_rps > ../../legacy_cpp_demo_output.txt
cd ../..
echo "Legacy C++ output written to legacy_cpp_demo_output.txt"

# Run C++ implementation
echo ""
echo "Running C++ implementation..."
cd cpp_implementation/build && ./neural_rps_demo > ../../cpp_demo_output.txt
cd ../..
echo "C++ output written to cpp_demo_output.txt"

# Run Golang implementation
echo ""
echo "Running Golang implementation..."
cd golang_implementation && ./bin/neural_rps > ../go_demo_output.txt
cd ..
echo "Golang output written to go_demo_output.txt"

# Run AlphaGo demo
echo ""
echo "Running AlphaGo demo..."
cd alphago_demo && ./tictactoe > ../alphago_demo_output.txt
cd ..
echo "AlphaGo demo output written to alphago_demo_output.txt"

# Validate outputs
echo ""
echo "Validating output formats..."
python3 validate_output_format.py

echo ""
echo "Demos completed. All output files follow the standardized format."
echo "You can now compare the outputs using:"
echo "  diff -y --suppress-common-lines cpp_demo_output.txt go_demo_output.txt | less"
echo "  diff -y --suppress-common-lines legacy_cpp_demo_output.txt cpp_demo_output.txt | less"
echo ""
echo "Or view them side by side using:"
echo "  paste -d '|' cpp_demo_output.txt go_demo_output.txt | less"
echo "" 