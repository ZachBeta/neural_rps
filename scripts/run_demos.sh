#!/bin/bash
# Run demos for all implementations and generate output files

echo "============================================="
echo "Neural RPS Implementation Demos"
echo "============================================="
echo "This script will run demos of all implementations"
echo "and generate output files for comparison."
echo

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Set up environment variables for C++ implementation
export EIGEN_DIR="/opt/homebrew/include/eigen3"
export CPLUS_INCLUDE_PATH="$EIGEN_DIR:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/lib:$LD_LIBRARY_PATH"
echo "C++ environment set up:"
echo "EIGEN_DIR=$EIGEN_DIR"

# Build all implementations
echo "Building all implementations..."
make build

# Legacy C++ Demo
echo
echo "============================================="
echo "Running Legacy C++ Neural RPS Implementation"
echo "============================================="
echo "Output will be saved to legacy_cpp_demo_output.txt"
echo "Note: This may take a while as it runs a full training session."
echo
cd legacy_cpp_implementation/build && ./src/legacy_neural_rps || echo "Legacy C++ demo failed to run. Please check your environment."
cd ../..

# C++ Demo
echo
echo "============================================="
echo "Running C++ Neural RPS Simplified Demo"
echo "============================================="
echo "Output will be saved to cpp_demo_output.txt"
echo
cd cpp_implementation/build && ./neural_rps_demo || echo "C++ demo failed to run. Please check your environment."
cd ../..

echo
echo "Note: The full C++ neural implementation can also be run with:"
echo "cd cpp_implementation/build && ./neural_rps_full"
echo "or using: make run-cpp-full"
echo

# Go Demo
echo
echo "============================================="
echo "Running Go Neural RPS Demo"
echo "============================================="
echo "Output will be saved to go_demo_output.txt"
echo
cd golang_implementation && ./bin/neural_rps
cd ..

# AlphaGo Demo
echo
echo "============================================="
echo "Running AlphaGo-style Tic-Tac-Toe Demo"
echo "============================================="
echo "Output will be saved to alphago_demo_output.txt"
echo
cd alphago_demo && ./run.sh
cd ..

echo
echo "============================================="
echo "Demo Summary"
echo "============================================="
echo "All demos completed. Output files in project root:"

# Check which files were successfully generated
if [ -f "legacy_cpp_demo_output.txt" ]; then
  echo "✅ legacy_cpp_demo_output.txt - Legacy C++ implementation output"
else
  echo "❌ legacy_cpp_demo_output.txt - NOT GENERATED (error during execution?)"
fi

if [ -f "cpp_demo_output.txt" ]; then
  echo "✅ cpp_demo_output.txt - Simplified C++ demo output"
else
  echo "❌ cpp_demo_output.txt - NOT GENERATED (error during execution?)"
fi

if [ -f "go_demo_output.txt" ]; then
  echo "✅ go_demo_output.txt - Go implementation output"
else
  echo "❌ go_demo_output.txt - NOT GENERATED (error during execution?)"
fi

if [ -f "alphago_demo_output.txt" ]; then
  echo "✅ alphago_demo_output.txt - AlphaGo-style demo output"
else
  echo "❌ alphago_demo_output.txt - NOT GENERATED (error during execution?)"
fi

echo
echo "You can view and compare these files to see how each"
echo "implementation handles neural networks differently."
echo "=============================================" 