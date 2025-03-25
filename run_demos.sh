#!/bin/bash
# Run demos for all implementations and generate output files

echo "============================================="
echo "Neural RPS Implementation Demos"
echo "============================================="
echo "This script will run demos of all three implementations"
echo "and generate output files for comparison."
echo

# Build all implementations
echo "Building all implementations..."
make build

# C++ Demo
echo
echo "============================================="
echo "Running C++ Neural RPS Demo"
echo "============================================="
echo "Output will be saved to cpp_demo_output.txt"
echo "(This may fail if Eigen library is not installed)"
echo
cd cpp_implementation/build && ./neural_rps
cd ../..

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
if [ -f "cpp_demo_output.txt" ]; then
  echo "✅ cpp_demo_output.txt - C++ implementation output"
else
  echo "❌ cpp_demo_output.txt - NOT GENERATED (Eigen library missing?)"
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