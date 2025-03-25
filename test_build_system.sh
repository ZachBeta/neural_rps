#!/bin/bash
# Test script for the Neural RPS build system

set -e  # Exit on error

# Function to print section headers
print_header() {
  echo ""
  echo "========================================="
  echo "  $1"
  echo "========================================="
  echo ""
}

# Function to run a make target and check success
run_target() {
  print_header "Running: make $1"
  make $1
  if [ $? -eq 0 ]; then
    echo "✅ Success: $1"
  else
    echo "❌ Failed: $1"
    exit 1
  fi
}

# Function to check if directory exists
check_directory() {
  if [ -d "$1" ]; then
    echo "✅ Directory exists: $1"
  else
    echo "❌ Directory missing: $1"
    exit 1
  fi
}

# Function to check if file exists
check_file() {
  if [ -f "$1" ]; then
    echo "✅ File exists: $1"
  else
    echo "❌ File missing: $1"
    exit 1
  fi
}

# Output the current directory
print_header "Starting test in directory $(pwd)"

# Clean everything first
run_target "clean"

# Test build targets
run_target "build-alphago"
run_target "build-go"
run_target "build-cpp"

# Check directories and binaries
print_header "Checking directories and binaries"
check_directory "alphago_demo/bin"
check_directory "golang_implementation/bin"
check_directory "cpp_implementation/build"

# Test documentation generation
print_header "Testing documentation generation"
run_target "doc" &
DOC_PID=$!
sleep 3
kill $DOC_PID

# Test AlphaGo quick training
print_header "Testing AlphaGo model training (quick mode)"
run_target "alphago-quick-train"

# Verify model files were created
print_header "Checking model output files"
check_file "alphago_demo/output/rps_policy1.model"
check_file "alphago_demo/output/rps_value1.model"
check_file "alphago_demo/output/rps_policy2.model"
check_file "alphago_demo/output/rps_value2.model"

# Test tournament functionality with minimal games
print_header "Testing tournament functionality (minimal games)"
cd golang_implementation
mkdir -p bin results
go build -o bin/tournament cmd/tournament/main.go
./bin/tournament --games 2 --alphago-sims 10 --ppo-hidden 64 > results/test_results.txt
check_file "results/test_results.txt"
cd ..

# Final cleanup
print_header "Final cleanup"
run_target "clean"

print_header "Build System Test Summary"
echo "✅ All tests completed successfully!"
echo "You can now use the build system as documented in docs/build_system.md"
echo ""
echo "Try these example commands:"
echo "  make build               - Build all implementations"
echo "  make alphago-train       - Train AlphaGo models (or make alphago-quick-train for faster testing)"
echo "  make golang-tournament   - Run a tournament between agents"
echo "  make run-alphago-rps     - Play the RPS card game"
echo "" 