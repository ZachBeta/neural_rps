# Legacy C++ Implementation - Neural RPS

This directory contains the original, fully-functional C++ implementation of the Neural Rock Paper Scissors project.

## Overview

This implementation uses a clean, modular structure with separate header and source files. It features:

- A neural network that learns to play Rock Paper Scissors through self-play
- Proximal Policy Optimization (PPO) reinforcement learning algorithm
- Training visualization and performance metrics
- Unit tests for core components

## Directory Structure

- `include/` - Header files
- `src/` - Implementation files
- `tests/` - Unit tests

## Building and Running

```bash
# Create a build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run the main application
./src/legacy_neural_rps

# Run the tests (if GTest is installed)
ctest
```

## Dependencies

- C++17 compiler
- CMake 3.14+
- Eigen3 (for matrix operations)
- GoogleTest (optional, for tests)

## Original Structure

This implementation preserves the original structure and functionality of the Neural RPS project before it was split into multiple implementations. 