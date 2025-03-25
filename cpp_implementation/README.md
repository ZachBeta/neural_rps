# Neural RPS - C++ Implementation

This directory contains the C++ implementation of the Neural Rock Paper Scissors project.

## Overview

This is the original C++ implementation of a neural network that learns to play Rock Paper Scissors.

## Dependencies

This implementation requires:
- A C++17 compatible compiler
- Eigen3 library for linear algebra (matrix operations)

To install Eigen3:
- On macOS: `brew install eigen`
- On Ubuntu: `apt-get install libeigen3-dev`
- On Windows: Download from http://eigen.tuxfamily.org/

## Files
- `main.cpp` - Main implementation file
- `NeuralNetwork.cpp` - Neural Network implementation
- `Environment.hpp` - Game environment
- `PPOAgent.hpp` - Proximal Policy Optimization agent
- `NetworkVisualizer.hpp` - Visualization utilities
- `CMakeLists.txt` - Build configuration for the C++ implementation
- `.clang-tidy` - Clang-tidy configuration for code quality

## Building

To build the C++ implementation:

```bash
cd cpp_implementation
mkdir build
cd build
cmake ..
make
```

## Known Issues

The C++ implementation may show build warnings and requires external dependencies. This implementation is kept for reference, but the Golang implementation is preferred for development.

The C++ implementation provides the core neural network functionality but with a simpler interface than the Go implementation. 