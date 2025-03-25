# Neural RPS - C++ Implementation

This directory contains two C++ implementations of the Neural Rock Paper Scissors project:
1. A simplified demonstration version (`neural_rps_demo`)
2. A full neural network implementation (`neural_rps_full`)

## Overview

The simplified demo provides a basic introduction to the project concepts and structure, while the full implementation contains a complete neural network that learns to play Rock Paper Scissors through reinforcement learning.

## Available Executables

- **neural_rps_demo**: A lightweight demonstration that shows the concept without actual training
- **neural_rps_full**: The complete implementation with neural network training (similar to the legacy implementation but with combined header files)

## Dependencies

This implementation requires:
- A C++17 compatible compiler
- Eigen3 library for linear algebra (matrix operations)

To install Eigen3:
- On macOS: `brew install eigen`
- On Ubuntu: `apt-get install libeigen3-dev`
- On Windows: Download from http://eigen.tuxfamily.org/

## Files
- `simple_demo.cpp` - The simplified demonstration implementation
- `main.cpp` - Main implementation file for the full neural network
- `NeuralNetwork.cpp` - Neural Network implementation
- `NeuralNetwork.hpp` - Neural Network header file
- `Environment.hpp` - Game environment
- `PPOAgent.hpp` - Proximal Policy Optimization agent
- `NetworkVisualizer.hpp` - Visualization utilities
- `CMakeLists.txt` - Build configuration for both implementations
- `.clang-tidy` - Clang-tidy configuration for code quality
- `cpp_demo_output.txt` - Sample output from the demo

## Building and Running

To build the C++ implementation:

```bash
cd cpp_implementation
mkdir build
cd build
cmake ..
make
```

To run the simplified demo:
```bash
./neural_rps_demo
```

To run the full neural network implementation (may take several minutes):
```bash
./neural_rps_full
```

From the project root, you can also use these make targets:
```bash
# Run the simple demo
make run-cpp

# Run the full neural network implementation
make run-cpp-full
```

## Implementation Approach

The simplified demo creates a text output explaining how the neural network would work. The full implementation actually trains a neural network using reinforcement learning with the following components:

- A game environment that simulates Rock Paper Scissors
- A neural network that learns from experience
- A PPO agent that optimizes the neural network's policy
- Visualization of training progress and game state

## Implementation Note

This implementation uses a more compact structure with header-only files for most components, compared to the legacy implementation which has separate header and source files. If you're interested in the original, more modular implementation, see the [`legacy_cpp_implementation`](../legacy_cpp_implementation) directory.

## Known Issues

The C++ implementation may show build warnings and requires external dependencies. This implementation is kept for reference, but the Golang implementation is preferred for development.

The C++ implementation provides the core neural network functionality but with a simpler interface than the Go implementation. 