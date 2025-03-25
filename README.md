# Neural Rock Paper Scissors

A neural network-based implementation of Rock Paper Scissors using different approaches and languages.

## Overview

This project explores different implementations of neural networks for playing Rock Paper Scissors:

1. **Legacy C++ Implementation** - The original, fully-functional implementation with complete neural network that learns to play Rock Paper Scissors effectively
2. **C++ Implementation** - A simplified demonstration version, plus an integrated full neural network
3. **Golang Implementation** - An improved implementation with better readability, performance, and development setup
4. **AlphaGo-Style Demo** - A demonstration of AlphaGo-like techniques applied to Tic-Tac-Toe

## Features

- Neural network implementation with one hidden layer
- PPO (Proximal Policy Optimization) algorithm for reinforcement learning
- Game environment with state tracking
- Visualization of network architecture, weights, and training progress
- Demonstration games after training showing the neural network's learned strategy
- AlphaGo-style demo with Monte Carlo Tree Search and neural networks for Tic-Tac-Toe

## Project Structure

```
.
├── legacy_cpp_implementation/ # Original fully-functional C++ implementation with separate include/src
├── cpp_implementation/        # Simplified C++ demonstration plus full implementation
├── golang_implementation/     # Go implementation with improved architecture
├── alphago_demo/              # AlphaGo-style Tic-Tac-Toe demo
└── output/                    # Training output and visualizations
```

## Requirements

- Go 1.16 or later for the Golang implementation and AlphaGo demo
- C++17 compatible compiler for the C++ implementations
- Eigen3 library for the C++ implementations (matrix operations)
  - On macOS: `brew install eigen`
  - On Ubuntu: `apt-get install libeigen3-dev`
  - On Windows: Download from http://eigen.tuxfamily.org/

## Building and Running

Use the provided Makefile to build and run the different implementations:

```bash
# Build all implementations
make build

# Run the Legacy C++ implementation (complete neural network)
make run-legacy-cpp

# Run the simplified C++ demo
make run-cpp

# Run the full C++ neural implementation (similar to legacy but different architecture)
make run-cpp-full

# Run the Golang implementation
make run-go

# Run the AlphaGo demo
make run-alphago
```

Or use the build_all.sh script to build and run all implementations in sequence:

```bash
./build_all.sh
```

## Running Demos and Comparing Output

Each implementation can generate demo output to compare their approaches:

```bash
# Run all demos and generate output files
make run-demos
```

This will:
1. Build all implementations
2. Run a demo of each implementation
3. Generate output files in the project root directory:
   - `legacy_cpp_demo_output.txt` - Original C++ implementation output
   - `cpp_demo_output.txt` - Simplified C++ demo output
   - `go_demo_output.txt` - Golang implementation output
   - `alphago_demo_output.txt` - AlphaGo demo output

These files can be compared to understand the differences in how each implementation:
- Represents the game state
- Trains the neural network
- Makes predictions
- Visualizes the model and training process

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zachbeta/neural_rps.git
cd neural_rps
```

2. Install dependencies:
```bash
# For the Golang implementation
cd golang_implementation && go mod tidy

# For the AlphaGo demo
cd alphago_demo && go mod tidy

# For the C++ implementations (if you're on macOS)
brew install eigen
```

## Implementation Comparison

Here's a quick comparison of the different implementations:

| Implementation | Language | Architecture | Training Method | Features |
|----------------|----------|--------------|----------------|----------|
| Legacy C++ | C++17 | Modular with separate header/source files | PPO | Original neural network with complete game logic |
| C++ Demo | C++17 | Combined header files | Simplified | Demonstration version and full implementation |
| Golang | Go 1.16+ | Modular packages | PPO | Improved architecture and performance |
| AlphaGo Demo | Go 1.16+ | AlphaGo-inspired | MCTS + Neural Network | Applied to Tic-Tac-Toe as a demonstration |

## Implementation Details

Each implementation has its own README in its respective directory with more detailed information:

- [Legacy C++ Implementation](legacy_cpp_implementation/README.md) - Original implementation with complete neural network
- [C++ Implementation](cpp_implementation/README.md) - Simplified version and full implementation
- [Golang Implementation](golang_implementation/README.md) - Go-based implementation with improved architecture
- [AlphaGo Demo](alphago_demo/README.md) - Demonstration of AlphaGo techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.