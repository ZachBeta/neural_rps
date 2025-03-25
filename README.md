# Neural Rock Paper Scissors

A neural network-based implementation of Rock Paper Scissors using different approaches and languages.

## Overview

This project explores different implementations of neural networks for playing Rock Paper Scissors:

1. **C++ Implementation** - The original implementation with basic neural network functionality
2. **Golang Implementation** - An improved implementation with better readability, performance, and development setup
3. **AlphaGo-Style Demo** - A demonstration of AlphaGo-like techniques applied to Tic-Tac-Toe

## Features

- Neural network implementation with one hidden layer
- PPO algorithm for policy optimization
- Game environment with state tracking
- Visualization of network architecture, weights, and training progress
- Demonstration games after training
- AlphaGo-style demo with Monte Carlo Tree Search and neural networks for Tic-Tac-Toe

## Project Structure

```
.
├── cpp_implementation/        # Original C++ implementation
├── golang_implementation/     # First-pass Golang implementation
├── alphago_demo/              # AlphaGo-style Tic-Tac-Toe demo
└── output/                    # Training output and visualizations
```

## Requirements

- Go 1.16 or later for the Golang implementation and AlphaGo demo
- C++17 compatible compiler for the C++ implementation
- Eigen3 library for the C++ implementation (matrix operations)
  - On macOS: `brew install eigen`
  - On Ubuntu: `apt-get install libeigen3-dev`

## Building and Running

Use the provided Makefile to build and run the different implementations:

```bash
# Build all implementations
make build

# Run the C++ implementation
make run-cpp

# Run the Golang implementation
make run-go

# Run the AlphaGo demo
make run-alphago
```

Or use the build_all.sh script to build and run all implementations in sequence:

```bash
./build_all.sh
```

Note: The C++ implementation requires the Eigen3 library to be installed on your system.

## Running Demos and Comparing Output

Each implementation can generate demo output to compare their approaches:

```bash
# Run all demos and generate output files
make run-demos

# Or use the dedicated script:
./run_demos.sh
```

This will:
1. Build all implementations
2. Run a demo of each implementation
3. Generate output files in the project root directory:
   - `cpp_demo_output.txt` - C++ implementation output
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

# For the C++ implementation (if you're on macOS)
brew install eigen
```

## Usage

Run the Golang implementation:
```bash
make run-go
```

Run the C++ implementation:
```bash
make run-cpp
```

Run the AlphaGo-style Tic-Tac-Toe demo:
```bash
make run-alphago
```

## Implementation Details

Each implementation has its own README in its respective directory with more detailed information:

- [C++ Implementation](cpp_implementation/README.md)
- [Golang Implementation](golang_implementation/README.md)
- [AlphaGo Demo](alphago_demo/README.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.