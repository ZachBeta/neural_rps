# Neural Rock Paper Scissors

A neural network-based implementation of Rock Paper Scissors using different approaches and languages.

## Overview

This project explores different implementations of neural networks for playing games:

1. **Legacy C++ Implementation** - The original, fully-functional implementation with complete neural network that learns to play Rock Paper Scissors effectively
2. **C++ Implementation** - A simplified demonstration version, plus an integrated full neural network
3. **Golang Implementation** - An improved implementation with better readability, performance, and development setup
4. **AlphaGo-Style Demos**:
   - **Tic-Tac-Toe** - A demonstration of AlphaGo-like techniques applied to Tic-Tac-Toe
   - **RPS Card Game** - A strategic card game combining Rock Paper Scissors with board placement

## Features

- Neural network implementations with various architectures
- Multiple reinforcement learning approaches:
  - PPO (Proximal Policy Optimization) for the standard RPS implementations
  - AlphaGo-style MCTS (Monte Carlo Tree Search) with neural networks for board games
- Game environments with state tracking
- Standardized output format for comparing model performance
- Visualization of network architecture, weights, and training progress
- Demonstration games showing the neural networks' learned strategies

## Project Structure

```
.
├── legacy_cpp_implementation/ # Original fully-functional C++ implementation with separate include/src
├── cpp_implementation/        # Simplified C++ demonstration plus full implementation
├── golang_implementation/     # Go implementation with improved architecture
├── alphago_demo/              # AlphaGo-style implementations (Tic-Tac-Toe and RPS Card Game)
├── scripts/                   # Shell scripts for running tests and demos
├── output/                    # Training output and visualizations
├── shared_output_format.md    # Specification for standardized output format
```

## Requirements

- Go 1.16 or later for the Golang implementation and AlphaGo demos
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

# Run the AlphaGo Tic-Tac-Toe demo
make run-alphago

# Run the AlphaGo RPS Card Game demo
make run-alphago-rps
```

Or use the scripts/build_all.sh script to build and run all implementations in sequence:

```bash
./scripts/build_all.sh
```

## Running Demos and Comparing Output

Each implementation can generate standardized output to compare their approaches:

```bash
# Run all demos and generate output files
make run-demos
```

This will:
1. Build all implementations
2. Run a demo of each implementation
3. Generate output files in the output directory:
   - `output/legacy_cpp_demo_output.txt` - Original C++ implementation output
   - `output/cpp_demo_output.txt` - Simplified C++ demo output
   - `output/go_demo_output.txt` - Golang implementation output
   - `output/alphago_demo_output.txt` - AlphaGo Tic-Tac-Toe demo output
   - `output/alphago_rps_demo_output.txt` - AlphaGo RPS Card Game demo output

These files can be compared to understand the differences in how each implementation:
- Represents the game state
- Trains the neural network
- Makes predictions
- Visualizes the model and training process

For details on the output format, see [shared_output_format.md](shared_output_format.md).

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

# For the AlphaGo demos
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
| AlphaGo Tic-Tac-Toe | Go 1.16+ | AlphaGo-inspired | MCTS + Neural Network | Tic-Tac-Toe with Monte Carlo Tree Search |
| AlphaGo RPS Card Game | Go 1.16+ | AlphaGo-inspired | MCTS + Neural Network | Card game combining RPS with board placement |

## Game Descriptions

### Rock Paper Scissors
The traditional game where Rock beats Scissors, Scissors beats Paper, and Paper beats Rock. The neural network learns to predict and counter opponent moves.

### RPS Card Game
A strategic card game where players place Rock, Paper, or Scissors cards on a 3×3 board. Cards can capture adjacent opponent cards according to RPS rules. The player with the most cards on the board when all cards are played wins.

### Tic-Tac-Toe
The classic game where players take turns placing X or O on a 3×3 grid, trying to get three in a row. This implementation demonstrates AlphaGo-style techniques with Monte Carlo Tree Search.

## Implementation Details

Each implementation has its own README in its respective directory with more detailed information:

- [Legacy C++ Implementation](legacy_cpp_implementation/README.md) - Original implementation with complete neural network
- [C++ Implementation](cpp_implementation/README.md) - Simplified version and full implementation
- [Golang Implementation](golang_implementation/README.md) - Go-based implementation with improved architecture
- [AlphaGo Demos](alphago_demo/README.md) - Demonstrations of AlphaGo techniques for Tic-Tac-Toe
- [RPS Card Game](alphago_demo/RPS_README.md) - AlphaGo-style implementation of the RPS card game

## License

This project is licensed under the MIT License - see the LICENSE file for details.