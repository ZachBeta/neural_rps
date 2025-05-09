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

## Project Organization

The project consists of multiple independent implementations with some integration points:

### Top-Level Structure

```
.
├── legacy_cpp_implementation/ # Original fully-functional C++ implementation with separate include/src
├── cpp_implementation/        # Simplified C++ demonstration plus full implementation
├── golang_implementation/     # Go implementation with improved architecture
├── alphago_demo/              # AlphaGo-style implementations (Tic-Tac-Toe and RPS Card Game)
├── output/                    # Shared output directory for all implementations
└── Makefile                   # Top-level makefile with convenience commands for all implementations
```

### Implementation Relationships

- The **golang_implementation** and **alphago_demo** packages can interact with each other:
  - The AlphaGo agent from alphago_demo can be used in golang_implementation tournaments
  - Both implement different approaches to the RPS game but share common interfaces

### Running Commands

The top-level Makefile provides simplified commands for common tasks. Each implementation also has its own Makefile for more detailed operations.

```bash
# Build all implementations
make build

# Run specific implementations
make run-legacy-cpp
make run-cpp
make run-go
make run-alphago-ttt
make run-alphago-rps

# AlphaGo-specific tasks
make alphago-train       # Train AlphaGo RPS models
make alphago-tournament  # Compare different AlphaGo models

### Custom Training via CLI Flags

You can run the trainer directly with custom hyperparameters:

```bash
cd alphago_demo
go build -o bin/train_models cmd/train_models/main.go

# Example: train a larger model with 256 hidden neurons,
# 2000 self-play games, 20 epochs, 500 MCTS simulations,
# exploration=1.2, 100 tournament games, on 16 threads:
./bin/train_models --parallel --threads 16 \
  --m2-hidden 256 --m2-games 2000 --m2-epochs 20 \
  --m2-sims 500 --m2-exploration 1.2 \
  --tournament-games 100
```

The available flags are:

- `--m1-games` (default 100)
- `--m1-epochs` (default 5)
- `--m1-hidden` (default 64)
- `--m1-sims` (default 300)
- `--m1-exploration` (default 1.5)
- `--m2-games` (default 1000)
- `--m2-epochs` (default 10)
- `--m2-hidden` (default 128)
- `--m2-sims` (default 200)
- `--m2-exploration` (default 1.0)
- `--tournament-games` (default 30)
- `--parallel` (enable parallel execution)
- `--threads` (number of threads, default auto)

# Golang-specific tasks
make golang-tournament   # Run tournaments between agents
make golang-vs-alphago   # Compare Golang and AlphaGo agents
```

## Features

- Neural network implementations with various architectures:
  - PPO (Proximal Policy Optimization) in the golang_implementation
  - AlphaGo-style MCTS (Monte Carlo Tree Search) with neural networks in alphago_demo
- Game environments with state tracking
- Tournament systems for comparing agent performance
- Comprehensive testing and documentation

## Requirements

- Go 1.16 or later for the Golang implementation and AlphaGo demos
- C++17 compatible compiler for the C++ implementations
- Eigen3 library for the C++ implementations (matrix operations)
  - On macOS: `brew install eigen`
  - On Ubuntu: `apt-get install libeigen3-dev`
  - On Windows: Download from http://eigen.tuxfamily.org/

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zachbeta/neural_rps.git
cd neural_rps
```

2. Install dependencies:
```bash
# Install all dependencies
make install-deps
```

## Implementation Details

Each implementation has its own README with more detailed information:

- [Legacy C++ Implementation](legacy_cpp_implementation/README.md)
- [C++ Implementation](cpp_implementation/README.md)
- [Golang Implementation](golang_implementation/README.md)
- [AlphaGo Demos](alphago_demo/README.md)

## Game Descriptions

### Rock Paper Scissors
The traditional game where Rock beats Scissors, Scissors beats Paper, and Paper beats Rock. The neural network learns to predict and counter opponent moves.

### RPS Card Game
A strategic card game where players place Rock, Paper, or Scissors cards on a 3×3 board. Cards can capture adjacent opponent cards according to RPS rules. The player with the most cards on the board when all cards are played wins.

### Tic-Tac-Toe
The classic game where players take turns placing X or O on a 3×3 grid, trying to get three in a row. This implementation demonstrates AlphaGo-style techniques with Monte Carlo Tree Search.

## License

This project is licensed under the MIT License - see the LICENSE file for details.