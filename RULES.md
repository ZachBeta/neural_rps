# Neural Network Rock Paper Scissors

This file defines the rules and standards for the Neural RPS project, which includes:

* C++ RPS implementation (legacy and modern)
* Golang RPS implementation  
* AlphaGo-style implementation for RPS card game and Tic-Tac-Toe

## Project Overview
This project explores different implementations of neural networks for playing games:

1. Rock Paper Scissors - Using basic neural networks with reinforcement learning
2. RPS Card Game - Combining RPS with strategic card placement
3. Tic-Tac-Toe - Using AlphaGo-style techniques (MCTS + neural networks)

## Code Style Guidelines
### C++ Implementations
- Use modern C++ features (C++17 or later)
- Follow Google C++ Style Guide
- Use clear, descriptive variable and function names
- Include comments for complex logic
- Write unit tests for core functionality

### Golang Implementations
- Follow standard Go code conventions
- Use idiomatic Go patterns
- Organize code into appropriate packages
- Provide good documentation
- Write comprehensive tests

## Project Structure
```
.
├── cpp_implementation/        # Modern C++ implementation
├── legacy_cpp_implementation/ # Original C++ implementation
├── golang_implementation/     # Go implementation of RPS
├── alphago_demo/              # AlphaGo-style implementations
├── scripts/                   # Utility scripts
├── output/                    # Training output and model data
└── shared_output_format.md    # Output format specification
```

## Build System
- C++ implementations use CMake
- Go implementations use standard Go build tools
- A unified Makefile provides common commands for all implementations

## Neural Network Architectures

### RPS Implementations
- Input layer: 6-9 neurons (one-hot encoding of previous moves)
- Hidden layer: 8-16 neurons with activation (ReLU or Sigmoid)
- Output layer: 3 neurons with softmax activation (rock, paper, scissors probabilities)

### AlphaGo-Style Implementations
- Policy network and value network
- Input: Board state encoding
- Policy output: Move probabilities
- Value output: Win probability estimate
- Combined with Monte Carlo Tree Search (MCTS)

## Standardized Output Format
For comparing implementations, all versions must follow a consistent output format as defined in `shared_output_format.md`:

1. Header & Implementation Info
2. Network Architecture
3. Training Process
4. Model Predictions
5. Model Parameters (Optional)

## Game Definitions

### Rock Paper Scissors
- Traditional game rules (rock beats scissors, scissors beats paper, paper beats rock)
- Neural network plays against various strategies (random, pattern-based, mimic)
- Training uses reinforcement learning with rewards for wins

### RPS Card Game
- Cards are placed on a 3×3 board
- When adjacent, cards capture according to RPS rules
- Players have a limited hand of cards
- Goal is to have the most cards on the board when all cards are played

### Tic-Tac-Toe
- Traditional 3×3 board
- Players alternate placing X and O
- First to get 3 in a row (horizontally, vertically, or diagonally) wins
- Used to demonstrate AlphaGo-style techniques 