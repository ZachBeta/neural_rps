# AlphaGo-Style Implementations

This directory contains implementations of AlphaGo-like self-learning AI for multiple games:
1. **Tic-Tac-Toe** - A simple implementation to demonstrate core AlphaGo concepts
2. **Rock-Paper-Scissors Card Game** - A strategic card placement game with RPS mechanics

Both implementations demonstrate the core principles of AlphaGo/AlphaZero in accessible game environments.

## Overview

These systems combine:
- Neural networks (policy and value)
- Monte Carlo Tree Search (MCTS)
- Self-play for training

For a detailed explanation of the implementation architecture, see [IMPLEMENTATION.md](IMPLEMENTATION.md).
For details on the RPS Card Game specifically, see [RPS_README.md](RPS_README.md).

## Architecture

All projects are organized into the following packages:

- **pkg/game**: Game logic and board representation
- **pkg/neural**: Neural network implementations (policy and value networks)
- **pkg/mcts**: Monte Carlo Tree Search implementation
- **pkg/training**: Self-play and training logic
- **cmd/tictactoe**: Tic-Tac-Toe application entry point
- **cmd/rps_card**: RPS Card Game application entry point

## Components

### Neural Networks

1. **Policy Network**: Predicts the probability distribution over possible moves
2. **Value Network**: Estimates the probability of winning from a given position

### Monte Carlo Tree Search (MCTS)

MCTS is used to improve the quality of moves by looking ahead and evaluating potential future positions. The neural networks guide the search process.

### Self-Play Training

The systems train by playing against themselves, using the results to improve the neural networks:
1. Generate games through self-play using MCTS guided by current neural networks
2. Extract training examples from these games
3. Train neural networks on these examples
4. Repeat with improved networks

## How to Run

### Tic-Tac-Toe Demo

Build and run the TicTacToe demo:

```bash
./run.sh
```

Or build it manually:

```bash
go build -o tictactoe cmd/tictactoe/main.go
./tictactoe
```

### RPS Card Game

Build and run the RPS Card Game:

```bash
./build.sh
./bin/rps_card
```

Or use the dedicated script:

```bash
./run_rps.sh
```

## Playing Against the AI

Both games support interactive play, allowing you to challenge the trained AI:

- **Tic-Tac-Toe**: Input moves as row,column coordinates (0-2,0-2)
- **RPS Card Game**: Select cards and placement positions as guided by the prompts

## Technical Notes

- The neural networks are simplified 2-layer networks with ReLU and softmax/sigmoid activations
- Training uses a basic implementation of gradient descent
- MCTS uses Upper Confidence Bounds (UCB) to balance exploration and exploitation
- All implementations are designed for educational purposes rather than maximum performance

## Future Improvements

Potential improvements include:
- More sophisticated neural network architectures
- Better training algorithms (e.g., adding regularization)
- Optimization for performance
- Extension to other games (e.g., Connect Four, Gomoku)
- Visualization of the MCTS search tree

## Comparison of Implementations

| Feature | Tic-Tac-Toe | RPS Card Game |
|---------|-------------|---------------|
| Board Size | 3×3 | 3×3 |
| Game Complexity | Low | Medium |
| State Space Size | ~10^3 | ~10^4 |
| Input Features | 9 | 81 |
| Training Time | Fast | Medium |
| MCTS Depth | Shallow | Medium | 