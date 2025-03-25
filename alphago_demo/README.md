# AlphaGo-Style Tic-Tac-Toe Demo

This is a simplified implementation of an AlphaGo-like self-learning AI for Tic-Tac-Toe. It demonstrates the core principles of AlphaGo/AlphaZero in a more accessible game environment.

## Overview

The system combines:
- Neural networks (policy and value)
- Monte Carlo Tree Search (MCTS)
- Self-play for training

For a detailed explanation of the implementation, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Architecture

The project is organized into the following packages:

- **pkg/game**: Game logic and board representation
- **pkg/neural**: Neural network implementations (policy and value networks)
- **pkg/mcts**: Monte Carlo Tree Search implementation
- **pkg/training**: Self-play and training logic
- **cmd/tictactoe**: Main application entry point

## Components

### Neural Networks

1. **Policy Network**: Predicts the probability distribution over possible moves
2. **Value Network**: Estimates the probability of winning from a given position

### Monte Carlo Tree Search (MCTS)

MCTS is used to improve the quality of moves by looking ahead and evaluating potential future positions. The neural networks guide the search process.

### Self-Play Training

The system trains by playing against itself, using the results to improve the neural networks:
1. Generate games through self-play using MCTS guided by current neural networks
2. Extract training examples from these games
3. Train neural networks on these examples
4. Repeat with improved networks

## How to Run

Build and run the TicTacToe demo:

```bash
./run.sh
```

Or build it manually:

```bash
go build -o tictactoe cmd/tictactoe/main.go
./tictactoe
```

## Playing Against the AI

The demo runs in a simulated mode with predefined moves, but you can modify the code to play interactively. The interactive mode allows you to play Tic-Tac-Toe against the trained AI by inputting moves as row,column coordinates (0-2,0-2).

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