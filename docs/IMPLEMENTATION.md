# AlphaGo-Style Tic-Tac-Toe Implementation Documentation

This document provides a detailed overview of the AlphaGo-style Tic-Tac-Toe implementation. This serves as a checkpoint for future development and a reference for understanding the architecture.

## Overview

This implementation demonstrates the core AlphaGo principles in a simplified environment:

1. **Neural Networks** for policy and value estimation
2. **Monte Carlo Tree Search** for move planning
3. **Self-Play** for training data generation

## Project Structure

```
alphago_demo/
├── cmd/
│   └── tictactoe/
│       └── main.go          # Main executable
├── pkg/
│   ├── game/
│   │   └── tictactoe.go     # Game rules and board representation
│   ├── mcts/
│   │   ├── node.go          # MCTS node implementation
│   │   └── search.go        # MCTS algorithm
│   ├── neural/
│   │   ├── policy.go        # Policy network (action probabilities)
│   │   └── value.go         # Value network (win probability)
│   └── training/
│       └── self_play.go     # Self-play and training logic
├── go.mod                   # Go module definition
├── README.md                # Project overview
└── run.sh                   # Script to build and run the game
```

## Components

### Game Logic (`pkg/game/tictactoe.go`)

The game logic handles:
- Board representation using a 3×3 grid
- Player management (X and O)
- Move validation and application
- Win/draw detection
- Board state serialization for neural network input

Key types:
- `AGPlayer`: Represents a player (Empty, PlayerX, PlayerO)
- `AGBoard`: Represents the 3×3 board
- `AGGame`: Maintains the game state
- `AGMove`: Represents a move with row, column, and player

### Neural Networks (`pkg/neural/`)

#### Policy Network (`policy.go`)

The policy network predicts the probability distribution over possible moves given a board state.

- Architecture: 2-layer neural network (input → hidden → output)
- Input: 9 features representing the board state (1 for X, -1 for O, 0 for Empty)
- Output: 9 probabilities corresponding to each board position
- Activation functions: ReLU for hidden layer, softmax for output layer

#### Value Network (`value.go`)

The value network estimates the win probability from a given board state.

- Architecture: 2-layer neural network (input → hidden → output)
- Input: 9 features representing the board state
- Output: Single value between 0 and 1 (0 = loss, 0.5 = draw, 1 = win)
- Activation functions: ReLU for hidden layer, sigmoid for output

Both networks include training methods using simplified backpropagation.

### Monte Carlo Tree Search (`pkg/mcts/`)

#### Node Structure (`node.go`)

Each MCTS node contains:
- Game state
- Move that led to this state
- Visit statistics (visits, total value)
- Tree structure (parent, children)
- Policy priors from the policy network

Methods include:
- UCB calculation for exploration/exploitation balance
- Node selection and update
- Child management

#### Search Algorithm (`search.go`)

The MCTS algorithm follows four phases:
1. **Selection**: Traverse tree using UCB until reaching an expandable node
2. **Expansion**: Add a new child node
3. **Simulation**: Use value network to evaluate position
4. **Backpropagation**: Update statistics up the tree

The search is guided by the neural networks, with the policy network improving node selection and the value network replacing random rollouts.

### Self-Play Training (`pkg/training/self_play.go`)

The training system:
1. Generates games through self-play
2. Extracts training examples (board states, policies, outcomes)
3. Trains the neural networks on these examples

Key components:
- `AGTrainingExample`: Stores board state, policy target, and value target
- `AGSelfPlay`: Manages self-play generation and network training
- Policy extraction from MCTS visit counts
- Batch-based training of neural networks

### Main Application (`cmd/tictactoe/main.go`)

The main application provides:
- Network initialization and training
- Interactive gameplay against the AI
- Simulated demo mode
- Network evaluation

## Implementation Details

### Neural Network Training

The networks are trained using examples from self-play:
- Policy targets come from MCTS visit counts (improved policies)
- Value targets come from game outcomes (0 for loss, 0.5 for draw, 1 for win)
- Training uses basic gradient descent with a fixed learning rate
- Cross-entropy loss for policy, mean squared error for value

### MCTS Enhancements

The MCTS implementation includes several enhancements from the AlphaGo approach:
- Prior probabilities from the policy network
- Value estimation from the value network instead of random rollouts
- UCB formula with policy priors for better exploration

### Performance Considerations

This implementation is optimized for clarity rather than performance, but includes:
- Batch-based training for improved efficiency
- Configurable simulation count for MCTS
- Game state copying to prevent state corruption

## Usage

### Running the Demo

```bash
cd alphago_demo
./run.sh
```

### Configuration Options

In `main.go`:
- `trainNetworks`: Enable/disable training
- `selfPlayGames`: Number of self-play games for training
- `trainingEpochs`: Number of training epochs
- `batchSize`: Batch size for training
- `learningRate`: Learning rate for neural networks

In `mcts/search.go`:
- `DefaultAGMCTSParams()`: MCTS parameters (simulation count, exploration constant)

## Limitations and Future Work

Current limitations:
- Simple neural network architecture
- Limited training data generation
- Basic training algorithm without regularization

Potential improvements:
- Deeper network architectures
- Training optimizations (learning rate schedules, regularization)
- Parallelized MCTS for faster play
- Support for larger board games (Connect Four, Go)
- Visualization of search trees and network internals
- Proper train/validate/test separation

## Conclusion

This implementation demonstrates the core principles of AlphaGo/AlphaZero in a simplified environment. While not as powerful as the full AlphaGo system, it illustrates how neural networks and MCTS can be combined to create a self-improving game-playing system. 