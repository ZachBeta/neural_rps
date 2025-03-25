# Golang Implementation of Neural RPS

This directory contains the Golang implementation of neural networks for playing the Rock Paper Scissors (RPS) card game. This implementation features a clean, modular design with a focus on performance and readability.

## Features

- **RPSCardGame Environment**: A complete implementation of the Rock Paper Scissors card game with a 3x3 board
- **Multiple Agent Types**:
  - **AlphaGo-style Agent**: Using Monte Carlo Tree Search with policy and value networks
  - **PPO Agent**: Using Proximal Policy Optimization reinforcement learning
  - **Random Agent**: A baseline agent making random moves
- **Tournament System**: Ability to pit different agents against each other in tournament play
- **Comprehensive Testing**: Unit tests for all game logic and agent behavior
- **Visualization**: String representation of game state for easy debugging

## Directory Structure

```
.
├── cmd/
│   ├── neural_rps/        # Main binary entry point
│   ├── tournament/        # Tournament runner between different agents
│   └── alphago_vs_alphago/ # Tournament comparing small vs large AlphaGo models
├── pkg/
│   ├── agent/             # Agent implementations (AlphaGo, PPO, Random)
│   ├── game/              # Game environment and rules
│   └── neural/            # Neural network implementation
├── results/               # Tournament results output
├── Makefile               # Build and run targets
└── go.mod                 # Go module definition
```

## Relationship with alphago_demo Package

This implementation has integration points with the alphago_demo package:

- **AlphaGo Agent Adapter**: The `pkg/agent/alphago_agent.go` file contains an adapter that allows using the AlphaGo neural networks from the alphago_demo package with the RPS card game environment defined in this package.
- **Model Loading**: The adapter can load trained neural network models from `alphago_demo/output/`.
- **Game State Conversion**: The adapter converts between the RPS card game state formats used by the two packages.

This integration lets us compare the performance of different agent types (PPO vs AlphaGo) in the same environment using the tournament system.

## Building and Running

You can use the provided Makefile to build and run the implementation:

```bash
# Build the main binary
make build

# Run tests
make test

# Run a tournament between AlphaGo and PPO agents
make tournament

# Run a tournament between AlphaGo and PPO agents with verbose output
make tournament-verbose

# Run a tournament between small and large AlphaGo models
make alphago-vs-alphago

# Run a tournament between small and large AlphaGo models with verbose output
make alphago-vs-alphago-verbose
```

## Agent Implementation Details

### AlphaGo Agent

The AlphaGo agent combines Monte Carlo Tree Search (MCTS) with neural networks:

- **Policy Network**: Predicts which moves are likely to be good
- **Value Network**: Evaluates board positions
- **MCTS**: Uses both networks to guide search through the game tree
- **Exploration Parameter**: Controls balance between exploration and exploitation

The AlphaGo agent is highly effective at strategic play, as it can look ahead several moves and evaluate positions.

### PPO Agent

The PPO (Proximal Policy Optimization) agent:

- Uses a single neural network to select actions
- Learns directly from gameplay experience
- Updates policy parameters to maximize expected reward
- Maintains a balance between learning new behaviors and preserving existing capabilities

The PPO agent is more lightweight than AlphaGo but can still learn effective strategies.

## Tournament System

The tournament system allows for comparing the performance of different agents:

- **Fair Play**: Alternates which agent goes first to ensure fairness
- **Statistics**: Tracks win rates, draw rates, and position-based winning patterns
- **Analysis**: Provides insights into which positions lead to more wins
- **Results Storage**: Saves tournament results to text files for later analysis

## Future Work

- Improve agent training methods
- Add more agent types (e.g., Q-learning, A3C)
- Enhance visualization with a GUI
- Parallelize MCTS for better performance 