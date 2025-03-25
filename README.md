# Neural Rock Paper Scissors

A neural network-based implementation of Rock Paper Scissors using Proximal Policy Optimization (PPO) in Go.

## Overview

This project implements a neural network agent that learns to play Rock Paper Scissors using reinforcement learning. The agent uses a Proximal Policy Optimization (PPO) algorithm to learn optimal strategies against a random opponent.

## Features

- Neural network implementation with one hidden layer
- PPO algorithm for policy optimization
- Game environment with state tracking
- Visualization of network architecture, weights, and training progress
- Demonstration games after training

## Project Structure

```
.
├── cmd/
│   └── neural_rps/        # Main program
├── pkg/
│   ├── agent/            # PPO agent implementation
│   ├── game/             # Game environment
│   ├── neural/           # Neural network implementation
│   └── visualizer/       # Visualization utilities
└── output/               # Training output and visualizations
```

## Requirements

- Go 1.16 or later
- gonum.org/v1/gonum/mat (for matrix operations)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zachbeta/neural_rps.git
cd neural_rps
```

2. Install dependencies:
```bash
go mod tidy
```

## Usage

Run the training program:
```bash
go run cmd/neural_rps/main.go
```

The program will:
1. Initialize the neural network and environment
2. Train the agent for 1000 episodes
3. Visualize the training progress
4. Play 3 demonstration games

## Training Process

The agent learns through the following steps:
1. Collects experience by playing games
2. Updates its policy using PPO every 10 episodes
3. Visualizes its progress and network state periodically

## Visualization

The program generates visualizations in the `output` directory:
- Network architecture
- Weight distributions
- Action probabilities
- Training progress

## License

This project is licensed under the MIT License - see the LICENSE file for details.