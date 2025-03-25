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

## Neural Network Implementation in Go

The neural network has been implemented in Go for better readability, performance, and developer experience. The implementation features:

### Architecture

- Simple feed-forward neural network with three layers:
  - Input layer: 6 neurons (for encoding both player and opponent previous moves)
  - Hidden layer: 12 neurons with ReLU activation
  - Output layer: 3 neurons with softmax activation

### Features

- **Parallelized Training**: Utilizes Go's goroutines for efficient parallel processing during training
- **Batch Processing**: Supports mini-batch gradient descent for improved training efficiency
- **Xavier Initialization**: Weights initialized using Xavier/Glorot initialization for better convergence
- **Visualization**: Built-in visualization tools for network architecture and training progress
- **Persistence**: Save and load network weights using Go's encoding/gob

### Usage

Create and train a neural network:

```go
// Create a new neural network with 6 inputs, 12 hidden neurons, and 3 outputs
nn := neural.NewNetwork(6, 12, 3)

// Set up training options
options := neural.TrainingOptions{
    LearningRate: 0.01,
    Epochs:       500,
    BatchSize:    32,
    Parallel:     true,
}

// Train the network
nn.Train(inputs, targets, options)

// Make predictions
prediction := nn.Predict(input)

// Save the trained model
nn.SaveWeights("model.gob")

// Load a saved model
nn.LoadWeights("model.gob")
```

Visualize the network:

```go
// Create a visualizer that writes to a file
visualizer, _ := neural.NewFileVisualizer("output.txt")
defer visualizer.Close()

// Visualize network architecture
visualizer.VisualizeArchitecture(nn, []string{"Input", "Hidden", "Output"})
visualizer.VisualizeNetworkGraphical(nn)

// Visualize weights
visualizer.VisualizeWeights(nn, inputLabels, hiddenLabels, outputLabels)

// Visualize predictions
visualizer.VisualizePrediction(nn, input, output, inputLabels, outputLabels)
```

### Example

See `cmd/neural_rps/main.go` for a complete example of training a neural network to play Rock, Paper, Scissors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.