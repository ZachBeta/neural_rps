# Neural RPS - Golang Implementation

This directory contains the Golang implementation of the Neural Rock Paper Scissors project.

## Overview

This Golang implementation offers better readability, performance, and a more ergonomic development/build/test setup compared to the C++ versions. It uses Go's concurrency features for optimal performance and follows idiomatic Go patterns.

## Project Structure

```
.
├── cmd/
│   └── neural_rps/     # Main program
├── pkg/
│   ├── agent/          # PPO agent implementation
│   ├── game/           # Game environment
│   ├── neural/         # Neural network implementation
│   └── visualizer/     # Visualization utilities
├── go.mod              # Go module definition
├── go.sum              # Go dependencies checksum
└── Makefile            # Build commands
```

## Building & Running

```bash
cd golang_implementation
make build   # Build the binary
make run     # Run the program
make test    # Run tests
```

## Features

- Neural network with one hidden layer and configurable size
- PPO algorithm for policy optimization
- Game environment with state tracking
- Visualization of network architecture and training progress
- Uses Go's concurrency features for parallelized training
- Standardized output format for comparison with other implementations

## Neural Network Architecture

- **Input Layer**: 9 neurons (encoding previous moves and game state)
- **Hidden Layer**: 16 neurons with ReLU activation
- **Output Layer**: 3 neurons with softmax activation (rock, paper, scissors probabilities)

The neural network learns to predict the optimal counter-move against each possible opponent move:
- When opponent plays Rock → Play Paper
- When opponent plays Paper → Play Scissors
- When opponent plays Scissors → Play Rock

## Standardized Output

This implementation generates standardized output in the `go_demo_output.txt` file that follows the format specified in [shared_output_format.md](../shared_output_format.md), making it easy to compare with other implementations.

Example output includes:
- Network architecture details
- Training process metrics
- Model predictions for each input type
- Summary of model parameters

## Performance

The Golang implementation achieves:
- Faster training time compared to C++ implementations
- Better final reward values
- Highly consistent prediction accuracy

For a detailed comparison with other implementations, see [rps_model_comparison_report.md](../rps_model_comparison_report.md). 