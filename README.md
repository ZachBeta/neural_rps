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

# Golang-specific tasks
make golang-tournament   # Run tournaments between agents
make golang-vs-alphago   # Compare Golang and AlphaGo agents
```

### Tournament Systems

The project includes several sophisticated tournament systems to evaluate and compare different AI agents:

#### ELO Tournament System

The ELO tournament system is the most comprehensive comparison tool, supporting all agent types:

```bash
cd alphago_demo
go run cmd/elo_tournament/main.go [options]
```

Options:
- `-games <n>`: Number of games per agent pair (default: 100)
- `-verbose`: Enable detailed output for each game
- `-cutoff <n>`: ELO threshold to prune underperforming agents (default: 1400)
- `-output <file>`: Specify output file for results (default: output/tournament_results.csv)
- `-top <n>`: Only use top N agents from previous tournament results

This tournament automatically discovers trained models in the output directory and creates a full round-robin tournament with ELO ratings for all agents.

#### Minimax Comparison Tournament

To evaluate neural networks against traditional minimax search algorithms:

```bash
cd alphago_demo
go run cmd/tournament_with_minimax/main.go [options]
```

Options:
- `-games <n>`: Number of games per agent pair (default: 30)
- `-verbose`: Show detailed output for each move
- `-output <file>`: Output file location (default: output/tournament_with_minimax_results.csv)
- `-max-networks <n>`: Limit number of neural networks of each type (default: 3)

This tournament includes minimax agents at different search depths and compares them against the neural network approaches.

### Training Entry Points

The project includes several specialized training entry points for different neural network approaches:

#### AlphaGo-Style Model Training

The primary training entry point for AlphaGo-style models with MCTS:

```bash
cd alphago_demo
go run cmd/train_models/main.go [options]
```

This comprehensive trainer offers:
- Self-play training with MCTS
- Parallel execution support
- Comparison of different network sizes
- Tournament evaluation

It supports two training methods:
- **AlphaGo**: Neural networks with MCTS (default)
- **NEAT**: Neuroevolution of Augmenting Topologies

#### Extended Training for Top Agents

For continuing training of pre-trained models:

```bash
cd alphago_demo
go run cmd/train_top_agents/main.go [options]
```

Options:
- `-games <n>`: Self-play games for training (default: 2000)
- `-sims <n>`: MCTS simulations per move (default: 400)
- `-tournament-only`: Skip training and run tournament only
- `-training-only`: Skip tournament and do training only
- `-output <dir>`: Directory for output files (default: output/extended_training)

#### Supervised Learning from Expert Data

For training models on existing expert gameplay data:

```bash
cd alphago_demo
go run cmd/train_supervised/main.go [options]
```

Options:
- `-hidden <n>`: Hidden layer size (default: 128)
- `-lr <n>`: Learning rate (default: 0.001)
- `-batch <n>`: Batch size (default: 32)
- `-epochs <n>`: Maximum epochs (default: 100)
- `-patience <n>`: Early stopping patience (default: 10)

#### Training Data Generation

Generate expert gameplay data using minimax search:

```bash
cd alphago_demo
go run cmd/generate_training_data/main.go [options]
```

Options:
- `-positions <n>`: Number of positions to generate (default: 10000)
- `-depth <n>`: Minimax search depth (default: 5)
- `-time-limit <dur>`: Time limit per move (default: 5s)
- `-output <file>`: Output file path (default: training_data.json)

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

## GPU Acceleration

This project now supports GPU acceleration through TensorFlow for significantly improved performance in neural network inference and training.

### Requirements

- Go 1.20 or later
- TensorFlow Go bindings (for GPU mode)
- Metal Performance Shaders (MPS) support on Apple Silicon macs

### Installation of TensorFlow Dependencies

To use GPU acceleration, you need to install TensorFlow and its Go bindings:

#### macOS with Apple Silicon (M1/M2/M3)

```bash
# Install TensorFlow dependencies
brew install tensorflow

# Set environment variables
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/lib
export CPATH=$CPATH:/opt/homebrew/include

# Install TensorFlow Go bindings
go get github.com/tensorflow/tensorflow/tensorflow/go@v2.15.0
```

#### Linux with NVIDIA GPU

```bash
# Install CUDA and cuDNN according to TensorFlow documentation
# https://www.tensorflow.org/install/gpu

# Install TensorFlow C library
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
sudo ldconfig

# Install TensorFlow Go bindings
go get github.com/tensorflow/tensorflow/tensorflow/go@v2.15.0
```

### Building with GPU Support

You can build any component with GPU support using the build script:

```bash
# Build with CPU-only support
./scripts/build.sh

# Build with GPU acceleration
./scripts/build.sh --gpu
```

Or manually using build tags:

```bash
go build -tags=gpu -o bin/benchmark cmd/benchmark/main.go
```

### Benchmarking CPU vs. GPU Performance

To compare CPU and GPU performance:

```bash
# Run CPU benchmark
./bin/benchmark

# Run GPU benchmark (if built with GPU support)
./bin/benchmark --gpu --batch 128 --samples 50000
```

The benchmark will show inference speed and performance metrics for both modes.

### Batched Operations

The GPU implementation supports batched operations for increased throughput:

1. **Neural Network Inference**: The `BatchedNeuralNetwork` interface allows processing multiple inputs in a single forward pass.
2. **MCTS Search**: The batched MCTS implementation collects multiple positions to evaluate in a single GPU call.

### Architecture

The GPU acceleration architecture follows these principles:

1. **Clean Separation**: GPU and CPU implementations use separate packages with build tags
2. **Shared Interfaces**: Common interfaces ensure compatibility between implementations
3. **Conditional Compilation**: GPU features are only included when using the `gpu` build tag
4. **Memory Optimization**: Tensor pooling reduces memory allocations
5. **Optional Dependencies**: TensorFlow is an optional dependency only needed for GPU builds

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