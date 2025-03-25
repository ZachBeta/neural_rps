# Legacy C++ Implementation - Neural RPS

This directory contains the original, fully-functional C++ implementation of the Neural Rock Paper Scissors project.

## Overview

This implementation uses a clean, modular structure with separate header and source files. It features:

- A neural network that learns to play Rock Paper Scissors through self-play
- Proximal Policy Optimization (PPO) reinforcement learning algorithm
- Training visualization and performance metrics
- Unit tests for core components

## How It Works

The neural network learns to play Rock Paper Scissors optimally through reinforcement learning:

1. The agent starts with random behavior
2. Through self-play, it collects experience (state, action, reward)
3. It uses PPO to update its policy, gradually learning the optimal strategy:
   - Rock beats Scissors
   - Paper beats Rock
   - Scissors beats Paper

After training (~500 episodes), the neural network learns the optimal counter-strategy against each move, achieving consistently high rewards.

## Neural Network Architecture

- **Input Layer**: 6 neurons (one-hot encoding of previous moves)
- **Hidden Layer**: 8 neurons with Sigmoid activation
- **Output Layer**: 3 neurons with softmax activation (rock, paper, scissors probabilities)

## Implementation Details

- **Environment.hpp**: Represents the game state and rules
- **NeuralNetwork.hpp/cpp**: Neural network implementation with forward/backward propagation
- **PPOAgent.hpp**: Reinforcement learning agent using PPO algorithm
- **Main.cpp**: Training loop and visualization

## Directory Structure

- `include/` - Header files defining core interfaces
- `src/` - Implementation files with actual neural network logic
- `tests/` - Unit tests for each component

## Building and Running

```bash
# Create a build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run the main application
./src/legacy_neural_rps

# Run the tests (if GTest is installed)
ctest
```

## Standardized Output

This implementation generates standardized output following the format specified in [shared_output_format.md](../shared_output_format.md). The standardized output includes:

- Network architecture details
- Training process metrics 
- Model predictions for each opponent move
- Summary of model parameters

The output is written to `legacy_cpp_demo_output.txt`.

## Training Process

When you run the application, you'll see:
1. The neural network's action probabilities displayed as training progresses
2. Episode rewards gradually increasing as the network learns
3. By the end of training, the network will have learned the optimal strategy
4. Demo games that show the network's learned behavior

## Performance Comparison

Compared to other implementations in this repository:
- More modular code structure than the modern C++ implementation
- Slightly slower training time than the modern Go and C++ implementations
- Same prediction accuracy as other implementations

For a detailed comparison with other implementations, see [rps_model_comparison_report.md](../rps_model_comparison_report.md).

## Dependencies

- C++17 compiler
- CMake 3.14+
- Eigen3 (for matrix operations)
- GoogleTest (optional, for tests)

## Historical Note

This implementation preserves the original structure and functionality of the Neural RPS project before it was split into multiple implementations. The clean separation between headers and implementation makes this version particularly well-suited for educational purposes and further development. 