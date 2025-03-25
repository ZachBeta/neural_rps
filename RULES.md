# Neural Network Rock Paper Scissors

* first pass in cpp rps
* second pass in golang rps
* third pass in golang agains tic tac toe

## Project Overview
This project implements a neural network to play Rock Paper Scissors. The neural network will learn from gameplay data to predict and make optimal moves against human players.

## Code Style Guidelines
- Use modern C++ features (C++17 or later)
- Follow Google C++ Style Guide
- Use clear, descriptive variable and function names
- Include comments for complex logic
- Write unit tests for core functionality

## Project Structure
```
.
├── src/           # Source files
├── include/       # Header files
├── tests/         # Test files
├── data/          # Training data
└── build/         # Build output
```

## Build System
- Use CMake for build configuration
- Require C++17 or later
- External dependencies:
  - Eigen (for matrix operations)
  - GoogleTest (for unit testing)

## Neural Network Architecture
- Input layer: 6 neurons (one-hot encoding of previous moves)
- Hidden layer: 12 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (rock, paper, scissors probabilities)

## Documentation Requirements
- Document all public functions and classes
- Include usage examples in README
- Keep design decisions documented 