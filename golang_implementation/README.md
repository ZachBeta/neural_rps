# Neural RPS - Golang Implementation

This directory contains the initial Golang implementation of the Neural Rock Paper Scissors project.

## Overview

This Golang implementation offers better readability, performance, and a more ergonomic development/build/test setup compared to the C++ version. It uses Go's concurrency features for optimal performance.

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

- Neural network with one hidden layer
- PPO algorithm for policy optimization
- Game environment with state tracking
- Visualization of network architecture and training progress
- Uses Go's concurrency features for parallelized training 