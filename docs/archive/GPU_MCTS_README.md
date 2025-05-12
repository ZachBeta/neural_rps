# GPU-Accelerated MCTS Implementation

This README provides information on how to use the GPU-accelerated Monte Carlo Tree Search (MCTS) implementation in the Neural Rock Paper Scissors project.

## Overview

The `GPUBatchedMCTS` implementation enhances the standard MCTS algorithm with GPU-accelerated batch processing for neural network operations, providing significant performance improvements for both policy and value network evaluations.

Key features:
- Batched neural network evaluations using TensorFlow with Metal GPU acceleration
- Asynchronous processing with background workers
- Client-server architecture via gRPC
- Configurable batch sizes for optimal performance
- Automatic fallback to CPU evaluation on errors
- Comprehensive performance statistics

## Requirements

- Go 1.16 or higher
- TensorFlow 2.5 or higher with GPU support (Metal for Apple Silicon)
- Python 3.8 or higher
- gRPC tools installed

## Getting Started

### 1. Start the Neural Service

Before using the GPU-accelerated MCTS, ensure the neural service is running:

```bash
./start_neural_service.sh
```

This script starts the TensorFlow gRPC service that provides GPU-accelerated neural network operations.

### 2. Using GPUBatchedMCTS

Import the necessary packages:

```go
import (
    "context"
    "github.com/zachbeta/neural_rps/pkg/agents/mcts"
    "github.com/zachbeta/neural_rps/pkg/game"
)
```

Create and configure a GPU-accelerated MCTS agent:

```go
// Create GPU-accelerated MCTS with default parameters
params := mcts.DefaultMCTSParams()
params.NumSimulations = 1000 // Configure search depth

agent, err := mcts.NewGPUBatchedMCTS("localhost:50051", params)
if err != nil {
    panic(err)
}
defer agent.Close() // Don't forget to close to release resources

// Set the game state
gameState := game.NewRPSCardGame()
agent.SetRootState(gameState)

// Run the search with a context
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Get the best move
bestMove := agent.Search(ctx)
```

### 3. Configuration

You can configure the GPU-accelerated MCTS with the following parameters:

```go
// Create custom MCTS parameters
params := mcts.MCTSParams{
    NumSimulations:   2000,   // Number of MCTS iterations
    ExplorationConst: 1.4,    // Exploration constant for UCB
}

// Create the agent with a custom batch size
agent, err := mcts.NewGPUBatchedMCTS("localhost:50051", params)
if err != nil {
    panic(err)
}

// Customize batch size (defaults to 64)
agent.SetBatchSize(128)

// Customize maximum wait time (defaults to 5ms)
agent.SetMaxWaitTime(10 * time.Millisecond)
```

## Benchmark Tool

We provide a benchmark tool to compare CPU and GPU MCTS performance:

```bash
# Run GPU benchmark with 1000 iterations
go run cmd/tools/benchmark_gpu_mcts.go -n 1000

# Run CPU benchmark with 1000 iterations
go run cmd/tools/benchmark_gpu_mcts.go -cpu -n 1000

# Run GPU benchmark with custom batch size
go run cmd/tools/benchmark_gpu_mcts.go -n 1000 -batch 128

# Run comprehensive benchmarks
./scripts/run_gpu_mcts_benchmark.sh
```

## Performance Tuning

For optimal performance:

1. **Batch Size**: The ideal batch size depends on your hardware. For most GPUs, batch sizes between 32-128 provide the best performance.

2. **Max Wait Time**: This controls how long to wait before flushing non-full batches. Lower values improve responsiveness but may reduce batch efficiency.

3. **Search Depth**: The performance benefits of GPU acceleration increase with search depth. For shallow searches (< 100 iterations), CPU may be faster due to lower overhead.

4. **Memory Management**: The implementation automatically manages memory pools for tensors, but for very large searches, monitor memory usage.

## Performance Statistics

You can retrieve performance statistics:

```go
stats := agent.GetStats()
fmt.Printf("Total policy batches: %d\n", stats["total_policy_batches"])
fmt.Printf("Average policy batch size: %.2f\n", stats["avg_policy_batch_size"])
fmt.Printf("Average policy latency: %.2f µs\n", stats["avg_policy_latency_us"])
```

## Troubleshooting

If you encounter issues:

1. **Connection Errors**: Ensure the neural service is running with `./start_neural_service.sh`

2. **Performance Issues**: Try different batch sizes (64 is a good default)

3. **Memory Errors**: Reduce the search depth or batch size

4. **GPU Utilization**: Monitor GPU utilization with:
   - Apple Silicon: `sudo powermetrics --samplers gpu_power -i 1000`
   - NVIDIA: `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` 