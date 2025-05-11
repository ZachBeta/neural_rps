# MCTS GPU Acceleration Tutorial

This tutorial guides mid-level software engineers through implementing GPU acceleration for Monte Carlo Tree Search (MCTS) in the Neural Rock Paper Scissors project.

## Prerequisites

- Understanding of MCTS algorithm fundamentals
- Basic knowledge of Go programming
- Familiarity with neural networks and TensorFlow
- Experience with gRPC communication

## Setup

1. Ensure you have the GPU acceleration infrastructure installed:
   ```bash
   cd neural_rps
   ./scripts/setup_gpu_service.sh
   ```

2. Test the existing GPU service:
   ```bash
   go run cmd/tools/test_gpu_service.go
   ```
   You should see successful connection and benchmark results.

## Part 1: Understanding the Current Architecture

Our GPU acceleration uses a client-server architecture:

1. **Python gRPC Service**: TensorFlow-based service leveraging Metal GPU acceleration
2. **Go gRPC Client**: Communicates with the Python service
3. **Protocol Buffer Interface**: Defines the neural network operations

The current implementation only accelerates individual neural network operations. Our goal is to modify MCTS to use batched operations for significant performance gains.

## Part 2: Creating the BatchedRPSMCTS Class

Start by creating a new file `pkg/mcts/batched_mcts.go`:

```go
package mcts

import (
    "context"
    
    "github.com/ZachBeta/neural_rps/pkg/game"
    "github.com/ZachBeta/neural_rps/pkg/neural/gpu"
)

// BatchSize is the default batch size for neural network evaluations
const DefaultBatchSize = 64

// BatchedRPSMCTS extends the standard MCTS with batched neural network operations
type BatchedRPSMCTS struct {
    *RPSMCTS
    
    // Batch queues
    policyQueue []BatchItem
    valueQueue  []BatchItem
    
    // Configuration
    batchSize  int
    maxWaitTime time.Duration
    
    // GPU client
    gpuClient *gpu.Client
}

// BatchItem represents a position waiting for neural network evaluation
type BatchItem struct {
    state    *game.State
    resultCh chan<- float32
}

// NewBatchedRPSMCTS creates a new batched MCTS instance
func NewBatchedRPSMCTS(config Config, gpuClient *gpu.Client) *BatchedRPSMCTS {
    baseMCTS := NewRPSMCTS(config)
    
    return &BatchedRPSMCTS{
        RPSMCTS:     baseMCTS,
        policyQueue: make([]BatchItem, 0, DefaultBatchSize),
        valueQueue:  make([]BatchItem, 0, DefaultBatchSize),
        batchSize:   DefaultBatchSize,
        maxWaitTime: 5 * time.Millisecond,
        gpuClient:   gpuClient,
    }
}
```

## Part 3: Implementing Batch Collection

Modify the MCTS algorithm to collect positions during tree traversal:

```go
// EvaluatePolicy batches policy network evaluations
func (b *BatchedRPSMCTS) EvaluatePolicy(ctx context.Context, state *game.State) ([]float32, error) {
    // Fast path: check cache first
    if probs, ok := b.policyCache.Get(state.Hash()); ok {
        return probs, nil
    }
    
    // Create a channel to receive results
    resultCh := make(chan []float32, 1)
    
    // Add to queue
    b.policyQueueMu.Lock()
    b.policyQueue = append(b.policyQueue, BatchItem{
        state:    state,
        resultCh: resultCh,
    })
    
    // If queue is full or context deadline approaching, flush the queue
    shouldFlush := len(b.policyQueue) >= b.batchSize
    b.policyQueueMu.Unlock()
    
    if shouldFlush {
        b.flushPolicyQueue(ctx)
    }
    
    // Wait for result
    select {
    case result := <-resultCh:
        return result, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

// flushPolicyQueue evaluates all queued positions with the policy network
func (b *BatchedRPSMCTS) flushPolicyQueue(ctx context.Context) {
    b.policyQueueMu.Lock()
    if len(b.policyQueue) == 0 {
        b.policyQueueMu.Unlock()
        return
    }
    
    // Extract batch
    batch := b.policyQueue
    b.policyQueue = make([]BatchItem, 0, b.batchSize)
    b.policyQueueMu.Unlock()
    
    // Prepare batch input
    inputs := make([][]float32, len(batch))
    for i, item := range batch {
        inputs[i] = item.state.ToTensor()
    }
    
    // Evaluate batch
    outputs, err := b.gpuClient.EvaluatePolicyBatch(ctx, inputs)
    
    // Distribute results
    for i, item := range batch {
        var result []float32
        if err != nil {
            // Fallback to CPU evaluation on error
            result, _ = b.RPSMCTS.evaluatePolicy(ctx, item.state)
        } else {
            result = outputs[i]
            // Update cache
            b.policyCache.Put(item.state.Hash(), result)
        }
        
        // Send result back
        item.resultCh <- result
    }
}
```

Implement similar code for value network evaluation.

## Part 4: Implementing Asynchronous Evaluation

To avoid blocking the search, implement a background worker:

```go
// Start background workers for batch processing
func (b *BatchedRPSMCTS) StartWorkers(ctx context.Context) {
    go b.policyWorker(ctx)
    go b.valueWorker(ctx)
}

func (b *BatchedRPSMCTS) policyWorker(ctx context.Context) {
    ticker := time.NewTicker(b.maxWaitTime)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            // Flush queue periodically even if not full
            b.flushPolicyQueue(ctx)
        }
    }
}
```

## Part 5: Integrating with Search

Replace the standard MCTS search with the batched version:

```go
// Search runs the MCTS algorithm with batched operations
func (b *BatchedRPSMCTS) Search(ctx context.Context, root *game.State, iterations int) *Node {
    // Start background workers
    workerCtx, cancel := context.WithCancel(ctx)
    defer cancel()
    b.StartWorkers(workerCtx)
    
    // Create root node
    rootNode := b.newNode(root)
    
    // Run search iterations
    for i := 0; i < iterations; i++ {
        select {
        case <-ctx.Done():
            return rootNode
        default:
            b.iteration(ctx, rootNode)
        }
    }
    
    return rootNode
}
```

## Part 6: Profiling and Benchmarking

Create a benchmark tool to measure performance:

```go
// cmd/tools/benchmark_mcts.go
package main

import (
    "context"
    "flag"
    "fmt"
    "time"
    
    "github.com/ZachBeta/neural_rps/pkg/game"
    "github.com/ZachBeta/neural_rps/pkg/mcts"
    "github.com/ZachBeta/neural_rps/pkg/neural/gpu"
)

func main() {
    useCPU := flag.Bool("cpu", false, "Use CPU-only MCTS")
    iterations := flag.Int("n", 1000, "Number of iterations")
    flag.Parse()
    
    // Initialize game state
    state := game.NewState()
    
    // Set up MCTS
    var search mcts.Searcher
    
    if *useCPU {
        config := mcts.DefaultConfig()
        search = mcts.NewRPSMCTS(config)
    } else {
        // Initialize GPU client
        client, err := gpu.NewClient("localhost:50051")
        if err != nil {
            panic(err)
        }
        defer client.Close()
        
        config := mcts.DefaultConfig()
        search = mcts.NewBatchedRPSMCTS(config, client)
    }
    
    // Run benchmark
    ctx := context.Background()
    start := time.Now()
    node := search.Search(ctx, state, *iterations)
    elapsed := time.Since(start)
    
    // Print results
    nodesPerSecond := float64(*iterations) / elapsed.Seconds()
    fmt.Printf("Search completed in %v\n", elapsed)
    fmt.Printf("Nodes per second: %.2f\n", nodesPerSecond)
    fmt.Printf("Best move: %s\n", node.BestAction())
}
```

Run the benchmark:
```bash
# CPU benchmark
go run cmd/tools/benchmark_mcts.go -cpu

# GPU benchmark
go run cmd/tools/benchmark_mcts.go
```

## Part 7: Optimization Tips

1. **Batch Size Tuning**: The optimal batch size depends on your hardware. Start with 64 and experiment.

2. **Cache Management**: Implement an LRU cache to avoid redundant evaluations:
   ```go
   import "github.com/hashicorp/golang-lru"
   
   cache, _ := lru.New(10000) // Cache size
   ```

3. **Asynchronous Processing**: Make your batch processing truly asynchronous to allow the search to continue:
   ```go
   // Submit evaluation task and continue search
   go func() {
       result := <-resultCh
       node.processResult(result)
   }()
   ```

4. **Memory Optimization**: Reuse tensors to reduce memory allocations:
   ```go
   // Reuse tensor pool
   tensorPool := sync.Pool{
       New: func() interface{} {
           return make([]float32, inputSize)
       },
   }
   ```

## Conclusion

You now have a GPU-accelerated MCTS implementation that leverages batched neural network operations. This approach should provide significant speedup compared to the CPU-only version, especially as the number of search iterations increases.

Remember to:
1. Test thoroughly with different batch sizes
2. Profile memory usage to avoid leaks
3. Compare gameplay quality between CPU and GPU versions
4. Integrate with the training pipeline

For any issues or questions, check the project documentation or file an issue on GitHub. 