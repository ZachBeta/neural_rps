# GPU Acceleration Plan for Neural RPS

This document outlines our strategy for implementing GPU acceleration to significantly improve both training and tournament performance in the Neural RPS project.

## Current Performance Bottlenecks

Our codebase has several computationally intensive components that would benefit from GPU acceleration:

1. **Neural Network Forward Pass** - Each prediction for game positions (~30-100μs per position on CPU)
2. **MCTS Search** - Tree traversal with many position evaluations (400+ evaluations per move)
3. **NEAT Training** - Evolutionary algorithm with game simulations for fitness evaluation
4. **Tournament Evaluation** - Head-to-head comparisons between agents

## GPU Acceleration Strategy

### 1. TensorFlow-Go Integration

We'll use TensorFlow's Go bindings to leverage Metal Performance Shaders (MPS) backend on Apple Silicon M1:

```go
// Basic implementation pattern:
import tf "github.com/tensorflow/tensorflow/tensorflow/go"

// Create networks with TensorFlow instead of pure Go
policyNet := neural.NewRPSTFPolicyNetwork(hiddenSize)
```

### 2. Batched Inference for Multiple Positions

Converting sequential evaluations to batched processing:

**Before (CPU):**
```go
// One-by-one evaluation
for _, position := range positions {
    prediction := network.Predict(position)
    // Use prediction...
}
```

**After (GPU):**
```go
// Batch collection
inputs := make([][]float64, len(positions))
for i, position := range positions {
    inputs[i] = position.GetBoardAsFeatures()
}

// Single GPU call for all positions
predictions := network.PredictBatch(inputs)
```

## Implementation Plan

### Phase 1: Core Neural Network Acceleration (1-2 days)

1. ✅ Create `RPSTFPolicyNetwork` implementation using TensorFlow-Go
2. Create `RPSTFValueNetwork` implementation using TensorFlow-Go
3. Implement weight transfer between CPU and GPU networks
4. Add batched inference methods for both networks
5. Profile forward pass performance gains

### Phase 2: MCTS Acceleration (2-3 days)

1. Modify MCTS to collect positions during tree traversal
2. Implement batch evaluation at key search points
3. Create `BatchedRPSMCTS` that uses batched neural network calls
4. Optimize GPU memory usage during search
5. Profile MCTS nodes/second improvement

### Phase 3: NEAT Training Optimization (3-4 days)

1. Modify NEAT evaluator to use batched GPU networks
2. Implement parallel game simulation with GPU evaluation
3. Optimize batch size for different evaluation stages
4. Add checkpointing for GPU-based training
5. Profile time-per-generation improvements

### Phase 4: Tournament System Integration (1-2 days)

1. Update tournament code to use GPU networks
2. Implement multi-game batching for tournament evaluation
3. Create multi-agent GPU tournament system
4. Profile tournament games/minute improvements

## Expected Performance Improvements

| Component | Current Performance | Expected with GPU | Improvement Factor |
|-----------|---------------------|-------------------|-------------------|
| Forward Pass | ~500-1000 pos/sec | ~50,000-100,000 pos/sec | 40-100x |
| MCTS Search | ~500-1000 nodes/sec | ~5,000-10,000 nodes/sec | 5-15x |
| NEAT Generation | ~30 seconds/gen | ~2-5 seconds/gen | 8-20x |
| Tournament | ~1-2 games/sec | ~5-10 games/sec | 3-8x |

## Profiling Methodology

We've created a profiling tool (`cmd/profile_gpu/main.go`) that will:

1. Benchmark CPU vs GPU performance for each component
2. Compare single and batched inference performance
3. Measure memory usage and optimization effects
4. Generate detailed performance reports

Example profiling command:

```bash
go run cmd/profile_gpu/main.go --task prediction --gpu --batch 128 --cpuprofile cpu_prediction.prof
```

## Resource Requirements

- **TensorFlow-Go**: Installation and Go module integration
- **Metal Performance Shaders**: Enabled by default on Apple Silicon
- **Memory**: 2-4GB of GPU memory for larger batches
- **Development Time**: ~8-10 days total implementation

## Prioritization

Implementation priorities based on expected gain:

1. Forward Pass Acceleration (highest ROI)
2. MCTS Batching (critical for gameplay performance)
3. NEAT Training Optimization (largest total time savings)
4. Tournament System Integration

## Conclusion

This GPU acceleration initiative is expected to dramatically improve all aspects of the Neural RPS project, enabling:

- Much faster training iterations
- More extensive tournament evaluations
- Deeper MCTS search in the same time budget
- Support for larger neural network architectures

With these improvements, we'll be able to train more sophisticated agents, evaluate them more thoroughly, and potentially discover stronger strategies for the RPS card game. 