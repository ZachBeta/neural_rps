# MCTS GPU Acceleration Plan

This document outlines our plan for implementing GPU acceleration for Monte Carlo Tree Search (MCTS) in the Neural Rock Paper Scissors project.

## Implementation Goals

1. Leverage the existing GPU acceleration infrastructure (Python gRPC Service with TensorFlow)
2. Modify MCTS to use batched neural network operations
3. Achieve 5-15x performance improvement in search speed
4. Compare performance metrics between CPU and GPU implementations

## Detailed Implementation Plan

### 1. Batch Collection During Tree Traversal

- Modify MCTS to collect positions during tree traversal instead of evaluating them one-by-one
- Implement a buffer system to accumulate positions requiring neural network evaluation
- When the buffer reaches the optimal batch size (64 based on current benchmarks), trigger a GPU evaluation
- Design the system to handle variable batch sizes based on search conditions

### 2. Key Integration Points

#### Selection Phase
- Batch policy network evaluations for multiple nodes when selecting actions
- Implement a priority queue for positions that need evaluation
- Use cached results when available to reduce duplicate evaluations

#### Expansion Phase
- Batch initialize new nodes when expanding multiple branches
- Share policy network evaluations across related positions

#### Evaluation Phase
- Batch value network evaluations for leaf nodes
- Implement asynchronous evaluation to avoid blocking the search

#### Backpropagation
- Remain as CPU-based operation (already efficient)
- Process results in batches once neural evaluations complete

### 3. BatchedRPSMCTS Implementation

- Create a new `BatchedRPSMCTS` class that extends the current MCTS implementation
- Add batch queues for both policy and value network evaluations
- Implement mechanisms to flush evaluation queues when:
  - Batch size threshold is reached
  - Search requires results to proceed
  - Maximum wait time is exceeded
- Provide compatibility with the existing MCTS interface

### 4. Memory Optimization

- Implement efficient tensor conversion between Go and Python
- Reuse memory buffers for batch operations
- Optimize GPU memory usage during search
- Implement caching strategies to avoid redundant evaluations

### 5. Performance Profiling

- Create benchmarking tools to measure:
  - Nodes explored per second
  - Time per self-play game
  - Memory usage patterns
  - GPU utilization percentage
- Compare against baseline CPU performance
- Analyze the impact of different batch sizes on performance

### 6. Training Comparison Metrics

- Measure and compare:
  - Nodes explored per second
  - Time per self-play game
  - Quality of gameplay at equivalent training time
  - Training convergence rate
  - Overall training time for equivalent performance

## Implementation Timeline

1. **Week 1**: Implement batch collection mechanisms in MCTS
2. **Week 2**: Create BatchedRPSMCTS class and integration with GPU service
3. **Week 3**: Optimize memory usage and batch processing logic
4. **Week 4**: Develop performance profiling tools and run benchmarks
5. **Week 5**: Analyze results, tune parameters, and document findings

## Expected Challenges

1. **Balancing Batch Size vs. Latency**: Finding optimal batch sizes for different search depths
2. **Asynchronous Evaluation**: Managing tree traversal while waiting for batch evaluations
3. **Memory Management**: Efficiently handling large tensors between Go and Python
4. **Cache Coherence**: Ensuring cached evaluations are correctly utilized across batched operations

## Success Criteria

1. Achieve at least 5x speedup in nodes explored per second
2. Demonstrate improved gameplay quality within equivalent training time
3. Maintain or reduce memory overhead compared to CPU implementation
4. Successfully integrate with existing training pipeline 