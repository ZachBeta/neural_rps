# AlphaGo RPS Training Improvement Plan

## Current State Analysis

Based on the codebase examination, the AlphaGo implementation has several areas where performance improvements can be made, particularly in:

1. **Parallelization:** The current implementation is mostly single-threaded, with key computation-intensive sections that could benefit from parallelization:
   - Self-play game generation
   - MCTS search simulations
   - Neural network training (forward and backward passes)

2. **Performance Monitoring:** Limited visibility into training progress, performance metrics, and resource utilization.

3. **ELO Tracking:** No current mechanism to track model strength over time using ELO ratings.

4. **Hardware Utilization:** Not fully utilizing the capabilities of the M1 Mac hardware.

## Improvement Plan

### 1. Parallelization Improvements

#### 1.1 Self-Play Game Generation

- Implement parallel game generation using worker pools
- Each worker will generate complete games independently
- Coordinate through a synchronized channel for results collection
- Target: Generate 8-16 games simultaneously on M1 (depends on optimal core utilization)

```go
// Pseudocode for parallel self-play
func (sp *RPSSelfPlay) GenerateGamesParallel(verbose bool) []RPSTrainingExample {
    numWorkers := runtime.NumCPU() // Use all available cores
    gamesChan := make(chan []RPSTrainingExample, sp.params.NumGames)
    var wg sync.WaitGroup
    
    // Create worker pool
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            // Each worker gets a proportional number of games to generate
            gamesPerWorker := sp.params.NumGames / numWorkers
            if workerID == numWorkers-1 {
                // Last worker takes any remainder
                gamesPerWorker += sp.params.NumGames % numWorkers
            }
            
            // Each worker has its own network copies
            localPolicyNet := sp.policyNetwork.Clone()
            localValueNet := sp.valueNetwork.Clone()
            
            for j := 0; j < gamesPerWorker; j++ {
                // Generate a single game
                examples := sp.playGameWithNetworks(localPolicyNet, localValueNet, verbose && j == 0)
                gamesChan <- examples
            }
        }(i)
    }
    
    // Start a goroutine to close the channel once all workers are done
    go func() {
        wg.Wait()
        close(gamesChan)
    }()
    
    // Collect results
    allExamples := make([]RPSTrainingExample, 0)
    for examples := range gamesChan {
        allExamples = append(allExamples, examples...)
    }
    
    return allExamples
}
```

#### 1.2 MCTS Parallel Search

- Parallelize the MCTS simulations using goroutines
- Use a bounded worker pool with shared statistics
- Implement appropriate synchronization for tree updates
- Target: 8-16x speedup on M1 hardware

```go
// Pseudocode for parallel MCTS search
func (mcts *RPSMCTS) SearchParallel() *RPSMCTSNode {
    if mcts.Root == nil {
        return nil
    }
    
    // Expand the root node if needed
    if len(mcts.Root.Children) == 0 {
        priors := mcts.PolicyNetwork.Predict(mcts.Root.GameState)
        mcts.Root.ExpandAll(priors)
    }
    
    numWorkers := runtime.NumCPU()
    simulationsPerWorker := mcts.Params.NumSimulations / numWorkers
    var wg sync.WaitGroup
    nodeMutex := &sync.Mutex{} // For tree synchronization
    
    // Run parallel simulations
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            // Each worker performs its share of simulations
            for j := 0; j < simulationsPerWorker; j++ {
                // Selection phase (needs read locks)
                node := mcts.selectionThreadSafe(mcts.Root, nodeMutex)
                
                // Take write lock for node expansion
                nodeMutex.Lock()
                if !node.GameState.IsGameOver() && node.Visits > 0 && len(node.Children) == 0 {
                    priors := mcts.PolicyNetwork.Predict(node.GameState)
                    node.ExpandAll(priors)
                    if len(node.Children) > 0 {
                        node = node.Children[0]
                    }
                }
                nodeMutex.Unlock()
                
                // Evaluation (independent)
                value := mcts.evaluate(node)
                
                // Backpropagation (needs write locks)
                mcts.backpropagateThreadSafe(node, value, nodeMutex)
            }
        }()
    }
    
    wg.Wait()
    
    // Return the most visited child of the root
    return mcts.Root.MostVisitedChild()
}
```

#### 1.3 Neural Network Batch Processing

- Implement mini-batch training with parallel forward/backward passes
- Use optimized matrix operations via a more efficient neural network library
- Consider replacing the manual neural net implementation with TensorFlow Go or Gorgonia
- Target: 4-10x training speedup

### 2. Performance Monitoring and Visibility

#### 2.1 Training Progress Dashboard

- Create a real-time training progress visualization tool
- Metrics to track:
  - Loss trends (policy and value)
  - Games per second
  - Search simulations per second
  - Resources utilization (CPU, memory)
  - Estimated time remaining

#### 2.2 Resource Utilization Monitoring

- Add CPU, memory and temperature monitoring during training
- Implement adaptive scaling of parallelism based on temperature
- Create utilization graphs to identify bottlenecks

#### 2.3 Game Quality Metrics

- Track metrics like average game length, move diversity
- Identify patterns or weaknesses in play style
- Create position heatmaps to visualize board coverage

### 3. ELO Rating System

#### 3.1 ELO Tracking Implementation

- Implement an ELO rating system to track model progress
- Save checkpoints of models at regular intervals
- Run tournaments between checkpoints to measure improvement
- Create an ELO history graph

```go
// Pseudocode for ELO rating system
type ELOTracker struct {
    BaseRating     float64
    ModelRatings   map[string]float64
    MatchHistory   []MatchResult
}

func (e *ELOTracker) UpdateRating(model1, model2 string, result float64) {
    // result: 1.0 for model1 win, 0.5 for draw, 0.0 for model2 win
    rating1 := e.ModelRatings[model1]
    rating2 := e.ModelRatings[model2]
    
    // Expected scores
    expected1 := 1.0 / (1.0 + math.Pow(10, (rating2-rating1)/400.0))
    expected2 := 1.0 / (1.0 + math.Pow(10, (rating1-rating2)/400.0))
    
    // K-factor (importance of match)
    k := 32.0
    
    // Update ratings
    e.ModelRatings[model1] += k * (result - expected1)
    e.ModelRatings[model2] += k * (1.0 - result - expected2)
    
    // Record match
    e.MatchHistory = append(e.MatchHistory, MatchResult{
        Model1: model1,
        Model2: model2,
        Result: result,
        NewRating1: e.ModelRatings[model1],
        NewRating2: e.ModelRatings[model2],
    })
}
```

#### 3.2 Tournament Management

- Create a tournament manager to regularly evaluate models
- Include different opponent types (rule-based, random, previous versions)
- Generate win probability matrices between models
- Visualize ELO progression over training time

### 4. Hardware Optimization for M1 Mac

#### 4.1 M1-Specific Optimizations

- Utilize Apple Metal for GPU acceleration where possible
- Configure optimal thread counts for M1 architecture (efficiency vs. performance cores)
- Profile and optimize memory access patterns
- Test different batch sizes to find optimal throughput

#### 4.2 Memory Optimization

- Implement more efficient data structures for game state representation
- Reduce memory allocations during MCTS search
- Share neural network parameters across goroutines when appropriate
- Monitor and optimize garbage collection pauses

## Implementation Plan

### Phase 1: Performance Baseline and Monitoring (Week 1)

1. Create performance benchmarks for current implementation
2. Implement basic monitoring and metrics collection
3. Establish ELO rating system
4. Create visualization tools for metrics

### Phase 2: Parallelization (Week 2-3)

1. Implement parallel self-play generation
2. Add parallel MCTS search
3. Optimize neural network operations
4. Benchmark and validate improvements

### Phase 3: Advanced Features and Refinements (Week 4)

1. Implement tournament automation for ELO tracking
2. Fine-tune parallelization parameters for M1 hardware
3. Create advanced visualizations for model strength progression
4. Document optimizations and performance characteristics

## Success Metrics

1. Training speed: 8-10x faster end-to-end training
2. Hardware utilization: 80%+ CPU utilization during intensive operations
3. Quality metrics: Monitor ELO progression to ensure optimization doesn't sacrifice model strength
4. Visibility: Complete dashboard of training progress and performance

## Conclusion

This improvement plan focuses on making the AlphaGo RPS implementation more efficient through parallelization, better monitoring, and hardware optimization. The changes should result in significantly faster training times while maintaining or improving model quality. 