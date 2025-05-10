# Supervised Learning and Performance Optimization Plan

This document outlines our strategy for implementing supervised learning to improve neural network performance, optimizing minimax caching, and tracking comparative metrics between supervised and unsupervised approaches.

## Current Status

Our tournament results have established the following baseline:

1. Minimax agents dominate (Minimax-5: 1821 ELO, 92.5% win rate; Minimax-3: 1808 ELO, 86.2% win rate)
2. Random agent performs surprisingly well (1612 ELO)
3. Current neural networks perform below random (1473-1268 ELO)

## Supervised Learning Implementation

### Data Generation

1. **Position Generation:**
   - Use Minimax-5 or Minimax-7 to generate optimal moves for diverse positions
   - Sample from all game phases (opening, midgame, endgame)
   - Generate 10,000-100,000 position/move pairs

2. **Dataset Structure:**
   - Input: Board state, available cards in hand (one-hot encoded)
   - Target: Optimal move from minimax (one-hot encoded)
   - Metadata: Game phase, position evaluation, search depth

3. **Data Distribution:**
   - Ensure balanced representation of opening, midgame, and endgame positions
   - Include positions with varying advantage levels (heavily favoring P1, balanced, heavily favoring P2)
   - Split into training (80%), validation (10%), and test sets (10%)

### Network Training

1. **Architecture:**
   - Input layer: Board state + hand cards (81 features)
   - Hidden layers: Test multiple sizes (64, 128, 256)
   - Output layer: 9 move positions with softmax activation

2. **Training Parameters:**
   - Loss function: Cross-entropy on move selection
   - Optimizer: Adam with learning rate 0.001
   - Batch size: 32-64
   - Epochs: Determined by validation loss plateau

3. **Iterative Improvement:**
   - Train initial network on full dataset
   - Identify positions with poorest agreement
   - Generate additional training data focused on problematic positions
   - Retrain with augmented dataset

## Minimax Caching Optimization

### Performance Monitoring

1. **Metrics to Track:**
   - Cache hit rates (overall and by search depth)
   - Time per position evaluation at different depths
   - Memory consumption during data generation
   - Positions evaluated per second
   - Total data generation throughput

2. **Critical Thresholds:**
   - **Cache Hit Rate:** <20% indicates ineffective caching
   - **Memory Usage:** >500MB indicates need for eviction policy
   - **Generation Speed:** <10 positions/second indicates bottleneck

### Potential Optimizations

1. **Stage 2 LRU Cache:**
   ```go
   // LRUCache implements a thread-safe LRU cache with limited size
   type LRUCache struct {
       capacity int
       mu       sync.RWMutex
       items    map[string]*list.Element
       queue    *list.List
       hits     int
       misses   int
   }
   ```

2. **Persistent Disk Cache:**
   - Save positions to disk between runs
   - Use hash-based directory structure for fast lookups
   - Periodically prune least-used entries

3. **Parallel Processing:**
   - Implement worker pool for data generation
   - Process multiple positions simultaneously
   - Synchronize cache access for thread safety

## Comparative Performance Tracking

### Supervised Learning Metrics

1. **Data Generation:**
   - Total time for dataset creation
   - Positions generated per minute
   - Minimax evaluation time vs. total time

2. **Training Performance:**
   - Move agreement % with minimax (overall)
   - Move agreement % by game phase
   - Epochs to reach target agreement threshold
   - Total training time

3. **Tournament Performance:**
   - ELO rating progression
   - Win rate vs. random baseline
   - Win rate vs. Minimax-3
   - Position-specific performance (opening/midgame/endgame)

### Unsupervised (Self-Play) Metrics

1. **Training Efficiency:**
   - Self-play games per hour
   - Policy/value network loss convergence
   - Time per training iteration
   - Total training time to reach target ELO

2. **Quality Metrics:**
   - ELO progression from initial to final networks
   - Move agreement with minimax
   - Stability of play style

## Implementation Timeline

### Phase 1: Infrastructure (Week 1)

1. Implement data generation script using minimax
2. Add detailed performance instrumentation
3. Create supervised learning pipeline

### Phase 2: Initial Training (Week 2)

1. Generate initial dataset (10K positions)
2. Train baseline neural networks
3. Evaluate against random and minimax agents

### Phase 3: Optimization (Week 3)

1. Identify performance bottlenecks
2. Implement caching improvements if needed
3. Generate expanded dataset

### Phase 4: Advanced Training (Week 4)

1. Train improved networks
2. Conduct full tournament evaluation
3. Compare with self-play approaches

## Success Criteria

1. **Supervised Learning:**
   - Move agreement >50% with minimax (significantly better than random)
   - Win rate >80% against random agent
   - ELO rating at least 1700 (approaching Minimax-3)

2. **Performance Optimization:**
   - Data generation speed >50 positions/second
   - Cache hit rate >50%
   - Memory usage <250MB

3. **Comparative Analysis:**
   - Clear metrics showing strengths/weaknesses of each approach
   - Documented performance/quality tradeoffs
   - Recommendation for hybrid approach if beneficial

This plan provides a structured approach to improving our neural networks through supervised learning while ensuring computational efficiency in our minimax implementation. The performance tracking will help us identify bottlenecks and make informed decisions about further optimizations. 