# Minimax as a Control Agent: Evaluation Plan

This document outlines our strategy for using minimax as a benchmark to evaluate the performance of our neural network agents in the RPS card game.

## Purpose and Goals

1. **Establish an Objective Baseline**: The minimax algorithm provides a theoretically optimal strategy within its search depth constraints.

2. **Measure Neural Network Progress**: Track how neural networks improve relative to optimal play through training iterations.

3. **Identify Weaknesses**: Discover specific game situations where neural networks underperform.

4. **Guide Training Improvements**: Use findings to enhance training techniques and network architecture.

## Implementation Strategy

### 1. Control Agent Configurations

We'll use the minimax analyzer at different depths to create a spectrum of opponent difficulties:

| Agent Name | Search Depth | Purpose |
|------------|--------------|---------|
| Minimax-3  | 3 | Fast baseline, represents weaker but quick analysis |
| Minimax-5  | 5 | Balanced approach, moderate strength and speed |
| Minimax-7  | 7 | Strong opponent, approaching optimal play |

All configurations will use:
- Alpha-beta pruning
- Transposition table
- StandardEvaluator for position scoring

### 2. Tournament Setup

**Match Format**:
- Head-to-head matches (neural network vs. minimax)
- 30-50 games per configuration
- Alternating first player
- Complete game playouts (until board filled or no valid moves)

**Time Controls**:
- Neural networks: unlimited (effectively instantaneous)
- Minimax: 3 seconds per move maximum
- Use iterative deepening to ensure minimax always returns valid moves

### 3. Neural Network Contestants

Test a variety of neural networks:
- Different training durations (10, 100, 1000 games)
- Different architectures (64, 128, 256 hidden neurons)
- Different training methods (supervised vs. reinforcement)

## Evaluation Metrics

### 1. Performance Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| Win Rate | % of games won against minimax | `wins / total_games * 100%` |
| Move Agreement | % of moves matching minimax | `matching_moves / total_moves * 100%` |
| Average Blunder | Average eval difference on disagreements | `sum(minimax_eval - network_eval) / disagreements` |
| Decisive Error Rate | % of games with a game-losing mistake | `games_with_decisive_error / total_games * 100%` |

### 2. Position-Based Analysis

Track neural network performance by:
- Game phase (opening, middle, endgame)
- Board patterns (open positions vs. contested positions)
- Material advantage/disadvantage situations

### 3. Decision Time Analysis

Compare decision quality vs. computation time:
- Neural network: near-instant decisions
- Minimax depth 3: ~10ms per move
- Minimax depth 5: ~100ms per move
- Minimax depth 7: ~1s per move

## Implementation Plan

### Phase 1: Basic Tournament

1. Create `MinimaxAgent` that implements the agent interface.
2. Implement tournament runner for minimax vs. neural network.
3. Add basic metrics collection and reporting.
4. Run initial tests with existing neural networks.

### Phase 2: Detailed Analysis

1. Add position classification (opening, middle, endgame).
2. Implement blunder detection and classification.
3. Create visualization for evaluation differences.
4. Generate recommendations for training improvements.

### Phase 3: Hybrid Approach (Optional)

1. Explore using minimax at shallow depths to verify neural network decisions.
2. Test neural network guided minimax search.
3. Implement a hybrid decision maker using both approaches.

## Expected Outcomes

1. **Quantitative Performance Gap**: Clear measurement of how far neural networks are from optimal play.

2. **Pattern Recognition**: Identification of position types where neural networks excel or struggle.

3. **Training Guidance**: Specific recommendations for improving neural network training based on weaknesses identified.

4. **Hybrid Strength**: Potential for combining the strengths of both approaches (neural pattern recognition + minimax calculation).

## Success Criteria

The experiment will be considered successful if we can:

1. Establish a clear correlation between training effort and performance against minimax.
2. Identify at least 3 specific position types where neural networks underperform.
3. Generate concrete recommendations for improving training that result in measurable performance gains.
4. Determine whether the RPS card game is primarily skill-based or luck-based at high levels of play. 