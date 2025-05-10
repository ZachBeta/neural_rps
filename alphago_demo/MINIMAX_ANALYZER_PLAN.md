# RPS Card Game Minimax Analyzer Implementation Plan

## Overview

This document outlines the implementation plan for a deterministic minimax analyzer for the Rock-Paper-Scissors card game. The analyzer will serve as an objective benchmark to evaluate the quality of trained neural network models without influencing the training process itself.

## Objectives

1. Create a reference implementation that can identify optimal or near-optimal moves in specific game states
2. Develop a suite of benchmark positions to test model decision quality
3. Provide objective metrics for comparing different training approaches
4. Enhance our understanding of what makes a strong RPS card game agent

## Technical Implementation

### 1. Minimax Search Engine

```go
// Core minimax search function with alpha-beta pruning
func minimax(state *game.RPSGame, depth int, alpha, beta float64, maximizingPlayer bool) (float64, game.RPSMove) {
    // Base case: terminal node or max depth reached
    if depth == 0 || state.IsGameOver() {
        return evaluatePosition(state), game.RPSMove{}
    }
    
    validMoves := state.GetValidMoves()
    
    if maximizingPlayer {
        maxEval := math.Inf(-1)
        var bestMove game.RPSMove
        
        for _, move := range validMoves {
            nextState := state.Copy()
            nextState.MakeMove(move)
            eval, _ := minimax(nextState, depth-1, alpha, beta, false)
            
            if eval > maxEval {
                maxEval = eval
                bestMove = move
            }
            alpha = math.Max(alpha, eval)
            if beta <= alpha {
                break // Beta cutoff
            }
        }
        return maxEval, bestMove
    } else {
        minEval := math.Inf(1)
        var bestMove game.RPSMove
        
        for _, move := range validMoves {
            nextState := state.Copy()
            nextState.MakeMove(move)
            eval, _ := minimax(nextState, depth-1, alpha, beta, true)
            
            if eval < minEval {
                minEval = eval
                bestMove = move
            }
            beta = math.Min(beta, eval)
            if beta <= alpha {
                break // Alpha cutoff
            }
        }
        return minEval, bestMove
    }
}
```

### 2. Position Evaluation Function

```go
// Evaluate a non-terminal board position
func evaluatePosition(state *game.RPSGame) float64 {
    // Material advantage (difference in number of cards on board)
    p1Cards := state.CountPlayerCards(game.Player1)
    p2Cards := state.CountPlayerCards(game.Player2)
    materialScore := float64(p1Cards - p2Cards)
    
    // Positional advantage (control of key positions like center and corners)
    positionalScore := evaluatePositionalAdvantage(state)
    
    // Card type advantage (assess RPS relationships of adjacent cards)
    relationshipScore := evaluateCardRelationships(state)
    
    // Combine factors with appropriate weights
    return materialScore*1.0 + positionalScore*0.5 + relationshipScore*0.8
}
```

### 3. Depth Management

Due to the branching factor of the RPS card game, full-depth search may be impractical. We'll implement:

- Iterative deepening for time-bounded analysis
- Variable depth based on game progression (deeper search in endgame)
- Move ordering heuristics to improve alpha-beta pruning efficiency

## Benchmark Suite

### 1. Position Categories

1. **Early Game Positions** (5-7 plies deep)
   - Focus on initial development and board control
   - Test strategic understanding of card placement

2. **Midgame Tactical Positions** (3-5 plies deep)
   - Positions with clear tactical opportunities
   - Test ability to find capturing sequences

3. **Endgame Positions** (1-3 plies deep) 
   - Positions with forced wins or key defensive moves
   - Test technical precision

### 2. Position Selection Criteria

- Clear optimal solutions exist
- Mixture of obvious and subtle best moves
- Cover diverse strategic themes
- Represent realistic game situations

### 3. Test Suite Format

Each test position will be stored as:
- Game state serialization
- Optimal move(s)
- Difficulty rating (1-5)
- Strategic theme/lesson

## Metrics and Analysis

### 1. Quality Metrics

- **Optimal Move Rate**: Percentage of positions where model finds the best move
- **Average Move Quality**: How close model moves are to optimal (0-1 scale)
- **Blunder Rate**: Frequency of clearly inferior moves
- **Decision Time**: How quickly models reach their decisions

### 2. Comparative Analysis

- Side-by-side comparison of different models on identical positions
- Breakdown of performance by position category
- Identification of systematic strengths/weaknesses

### 3. Visualization

- Heat maps showing move preference differences
- Decision tree visualizations for key positions
- Performance graphs across different position types

## Integration with Existing Code

### 1. Analyzer Interface

```go
type MinimaxAnalyzer struct {
    maxDepth int
    positions []*BenchmarkPosition
    // Additional configuration
}

func NewMinimaxAnalyzer(maxDepth int) *MinimaxAnalyzer {
    // Initialize analyzer with default benchmark positions
}

func (m *MinimaxAnalyzer) AnalyzeAgent(agent Agent) *AnalysisReport {
    // Run agent against benchmark positions
    // Compare with minimax solutions
    // Generate comprehensive report
}
```

### 2. Command Line Tool

Create a standalone command for analyzing agents:

```
go run cmd/analyze_model/main.go --model-path=path/to/model --depth=5 --positions=standard
```

## Implementation Timeline

1. **Phase 1**: Core minimax implementation with basic evaluation (1-2 days)
2. **Phase 2**: Position suite development and validation (1-2 days)
3. **Phase 3**: Integration with model loading and analysis reporting (1 day)
4. **Phase 4**: Optimization and performance tuning (1 day)

## Expected Outcomes

1. Objective quality metrics for comparing different training methods
2. Insights into strengths/weaknesses of neural approaches
3. Better understanding of the strategic depth of the RPS card game
4. Potential hybrid approaches combining neural and tree search methods

## Future Extensions

1. Use minimax as an opponent during training
2. Integrate with MCTS for more powerful search
3. Expand to analyze other similar games
4. Add pruning heuristics specific to RPS card game 