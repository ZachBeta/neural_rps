# Implementing Minimax for the RPS Card Game: A Tutorial

## Introduction

This tutorial guides you through implementing a minimax algorithm with alpha-beta pruning for the Rock-Paper-Scissors card game. We'll create a deterministic analyzer that can find optimal moves and evaluate neural network performance.

This guide assumes:
- You're familiar with Go programming
- You understand basic game theory concepts
- You've worked with tree-based algorithms before

## Understanding Minimax and Alpha-Beta Pruning

Minimax is a decision-making algorithm for determining optimal play in zero-sum games. It recursively evaluates possible moves by simulating the game forward to a certain depth, then propagating values upward through the game tree.

Alpha-beta pruning is an optimization that eliminates branches that cannot influence the final decision, dramatically improving performance.

### Key Principles

1. The algorithm alternates between minimizing and maximizing players
2. The maximizing player (typically Player 1) tries to maximize the score
3. The minimizing player (typically Player 2) tries to minimize the score
4. Alpha-beta pruning maintains bounds on possible scores to eliminate branches

## Implementation Steps

### 1. Setting Up the Package Structure

First, create the analysis package directory:

```bash
mkdir -p pkg/analysis
touch pkg/analysis/minimax.go
touch pkg/analysis/evaluation.go
```

### 2. Implementing the Minimax Core (minimax.go)

```go
package analysis

import (
	"math"
	"time"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// MinimaxEngine implements an alpha-beta search for RPS card game
type MinimaxEngine struct {
	MaxDepth       int
	NodesEvaluated int
	MaxTime        time.Duration
	StartTime      time.Time
	EvaluationFn   func(*game.RPSGame) float64
}

// NewMinimaxEngine creates a new minimax search engine
func NewMinimaxEngine(maxDepth int, evalFn func(*game.RPSGame) float64) *MinimaxEngine {
	return &MinimaxEngine{
		MaxDepth:     maxDepth,
		EvaluationFn: evalFn,
		MaxTime:      30 * time.Second, // Default 30-second time limit
	}
}

// FindBestMove returns the best move for the current player
func (m *MinimaxEngine) FindBestMove(state *game.RPSGame) (game.RPSMove, float64) {
	m.NodesEvaluated = 0
	m.StartTime = time.Now()
	
	// Initialize alpha-beta bounds
	alpha := math.Inf(-1)
	beta := math.Inf(1)
	
	// Determine if current player is maximizing
	maximizingPlayer := state.CurrentPlayer == game.Player1
	
	// Call minimax search
	value, move := m.minimax(state, m.MaxDepth, alpha, beta, maximizingPlayer)
	
	return move, value
}

// minimax performs alpha-beta pruned minimax search
func (m *MinimaxEngine) minimax(state *game.RPSGame, depth int, alpha, beta float64, maximizingPlayer bool) (float64, game.RPSMove) {
	m.NodesEvaluated++
	
	// Check for timeout
	if time.Since(m.StartTime) > m.MaxTime {
		return m.EvaluationFn(state), game.RPSMove{}
	}
	
	// Base case: terminal node or max depth reached
	if depth == 0 || state.IsGameOver() {
		return m.EvaluationFn(state), game.RPSMove{}
	}
	
	validMoves := state.GetValidMoves()
	
	// No valid moves (shouldn't happen if IsGameOver is correct, but just in case)
	if len(validMoves) == 0 {
		return m.EvaluationFn(state), game.RPSMove{}
	}
	
	var bestMove game.RPSMove
	
	if maximizingPlayer {
		maxEval := math.Inf(-1)
		
		for _, move := range validMoves {
			// Create a copy of the state and apply the move
			nextState := state.Copy()
			moveCopy := move // Create a copy to avoid reference issues
			moveCopy.Player = nextState.CurrentPlayer
			
			err := nextState.MakeMove(moveCopy)
			if err != nil {
				continue // Skip invalid moves
			}
			
			// Recursively evaluate the resulting position
			eval, _ := m.minimax(nextState, depth-1, alpha, beta, !maximizingPlayer)
			
			// Update maxEval and bestMove if we found a better move
			if eval > maxEval {
				maxEval = eval
				bestMove = move
			}
			
			// Update alpha
			alpha = math.Max(alpha, eval)
			
			// Alpha-beta pruning
			if beta <= alpha {
				break
			}
		}
		
		return maxEval, bestMove
	} else {
		minEval := math.Inf(1)
		
		for _, move := range validMoves {
			// Create a copy of the state and apply the move
			nextState := state.Copy()
			moveCopy := move // Create a copy to avoid reference issues
			moveCopy.Player = nextState.CurrentPlayer
			
			err := nextState.MakeMove(moveCopy)
			if err != nil {
				continue // Skip invalid moves
			}
			
			// Recursively evaluate the resulting position
			eval, _ := m.minimax(nextState, depth-1, alpha, beta, !maximizingPlayer)
			
			// Update minEval and bestMove if we found a better move
			if eval < minEval {
				minEval = eval
				bestMove = move
			}
			
			// Update beta
			beta = math.Min(beta, eval)
			
			// Alpha-beta pruning
			if beta <= alpha {
				break
			}
		}
		
		return minEval, bestMove
	}
}
```

### 3. Position Evaluation (evaluation.go)

```go
package analysis

import (
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// StandardEvaluator provides a comprehensive evaluation function
func StandardEvaluator(state *game.RPSGame) float64 {
	if state.IsGameOver() {
		winner := state.GetWinner()
		if winner == game.Player1 {
			return 1000.0 // Large positive value for Player1 win
		} else if winner == game.Player2 {
			return -1000.0 // Large negative value for Player2 win
		}
		return 0.0 // Draw
	}
	
	// Combine multiple evaluation factors with appropriate weights
	return materialScore(state)*1.0 + positionalScore(state)*0.5 + relationshipScore(state)*0.8
}

// materialScore evaluates the material advantage (difference in number of cards)
func materialScore(state *game.RPSGame) float64 {
	p1Cards := state.CountPlayerCards(game.Player1)
	p2Cards := state.CountPlayerCards(game.Player2)
	return float64(p1Cards - p2Cards) * 10.0
}

// positionalScore evaluates board control and positioning
func positionalScore(state *game.RPSGame) float64 {
	score := 0.0
	
	// Define position values (center is most valuable, corners next)
	positionValues := [][]float64{
		{0.7, 0.5, 0.7}, // Top row
		{0.5, 1.0, 0.5}, // Middle row (center = 1.0)
		{0.7, 0.5, 0.7}, // Bottom row
	}
	
	// Calculate positional score based on occupied positions
	board := state.GetBoard()
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			cell := board[row][col]
			if cell.Player == game.Player1 {
				score += positionValues[row][col]
			} else if cell.Player == game.Player2 {
				score -= positionValues[row][col]
			}
		}
	}
	
	return score * 5.0 // Weight position appropriately
}

// relationshipScore evaluates the RPS relationships between adjacent cards
func relationshipScore(state *game.RPSGame) float64 {
	score := 0.0
	board := state.GetBoard()
	
	// Define directional offsets to check adjacency (horizontal, vertical, diagonal)
	directions := []struct{ dRow, dCol int }{
		{0, 1}, {1, 0}, {1, 1}, {1, -1}, // Right, Down, Diagonal down-right, Diagonal down-left
	}
	
	// Check each cell on the board
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			cell := board[row][col]
			if cell.Player == game.NoPlayer {
				continue // Skip empty cells
			}
			
			// Check adjacent cells in all directions
			for _, dir := range directions {
				newRow, newCol := row+dir.dRow, col+dir.dCol
				
				// Check if position is within bounds
				if newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3 {
					adjCell := board[newRow][newCol]
					if adjCell.Player != game.NoPlayer && adjCell.Player != cell.Player {
						// Calculate advantage based on RPS relationships
						advantage := getCardAdvantage(cell.CardType, adjCell.CardType)
						
						if cell.Player == game.Player1 {
							score += advantage
						} else {
							score -= advantage
						}
					}
				}
			}
		}
	}
	
	return score * 3.0 // Weight relationships appropriately
}

// getCardAdvantage returns the advantage of card1 over card2
// 1.0 if card1 beats card2, -1.0 if card2 beats card1, 0.0 if tie
func getCardAdvantage(card1, card2 game.CardType) float64 {
	if card1 == card2 {
		return 0.0 // Same card type
	}
	
	// Rock beats Scissors, Scissors beats Paper, Paper beats Rock
	if (card1 == game.Rock && card2 == game.Scissors) ||
	   (card1 == game.Scissors && card2 == game.Paper) ||
	   (card1 == game.Paper && card2 == game.Rock) {
		return 1.0
	}
	
	return -1.0
}
```

## Performance Optimization Techniques

### 1. Move Ordering

To improve alpha-beta pruning efficiency, implement move ordering:

```go
// orderMoves prioritizes moves that are likely to be best
func (m *MinimaxEngine) orderMoves(state *game.RPSGame, moves []game.RPSMove) []game.RPSMove {
	// Create a slice of move-score pairs
	type moveScore struct {
		move  game.RPSMove
		score float64
	}
	
	scoredMoves := make([]moveScore, 0, len(moves))
	
	// Score each move with a simple heuristic
	for _, move := range moves {
		nextState := state.Copy()
		moveCopy := move
		moveCopy.Player = nextState.CurrentPlayer
		
		if err := nextState.MakeMove(moveCopy); err != nil {
			continue
		}
		
		// Use a simplified evaluation for move ordering
		score := m.quickEvaluate(nextState)
		
		// If minimizing player, negate score for proper ordering
		if nextState.CurrentPlayer == game.Player2 {
			score = -score
		}
		
		scoredMoves = append(scoredMoves, moveScore{move, score})
	}
	
	// Sort moves by score (descending)
	sort.Slice(scoredMoves, func(i, j int) bool {
		return scoredMoves[i].score > scoredMoves[j].score
	})
	
	// Extract sorted moves
	sortedMoves := make([]game.RPSMove, 0, len(scoredMoves))
	for _, ms := range scoredMoves {
		sortedMoves = append(sortedMoves, ms.move)
	}
	
	return sortedMoves
}

// quickEvaluate provides a simpler, faster evaluation for move ordering
func (m *MinimaxEngine) quickEvaluate(state *game.RPSGame) float64 {
	// Just use material difference for quick evaluation
	p1Cards := state.CountPlayerCards(game.Player1)
	p2Cards := state.CountPlayerCards(game.Player2)
	return float64(p1Cards - p2Cards) * 10.0
}
```

### 2. Transposition Table

Implement a transposition table to avoid redundant calculations:

```go
type TranspositionTable struct {
	entries map[string]TranspositionEntry
	mu      sync.RWMutex
}

type TranspositionEntry struct {
	depth     int
	value     float64
	moveType  int // Exact, LowerBound, or UpperBound
	bestMove  game.RPSMove
	timestamp time.Time
}

func NewTranspositionTable() *TranspositionTable {
	return &TranspositionTable{
		entries: make(map[string]TranspositionEntry),
	}
}

func (tt *TranspositionTable) Store(state *game.RPSGame, depth int, value float64, moveType int, bestMove game.RPSMove) {
	tt.mu.Lock()
	defer tt.mu.Unlock()
	
	key := stateToKey(state)
	tt.entries[key] = TranspositionEntry{
		depth:     depth,
		value:     value,
		moveType:  moveType,
		bestMove:  bestMove,
		timestamp: time.Now(),
	}
}

func (tt *TranspositionTable) Lookup(state *game.RPSGame) (TranspositionEntry, bool) {
	tt.mu.RLock()
	defer tt.mu.RUnlock()
	
	key := stateToKey(state)
	entry, found := tt.entries[key]
	return entry, found
}

// stateToKey generates a unique string key from a game state
func stateToKey(state *game.RPSGame) string {
	// Implement a string representation of the board state
	// This could be JSON, or a more compact representation
	return fmt.Sprintf("%v", state) // Simplified version
}
```

### 3. Iterative Deepening

Implement iterative deepening to get best results within a time constraint:

```go
// FindBestMoveIterative performs iterative deepening search
func (m *MinimaxEngine) FindBestMoveIterative(state *game.RPSGame, maxTime time.Duration) (game.RPSMove, float64) {
	m.NodesEvaluated = 0
	m.StartTime = time.Now()
	m.MaxTime = maxTime
	
	var bestMove game.RPSMove
	var bestValue float64
	
	// Start with depth 1 and increase gradually
	for depth := 1; depth <= m.MaxDepth; depth++ {
		// If we've used more than 80% of our time, don't start a new iteration
		if float64(time.Since(m.StartTime)) > float64(maxTime)*0.8 {
			break
		}
		
		move, value := m.FindBestMove(state)
		
		// Keep track of the best move found so far
		if time.Since(m.StartTime) <= maxTime {
			bestMove = move
			bestValue = value
		} else {
			// If we timed out during this iteration, the results might be unreliable
			break
		}
		
		// If we found a forced win/loss, no need to search deeper
		if value > 900 || value < -900 {
			break
		}
	}
	
	return bestMove, bestValue
}
```

## Testing the Implementation

Create a simple test program to validate your minimax implementation:

```go
package main

import (
	"fmt"
	"time"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/analysis"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func main() {
	// Create a new game state
	gameState := game.NewRPSGame(21, 5, 10)
	
	// Make a few moves to set up an interesting position
	// ... (make some moves here)
	
	// Create the minimax engine
	engine := analysis.NewMinimaxEngine(5, analysis.StandardEvaluator)
	
	// Find the best move
	fmt.Println("Analyzing position...")
	startTime := time.Now()
	bestMove, value := engine.FindBestMoveIterative(gameState, 10*time.Second)
	elapsed := time.Since(startTime)
	
	// Print results
	fmt.Printf("Best move: %v\n", bestMove)
	fmt.Printf("Position value: %.2f\n", value)
	fmt.Printf("Nodes evaluated: %d\n", engine.NodesEvaluated)
	fmt.Printf("Time taken: %s\n", elapsed)
	fmt.Printf("Nodes per second: %.2f\n", float64(engine.NodesEvaluated)/elapsed.Seconds())
}
```

## Benchmarking and Analysis

To properly benchmark your minimax implementation:

1. Create several representative game positions
2. Test search performance at different depths (1-8)
3. Measure nodes evaluated per second
4. Analyze the impact of different optimizations

Example benchmark function:

```go
func benchmarkSearch(state *game.RPSGame, maxDepth int) {
	engine := analysis.NewMinimaxEngine(maxDepth, analysis.StandardEvaluator)
	
	for depth := 1; depth <= maxDepth; depth++ {
		engine.MaxDepth = depth
		
		startTime := time.Now()
		move, value := engine.FindBestMove(state)
		elapsed := time.Since(startTime)
		
		fmt.Printf("Depth %d: Value=%.2f, Move=%v, Nodes=%d, Time=%.3fs, NPS=%.2f\n",
			depth, value, move, engine.NodesEvaluated,
			elapsed.Seconds(), float64(engine.NodesEvaluated)/elapsed.Seconds())
	}
}
```

## Practical Considerations

### 1. Memory Management

The RPS card game has a moderate branching factor (typically 5-15 moves per position). At higher search depths, be mindful of:

- Memory consumption from recursive calls
- Size of the transposition table
- Copying game states efficiently

### 2. Evaluation Function Tuning

The evaluation function significantly impacts search quality:

- Start with simple material-based evaluation and gradually add complexity
- Tune weights empirically based on performance against known good positions
- Consider using machine learning to optimize evaluation weights

### 3. Time Management

For practical use, time management is crucial:

- Use iterative deepening with a fixed time budget
- Allocate more time for complex positions
- Save time in the endgame when fewer moves are available

## Conclusion

Implementing minimax with alpha-beta pruning for the RPS card game provides:

1. A baseline for evaluating neural network performance
2. Insights into game complexity and strategic depth
3. Better understanding of the skill vs. luck balance
4. A tool for generating optimal training examples

The next steps after completing this implementation would be to:

1. Create a suite of benchmark positions
2. Analyze neural network decisions against minimax recommendations
3. Measure the theoretical maximum win rate against random play
4. Possibly tune game rules based on your findings

Happy implementing! 