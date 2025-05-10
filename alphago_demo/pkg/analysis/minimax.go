package analysis

import (
	"math"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// MinimaxEngine implements an alpha-beta search for RPS card game
type MinimaxEngine struct {
	MaxDepth           int
	NodesEvaluated     int
	MaxTime            time.Duration
	StartTime          time.Time
	EvaluationFn       func(*game.RPSGame) float64
	TranspositionTable *SimpleTranspositionTable // Added transposition table
}

// NewMinimaxEngine creates a new minimax search engine
func NewMinimaxEngine(maxDepth int, evalFn func(*game.RPSGame) float64) *MinimaxEngine {
	return &MinimaxEngine{
		MaxDepth:     maxDepth,
		EvaluationFn: evalFn,
		MaxTime:      30 * time.Second, // Default 30-second time limit
	}
}

// EnableTranspositionTable enables caching of positions
func (m *MinimaxEngine) EnableTranspositionTable() {
	m.TranspositionTable = NewSimpleTranspositionTable()
}

// DisableTranspositionTable turns off position caching
func (m *MinimaxEngine) DisableTranspositionTable() {
	m.TranspositionTable = nil
}

// GetCacheStats returns statistics about the transposition table if enabled
func (m *MinimaxEngine) GetCacheStats() (hits int, misses int, hitRate float64) {
	if m.TranspositionTable == nil {
		return 0, 0, 0.0
	}
	return m.TranspositionTable.GetStats()
}

// FindBestMove returns the best move for the current player
func (m *MinimaxEngine) FindBestMove(state *game.RPSGame) (game.RPSMove, float64) {
	// If we have a transposition table, check it first
	if m.TranspositionTable != nil {
		if result, found := m.TranspositionTable.Get(state); found {
			// Only use cached result if it was searched at sufficient depth
			if result.Depth >= m.MaxDepth {
				return result.BestMove, result.Value
			}
		}
	}

	m.NodesEvaluated = 0
	m.StartTime = time.Now()

	// Initialize alpha-beta bounds
	alpha := math.Inf(-1)
	beta := math.Inf(1)

	// Determine if current player is maximizing
	maximizingPlayer := state.CurrentPlayer == game.Player1

	// Call minimax search
	value, move := m.minimax(state, m.MaxDepth, alpha, beta, maximizingPlayer)

	// Cache the result if transposition table is enabled
	if m.TranspositionTable != nil {
		m.TranspositionTable.Put(state, PositionResult{
			BestMove:      move,
			Value:         value,
			Depth:         m.MaxDepth,
			NodesExplored: m.NodesEvaluated,
		})
	}

	return move, value
}

// minimax performs alpha-beta pruned minimax search
func (m *MinimaxEngine) minimax(state *game.RPSGame, depth int, alpha, beta float64, maximizingPlayer bool) (float64, game.RPSMove) {
	// Check transposition table for this position at current depth
	if m.TranspositionTable != nil && depth > 0 {
		if result, found := m.TranspositionTable.Get(state); found {
			if result.Depth >= depth {
				return result.Value, result.BestMove
			}
		}
	}

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

		// Store result in transposition table if enabled
		if m.TranspositionTable != nil && depth > 0 {
			m.TranspositionTable.Put(state, PositionResult{
				BestMove:      bestMove,
				Value:         maxEval,
				Depth:         depth,
				NodesExplored: 0, // Not tracked per subtree
			})
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

		// Store result in transposition table if enabled
		if m.TranspositionTable != nil && depth > 0 {
			m.TranspositionTable.Put(state, PositionResult{
				BestMove:      bestMove,
				Value:         minEval,
				Depth:         depth,
				NodesExplored: 0, // Not tracked per subtree
			})
		}

		return minEval, bestMove
	}
}

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
