package agents

import (
	"fmt"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/analysis"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// MinimaxAgent implements an agent that uses minimax search to make decisions
type MinimaxAgent struct {
	name               string
	searchDepth        int
	timeLimit          time.Duration
	useCache           bool
	minimaxEngine      *analysis.MinimaxEngine
	positionsEvaluated int
	totalMoveTime      time.Duration
	moveCount          int
	verbose            bool
}

// NewMinimaxAgent creates a new minimax-based agent
func NewMinimaxAgent(name string, depth int, timeLimit time.Duration, useCache bool) *MinimaxAgent {
	// Create minimax engine with StandardEvaluator
	engine := analysis.NewMinimaxEngine(depth, analysis.StandardEvaluator)

	// Enable transposition table if requested
	if useCache {
		engine.EnableTranspositionTable()
	}

	// Set time limit for moves (with a default if not specified)
	if timeLimit == 0 {
		timeLimit = 3 * time.Second
	}

	return &MinimaxAgent{
		name:               name,
		searchDepth:        depth,
		timeLimit:          timeLimit,
		useCache:           useCache,
		minimaxEngine:      engine,
		positionsEvaluated: 0,
		totalMoveTime:      0,
		moveCount:          0,
		verbose:            false,
	}
}

// Name returns the agent's name
func (a *MinimaxAgent) Name() string {
	return a.name
}

// SetVerbose enables or disables verbose logging
func (a *MinimaxAgent) SetVerbose(verbose bool) {
	a.verbose = verbose
}

// GetMove returns the best move according to minimax search
func (a *MinimaxAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	startTime := time.Now()

	// Use iterative deepening with time limit
	move, value := a.minimaxEngine.FindBestMoveIterative(state.Copy(), a.timeLimit)

	// Update stats
	moveTime := time.Since(startTime)
	a.totalMoveTime += moveTime
	a.moveCount++
	a.positionsEvaluated += a.minimaxEngine.NodesEvaluated

	// Log the move for analysis only if verbose mode is enabled
	if a.verbose {
		if a.useCache {
			hits, misses, hitRate := a.minimaxEngine.GetCacheStats()
			fmt.Printf("Minimax move: %v, value: %.2f, time: %v, positions: %d, cache: %d hits, %d misses (%.1f%%)\n",
				move, value, moveTime, a.minimaxEngine.NodesEvaluated, hits, misses, hitRate)
		} else {
			fmt.Printf("Minimax move: %v, value: %.2f, time: %v, positions: %d\n",
				move, value, moveTime, a.minimaxEngine.NodesEvaluated)
		}
	}

	// Set player for the move
	move.Player = state.CurrentPlayer

	return move, nil
}

// GetStats returns statistics about the agent's performance
func (a *MinimaxAgent) GetStats() (avgTime time.Duration, totalPositions int, avgPositionsPerMove float64) {
	if a.moveCount == 0 {
		return 0, 0, 0
	}

	avgTime = a.totalMoveTime / time.Duration(a.moveCount)
	totalPositions = a.positionsEvaluated
	avgPositionsPerMove = float64(a.positionsEvaluated) / float64(a.moveCount)

	return
}

// ResetStats resets the agent's performance statistics
func (a *MinimaxAgent) ResetStats() {
	a.positionsEvaluated = 0
	a.totalMoveTime = 0
	a.moveCount = 0

	// Also reset cache stats if using cache
	if a.useCache {
		a.minimaxEngine.DisableTranspositionTable()
		a.minimaxEngine.EnableTranspositionTable()
	}
}

// Implements any Agent interface required by the game system
type Agent interface {
	Name() string
	GetMove(state *game.RPSGame) (game.RPSMove, error)
}

var _ Agent = (*MinimaxAgent)(nil) // Verify MinimaxAgent implements Agent
