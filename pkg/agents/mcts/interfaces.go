package mcts

import (
	"context"

	"github.com/zachbeta/neural_rps/pkg/game"
)

// GameState defines the interface for game states used by MCTS
type GameState interface {
	// Core game state methods
	GetLegalMoves() []game.RPSCardMove
	ApplyMove(move game.RPSCardMove)
	IsGameOver() bool
	GetWinner() game.Player
	GetCurrentPlayer() game.Player
	Clone() GameState

	// Methods needed for neural network integration
	ToTensor() []float32
	GetLastMove() game.RPSCardMove
}

// MCTSNode represents a node in the MCTS tree
type MCTSNode struct {
	Parent     *MCTSNode
	Children   []*MCTSNode
	State      GameState
	Move       game.RPSCardMove
	Visits     int
	TotalValue float64
	Prior      float32
	UCBScore   float64
}

// MCTSParams contains configuration for MCTS algorithm
type MCTSParams struct {
	NumSimulations   int     // Number of simulations per move
	ExplorationConst float64 // Exploration constant for UCB
	Seed             int64   // Random seed
	Temperature      float64 // Temperature for move selection
}

// DefaultMCTSParams returns reasonable default parameters
func DefaultMCTSParams() MCTSParams {
	return MCTSParams{
		NumSimulations:   1000,
		ExplorationConst: 1.5,
		Seed:             42,
		Temperature:      1.0,
	}
}

// MCTSAgent defines the interface for MCTS-based agents
type MCTSAgent interface {
	SetRootState(state GameState)
	Search(ctx context.Context) game.RPSCardMove
	SetBatchSize(size int)
	GetStats() map[string]interface{}
	Close()
}
