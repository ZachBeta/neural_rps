package neural

import (
	"fmt"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// Agent defines the interface for game-playing agents
type Agent interface {
	Name() string
	GetMove(state *game.RPSGame) (game.RPSMove, error)
}

// NeuralAgent wraps a policy network for gameplay
type NeuralAgent struct {
	name          string
	policyNetwork *RPSPolicyNetwork
	movesCount    int
}

// NewNeuralAgent creates a new neural network based agent
func NewNeuralAgent(name string, policyNetwork *RPSPolicyNetwork) *NeuralAgent {
	return &NeuralAgent{
		name:          name,
		policyNetwork: policyNetwork,
		movesCount:    0,
	}
}

// Name returns the agent's name
func (a *NeuralAgent) Name() string {
	return a.name
}

// GetMove returns a move based on neural network prediction
func (a *NeuralAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	// Get valid moves
	validMoves := state.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}

	// Get policy predictions
	predictions := a.policyNetwork.Predict(state)

	// Find best valid move
	bestScore := -1.0
	var bestMove game.RPSMove

	for _, move := range validMoves {
		// Get prediction score for this position
		moveIndex := move.Position

		if predictions[moveIndex] > bestScore {
			bestScore = predictions[moveIndex]
			bestMove = move
		}
	}

	// Set player for the move
	bestMove.Player = state.CurrentPlayer
	a.movesCount++

	fmt.Printf("Neural move: pos %d, score %.3f\n", bestMove.Position, bestScore)

	return bestMove, nil
}

// GetStats returns the number of moves made by this agent
func (a *NeuralAgent) GetStats() int {
	return a.movesCount
}

// ResetStats resets the agent's statistics
func (a *NeuralAgent) ResetStats() {
	a.movesCount = 0
}
