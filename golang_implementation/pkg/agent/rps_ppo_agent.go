package agent

import (
	"fmt"
	"math/rand"

	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/neural"
)

// RPSPPOAgent wraps a PPO agent to work with the RPS card game
type RPSPPOAgent struct {
	name       string
	network    *neural.Network
	stateSize  int
	actionSize int // 9 possible positions
}

// NewRPSPPOAgent creates a new RPS PPO agent
func NewRPSPPOAgent(name string, hiddenSize int) *RPSPPOAgent {
	return &RPSPPOAgent{
		name:       name,
		network:    neural.NewNetwork(81, hiddenSize, 9), // 81 input features, 9 output positions
		stateSize:  81,
		actionSize: 9,
	}
}

// Name returns the agent's name
func (a *RPSPPOAgent) Name() string {
	return a.name
}

// GetMove returns the best move according to the agent's policy
func (a *RPSPPOAgent) GetMove(gameState *game.RPSCardGame) (game.RPSCardMove, error) {
	// Convert game state to features
	features := gameState.GetBoardAsFeatures()

	// Get action probabilities
	probs := a.network.Forward(features)

	// Get valid moves
	validMoves := gameState.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSCardMove{}, fmt.Errorf("no valid moves")
	}

	// Create a mapping from board positions to moves
	// This is needed because PPO outputs position probabilities, but we need to select a card too
	posMoves := make(map[int][]game.RPSCardMove)
	for _, move := range validMoves {
		posMoves[move.Position] = append(posMoves[move.Position], move)
	}

	// Mask invalid positions
	validPositions := make([]int, 0)
	for pos := 0; pos < 9; pos++ {
		if moves, ok := posMoves[pos]; ok && len(moves) > 0 {
			validPositions = append(validPositions, pos)
			// Keep probability as is
		} else {
			// Set probability to 0 for invalid positions
			probs[pos] = 0
		}
	}

	// Renormalize probabilities
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	} else {
		// If all probabilities are 0, use uniform distribution for valid positions
		for _, pos := range validPositions {
			probs[pos] = 1.0 / float64(len(validPositions))
		}
	}

	// Sample position based on probability
	pos := sampleFromDistribution(probs)

	// If we have multiple cards that can be played at this position, choose randomly
	moves := posMoves[pos]
	move := moves[rand.Intn(len(moves))]

	return move, nil
}

// sampleFromDistribution samples an index from a probability distribution
func sampleFromDistribution(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0

	for i, prob := range probs {
		cumulativeProb += prob
		if r <= cumulativeProb {
			return i
		}
	}

	// Fallback to first index (shouldn't happen with properly normalized probs)
	return 0
}

// Train trains the agent on a batch of games
func (a *RPSPPOAgent) Train(games []*game.RPSCardGame, learningRate float64) {
	if len(games) == 0 {
		return
	}

	// Prepare training data
	inputs := make([][]float64, 0)
	targets := make([][]float64, 0)

	for _, g := range games {
		// Get board features
		input := g.GetBoardAsFeatures()
		inputs = append(inputs, input)

		// Get valid moves
		validMoves := g.GetValidMoves()

		// Create target distribution (uniform over valid positions)
		target := make([]float64, 9)
		validPositions := make(map[int]bool)

		for _, move := range validMoves {
			validPositions[move.Position] = true
		}

		// Set uniform probability for valid positions
		validCount := len(validPositions)
		if validCount > 0 {
			prob := 1.0 / float64(validCount)
			for pos := range validPositions {
				target[pos] = prob
			}
		}

		targets = append(targets, target)
	}

	// Train network
	options := neural.TrainingOptions{
		LearningRate: learningRate,
		Epochs:       5,
		BatchSize:    32,
		Parallel:     true,
	}

	a.network.Train(inputs, targets, options)
}

// SaveWeights saves the network weights to a file
func (a *RPSPPOAgent) SaveWeights(filename string) error {
	return a.network.SaveWeights(filename)
}

// LoadWeights loads the network weights from a file
func (a *RPSPPOAgent) LoadWeights(filename string) error {
	return a.network.LoadWeights(filename)
}
