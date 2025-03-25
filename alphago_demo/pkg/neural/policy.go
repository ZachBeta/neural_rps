package neural

import (
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// AGPolicyNetwork represents a neural network that predicts move probabilities
type AGPolicyNetwork struct {
	// Simple 2-layer neural network
	inputSize  int
	hiddenSize int
	outputSize int

	// Weights and biases
	weightsInputHidden  [][]float64
	biasesHidden        []float64
	weightsHiddenOutput [][]float64
	biasesOutput        []float64
}

// NewAGPolicyNetwork creates a new policy network
func NewAGPolicyNetwork(inputSize, hiddenSize int) *AGPolicyNetwork {
	// For Tic-Tac-Toe, the output size is 9 (3x3 board)
	outputSize := 9

	network := &AGPolicyNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,

		weightsInputHidden:  make([][]float64, hiddenSize),
		biasesHidden:        make([]float64, hiddenSize),
		weightsHiddenOutput: make([][]float64, outputSize),
		biasesOutput:        make([]float64, outputSize),
	}

	// Initialize weights with Xavier initialization
	xavierInput := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	xavierHidden := math.Sqrt(2.0 / float64(hiddenSize+outputSize))

	// Initialize input->hidden weights and biases
	for i := 0; i < hiddenSize; i++ {
		network.weightsInputHidden[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			network.weightsInputHidden[i][j] = (rand.Float64()*2 - 1) * xavierInput
		}
		network.biasesHidden[i] = 0
	}

	// Initialize hidden->output weights and biases
	for i := 0; i < outputSize; i++ {
		network.weightsHiddenOutput[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			network.weightsHiddenOutput[i][j] = (rand.Float64()*2 - 1) * xavierHidden
		}
		network.biasesOutput[i] = 0
	}

	return network
}

// Predict returns the move probabilities for a given game state
func (n *AGPolicyNetwork) Predict(gameState *game.AGGame) []float64 {
	// Convert game state to input features
	input := gameState.GetBoardAsFeatures()

	// Forward pass through the network
	return n.forward(input)
}

// PredictMove returns the best move according to the policy network
func (n *AGPolicyNetwork) PredictMove(gameState *game.AGGame) game.AGMove {
	// Get valid moves
	validMoves := gameState.GetValidMoves()
	if len(validMoves) == 0 {
		return game.AGMove{} // No valid moves
	}

	// Get probabilities
	probs := n.Predict(gameState)

	// Find valid move with highest probability
	bestMove := validMoves[0]
	bestProb := probs[bestMove.Row*3+bestMove.Col]

	for _, move := range validMoves[1:] {
		idx := move.Row*3 + move.Col
		if probs[idx] > bestProb {
			bestProb = probs[idx]
			bestMove = move
		}
	}

	return bestMove
}

// forward performs a forward pass through the network
func (n *AGPolicyNetwork) forward(input []float64) []float64 {
	// Hidden layer activation
	hidden := make([]float64, n.hiddenSize)
	for i := 0; i < n.hiddenSize; i++ {
		sum := n.biasesHidden[i]
		for j := 0; j < n.inputSize; j++ {
			sum += n.weightsInputHidden[i][j] * input[j]
		}
		hidden[i] = relu(sum)
	}

	// Output layer
	output := make([]float64, n.outputSize)
	for i := 0; i < n.outputSize; i++ {
		sum := n.biasesOutput[i]
		for j := 0; j < n.hiddenSize; j++ {
			sum += n.weightsHiddenOutput[i][j] * hidden[j]
		}
		output[i] = sum
	}

	// Apply softmax to get probabilities
	return softmax(output)
}

// Train updates the network weights based on a batch of input features and target probabilities
// Returns the average loss across the batch
func (n *AGPolicyNetwork) Train(inputFeatures [][]float64, targetProbs [][]float64, learningRate float64) float64 {
	totalLoss := 0.0

	// Implement backpropagation training (simplified version for the demo)
	// In a full implementation, this would use techniques like stochastic gradient descent
	// For this demo, we'll use a very simple update rule

	for i, input := range inputFeatures {
		// Forward pass
		hidden := make([]float64, n.hiddenSize)
		for j := 0; j < n.hiddenSize; j++ {
			sum := n.biasesHidden[j]
			for k := 0; k < n.inputSize; k++ {
				sum += n.weightsInputHidden[j][k] * input[k]
			}
			hidden[j] = relu(sum)
		}

		// Output layer
		output := make([]float64, n.outputSize)
		for j := 0; j < n.outputSize; j++ {
			sum := n.biasesOutput[j]
			for k := 0; k < n.hiddenSize; k++ {
				sum += n.weightsHiddenOutput[j][k] * hidden[k]
			}
			output[j] = sum
		}

		// Apply softmax
		probs := softmax(output)

		// Calculate cross-entropy loss
		batchLoss := 0.0
		for j := 0; j < n.outputSize; j++ {
			if targetProbs[i][j] > 0 {
				batchLoss -= targetProbs[i][j] * math.Log(probs[j])
			}
		}
		totalLoss += batchLoss

		// Compute output layer gradients (simplified cross-entropy gradient)
		outputGradients := make([]float64, n.outputSize)
		for j := 0; j < n.outputSize; j++ {
			outputGradients[j] = probs[j] - targetProbs[i][j]
		}

		// Update output layer weights and biases
		for j := 0; j < n.outputSize; j++ {
			for k := 0; k < n.hiddenSize; k++ {
				n.weightsHiddenOutput[j][k] -= learningRate * outputGradients[j] * hidden[k]
			}
			n.biasesOutput[j] -= learningRate * outputGradients[j]
		}

		// Compute hidden layer gradients
		hiddenGradients := make([]float64, n.hiddenSize)
		for j := 0; j < n.hiddenSize; j++ {
			for k := 0; k < n.outputSize; k++ {
				hiddenGradients[j] += outputGradients[k] * n.weightsHiddenOutput[k][j]
			}
			// Apply ReLU derivative
			if hidden[j] > 0 {
				hiddenGradients[j] *= 1
			} else {
				hiddenGradients[j] = 0
			}
		}

		// Update hidden layer weights and biases
		for j := 0; j < n.hiddenSize; j++ {
			for k := 0; k < n.inputSize; k++ {
				n.weightsInputHidden[j][k] -= learningRate * hiddenGradients[j] * input[k]
			}
			n.biasesHidden[j] -= learningRate * hiddenGradients[j]
		}
	}

	if len(inputFeatures) > 0 {
		return totalLoss / float64(len(inputFeatures))
	}
	return 0
}
