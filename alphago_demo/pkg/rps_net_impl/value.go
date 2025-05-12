package neural

import (
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// AGValueNetwork represents a neural network that predicts the game outcome value
type AGValueNetwork struct {
	// Simple 2-layer neural network
	inputSize  int
	hiddenSize int

	// Weights and biases
	weightsInputHidden  [][]float64
	biasesHidden        []float64
	weightsHiddenOutput []float64
	biasOutput          float64
}

// NewAGValueNetwork creates a new value network
func NewAGValueNetwork(inputSize, hiddenSize int) *AGValueNetwork {
	network := &AGValueNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,

		weightsInputHidden:  make([][]float64, hiddenSize),
		biasesHidden:        make([]float64, hiddenSize),
		weightsHiddenOutput: make([]float64, hiddenSize),
		biasOutput:          0,
	}

	// Initialize weights with Xavier initialization
	xavierInput := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	xavierHidden := math.Sqrt(2.0 / float64(hiddenSize+1))

	// Initialize input->hidden weights and biases
	for i := 0; i < hiddenSize; i++ {
		network.weightsInputHidden[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			network.weightsInputHidden[i][j] = (rand.Float64()*2 - 1) * xavierInput
		}
		network.biasesHidden[i] = 0
		network.weightsHiddenOutput[i] = (rand.Float64()*2 - 1) * xavierHidden
	}

	return network
}

// Predict returns the estimated value (win probability) for a given game state
// Returns a value between 0 and 1 where:
// - 1 means current player will win
// - 0 means current player will lose
// - 0.5 means draw or uncertain
func (n *AGValueNetwork) Predict(gameState *game.AGGame) float64 {
	// Convert game state to input features
	input := gameState.GetBoardAsFeatures()

	// If the game is already over, return the actual outcome
	if gameState.IsGameOver() {
		winner := gameState.GetWinner()
		if winner == game.Empty {
			return 0.5 // Draw
		} else if winner == gameState.CurrentPlayer {
			return 1.0 // Current player won
		} else {
			return 0.0 // Current player lost
		}
	}

	// Forward pass through the network
	return n.forward(input)
}

// forward performs a forward pass through the network
func (n *AGValueNetwork) forward(input []float64) float64 {
	// Hidden layer activation
	hidden := make([]float64, n.hiddenSize)
	for i := 0; i < n.hiddenSize; i++ {
		sum := n.biasesHidden[i]
		for j := 0; j < n.inputSize; j++ {
			sum += n.weightsInputHidden[i][j] * input[j]
		}
		hidden[i] = relu(sum)
	}

	// Output layer - single value
	sum := n.biasOutput
	for i := 0; i < n.hiddenSize; i++ {
		sum += n.weightsHiddenOutput[i] * hidden[i]
	}

	// Apply sigmoid to get value between 0 and 1
	return sigmoid(sum)
}

// Train updates the network weights based on a batch of input features and target values
// Returns the average loss across the batch
func (n *AGValueNetwork) Train(inputFeatures [][]float64, targetValues []float64, learningRate float64) float64 {
	// Implement backpropagation training (simplified version for the demo)
	totalLoss := 0.0

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

		// Output layer - single value
		outputSum := n.biasOutput
		for j := 0; j < n.hiddenSize; j++ {
			outputSum += n.weightsHiddenOutput[j] * hidden[j]
		}
		output := sigmoid(outputSum)

		// Calculate mean squared error loss
		error := output - targetValues[i]
		batchLoss := error * error
		totalLoss += batchLoss

		// Compute output layer gradient (simplified mean squared error gradient)
		// For binary cross-entropy, the gradient is (output - target)
		outputGradient := output - targetValues[i]

		// Compute sigmoid gradient
		sigmoidGradient := output * (1 - output)

		// Combine gradients
		outputGradient *= sigmoidGradient

		// Update output layer weights and bias
		for j := 0; j < n.hiddenSize; j++ {
			n.weightsHiddenOutput[j] -= learningRate * outputGradient * hidden[j]
		}
		n.biasOutput -= learningRate * outputGradient

		// Compute hidden layer gradients
		hiddenGradients := make([]float64, n.hiddenSize)
		for j := 0; j < n.hiddenSize; j++ {
			hiddenGradients[j] = outputGradient * n.weightsHiddenOutput[j]

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
