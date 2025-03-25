package neural

import (
	"errors"
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// RPSValueNetwork represents a neural network that predicts the value of a position
type RPSValueNetwork struct {
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

// NewRPSValueNetwork creates a new value network for RPS
func NewRPSValueNetwork(hiddenSize int) *RPSValueNetwork {
	// For RPS, the input size is 81 (9 positions * 9 features per position)
	inputSize := 81
	outputSize := 1

	network := &RPSValueNetwork{
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

// Predict returns the value (win probability) for a given game state
func (n *RPSValueNetwork) Predict(gameState *game.RPSGame) float64 {
	// Convert game state to input features
	input := gameState.GetBoardAsFeatures()

	// Forward pass through the network
	return n.forward(input)
}

// forward performs a forward pass through the network
func (n *RPSValueNetwork) forward(input []float64) float64 {
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
	output := n.biasesOutput[0]
	for i := 0; i < n.hiddenSize; i++ {
		output += n.weightsHiddenOutput[0][i] * hidden[i]
	}

	// Apply sigmoid to get a value between 0 and 1
	return sigmoid(output)
}

// Train updates the network weights based on a batch of input features and target values
// Returns the average loss across the batch
func (n *RPSValueNetwork) Train(inputFeatures [][]float64, targetValues []float64, learningRate float64) float64 {
	batchSize := len(inputFeatures)
	if batchSize == 0 {
		return 0
	}

	totalLoss := 0.0

	for b := 0; b < batchSize; b++ {
		input := inputFeatures[b]
		target := targetValues[b]

		// Forward pass
		hidden := make([]float64, n.hiddenSize)
		for i := 0; i < n.hiddenSize; i++ {
			sum := n.biasesHidden[i]
			for j := 0; j < n.inputSize; j++ {
				sum += n.weightsInputHidden[i][j] * input[j]
			}
			hidden[i] = relu(sum)
		}

		// Output before sigmoid
		logit := n.biasesOutput[0]
		for i := 0; i < n.hiddenSize; i++ {
			logit += n.weightsHiddenOutput[0][i] * hidden[i]
		}

		// Apply sigmoid
		prediction := sigmoid(logit)

		// Calculate mean squared error loss
		loss := (prediction - target) * (prediction - target)
		totalLoss += loss

		// Backward pass: calculate gradients
		// Output layer gradient
		outputGradient := 2 * (prediction - target) * prediction * (1 - prediction)

		// Update hidden->output weights and bias
		for i := 0; i < n.hiddenSize; i++ {
			n.weightsHiddenOutput[0][i] -= learningRate * outputGradient * hidden[i]
		}
		n.biasesOutput[0] -= learningRate * outputGradient

		// Hidden layer gradients
		hiddenGradients := make([]float64, n.hiddenSize)
		for i := 0; i < n.hiddenSize; i++ {
			hiddenGradients[i] = outputGradient * n.weightsHiddenOutput[0][i]
			// Apply ReLU gradient
			if hidden[i] <= 0 {
				hiddenGradients[i] = 0
			}
		}

		// Update input->hidden weights and biases
		for i := 0; i < n.hiddenSize; i++ {
			for j := 0; j < n.inputSize; j++ {
				n.weightsInputHidden[i][j] -= learningRate * hiddenGradients[i] * input[j]
			}
			n.biasesHidden[i] -= learningRate * hiddenGradients[i]
		}
	}

	return totalLoss / float64(batchSize)
}

// SaveToFile saves the network weights and biases to a file
func (n *RPSValueNetwork) SaveToFile(filename string) error {
	// Create a serializable representation of the network
	data := map[string]interface{}{
		"inputSize":           n.inputSize,
		"hiddenSize":          n.hiddenSize,
		"weightsInputHidden":  n.weightsInputHidden,
		"biasesHidden":        n.biasesHidden,
		"weightsHiddenOutput": n.weightsHiddenOutput,
		"biasOutput":          n.biasesOutput[0],
	}

	// Marshal and save to file using the helper function
	return saveToJSON(filename, data)
}

// LoadFromFile loads the network weights and biases from a file
func (n *RPSValueNetwork) LoadFromFile(filename string) error {
	// Load data from file
	var data map[string]interface{}
	err := loadFromJSON(filename, &data)
	if err != nil {
		return err
	}

	// Extract structure and size information
	inputSize, ok1 := data["inputSize"].(float64)
	hiddenSize, ok2 := data["hiddenSize"].(float64)

	if !ok1 || !ok2 {
		return errors.New("invalid network structure in file")
	}

	// Check compatibility
	if int(inputSize) != n.inputSize {
		return errors.New("incompatible network structure")
	}

	// Resize network if hidden size differs
	if int(hiddenSize) != n.hiddenSize {
		n.hiddenSize = int(hiddenSize)
		n.weightsInputHidden = make([][]float64, n.hiddenSize)
		n.biasesHidden = make([]float64, n.hiddenSize)
		for i := 0; i < n.hiddenSize; i++ {
			n.weightsInputHidden[i] = make([]float64, n.inputSize)
		}
		n.weightsHiddenOutput = make([][]float64, n.outputSize)
		for i := 0; i < n.outputSize; i++ {
			n.weightsHiddenOutput[i] = make([]float64, n.hiddenSize)
		}
		n.biasesOutput = make([]float64, n.outputSize)
	}

	// Load weights and biases
	loadWeightsMatrix(data["weightsInputHidden"], &n.weightsInputHidden)
	loadWeightsVector(data["biasesHidden"], &n.biasesHidden)
	loadWeightsMatrix(data["weightsHiddenOutput"], &n.weightsHiddenOutput)

	// Load bias output (which is a single value)
	if biasOutput, ok := data["biasOutput"].(float64); ok {
		n.biasesOutput[0] = biasOutput
	}

	return nil
}
