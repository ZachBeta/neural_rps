package neural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// RPSPolicyNetwork represents a neural network that predicts move probabilities for RPS
type RPSPolicyNetwork struct {
	// Simple 2-layer neural network
	inputSize  int
	hiddenSize int
	outputSize int

	// Weights and biases
	weightsInputHidden  [][]float64
	biasesHidden        []float64
	weightsHiddenOutput [][]float64
	biasesOutput        []float64

	// Debug information
	DebugEpochCount []int
}

// NewRPSPolicyNetwork creates a new policy network for RPS
func NewRPSPolicyNetwork(hiddenSize int) *RPSPolicyNetwork {
	// For RPS, the input size is 81 (9 positions * 9 features per position)
	inputSize := 81
	// The output is 9 positions (we'll select which card to play separately)
	outputSize := 9

	network := &RPSPolicyNetwork{
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

// Predict returns the position probabilities for a given game state
func (n *RPSPolicyNetwork) Predict(gameState *game.RPSGame) []float64 {
	// Convert game state to input features
	input := gameState.GetBoardAsFeatures()

	// Forward pass through the network
	return n.forward(input)
}

// PredictMove returns the best move according to the policy network
func (n *RPSPolicyNetwork) PredictMove(gameState *game.RPSGame) game.RPSMove {
	// Get valid moves
	validMoves := gameState.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{} // No valid moves
	}

	// Get position probabilities
	positionProbs := n.Predict(gameState)

	// Group moves by position
	movesByPosition := make(map[int][]game.RPSMove)
	for _, move := range validMoves {
		movesByPosition[move.Position] = append(movesByPosition[move.Position], move)
	}

	// Find the best position according to the policy network
	bestPosition := 0
	bestProb := positionProbs[0]
	for pos, prob := range positionProbs {
		if prob > bestProb && len(movesByPosition[pos]) > 0 {
			bestProb = prob
			bestPosition = pos
		}
	}

	// Choose the first move that places a card at the best position
	// In a more advanced implementation, we'd have another network to decide which card to play
	possibleMoves := movesByPosition[bestPosition]
	if len(possibleMoves) > 0 {
		return possibleMoves[0]
	}

	// Fallback: return the first valid move
	return validMoves[0]
}

// forward performs a forward pass through the network
func (n *RPSPolicyNetwork) forward(input []float64) []float64 {
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
func (n *RPSPolicyNetwork) Train(inputFeatures [][]float64, targetProbs [][]float64, learningRate float64) float64 {
	batchSize := len(inputFeatures)
	if batchSize == 0 {
		return 0
	}

	totalLoss := 0.0

	// Debug flag to track potential instability in epochs 4-6
	debug := false
	if len(n.DebugEpochCount) > 0 && n.DebugEpochCount[0] >= 4 && n.DebugEpochCount[0] <= 6 {
		debug = true
	}

	// Gradient clipping threshold
	const gradientThreshold = 1.0

	for b := 0; b < batchSize; b++ {
		input := inputFeatures[b]
		target := targetProbs[b]

		// Forward pass
		hidden := make([]float64, n.hiddenSize)
		for i := 0; i < n.hiddenSize; i++ {
			sum := n.biasesHidden[i]
			for j := 0; j < n.inputSize; j++ {
				sum += n.weightsInputHidden[i][j] * input[j]
			}
			hidden[i] = relu(sum)
		}

		// Output before softmax
		logits := make([]float64, n.outputSize)
		for i := 0; i < n.outputSize; i++ {
			sum := n.biasesOutput[i]
			for j := 0; j < n.hiddenSize; j++ {
				sum += n.weightsHiddenOutput[i][j] * hidden[j]
			}
			logits[i] = sum

			// Debug output for very large logits that might lead to softmax instability
			if debug && (math.Abs(sum) > 100) {
				fmt.Printf("WARNING: Large logit detected: %.4f at output %d\n", sum, i)
			}
		}

		// Apply softmax
		probs := softmax(logits)

		// Check for NaN in probabilities which indicates unstable training
		for i, p := range probs {
			if CheckForNaN(p) {
				fmt.Printf("ERROR: NaN detected in probability at epoch %d. Logit: %.4f\n",
					n.DebugEpochCount[0], logits[i])
				// Return a high loss but avoid crashing
				return 100.0
			}
		}

		// Calculate cross-entropy loss
		batchLoss := 0.0
		for i := 0; i < n.outputSize; i++ {
			if target[i] > 0 {
				// Ensure probability isn't too small to avoid numerical issues
				p := math.Max(probs[i], 1e-15)
				batchLoss -= target[i] * math.Log(p)
			}
		}

		// Debug output for unusually high loss values
		if debug && batchLoss > 10.0 {
			fmt.Printf("WARNING: High batch loss detected: %.4f\n", batchLoss)
		}

		totalLoss += batchLoss

		// Backward pass: calculate gradients
		// Output layer gradients
		outputGradients := make([]float64, n.outputSize)
		for i := 0; i < n.outputSize; i++ {
			outputGradients[i] = probs[i] - target[i]
			// Apply gradient clipping to prevent explosion
			outputGradients[i] = clipGradient(outputGradients[i], gradientThreshold)
		}

		// Update hidden->output weights and biases
		for i := 0; i < n.outputSize; i++ {
			for j := 0; j < n.hiddenSize; j++ {
				update := learningRate * outputGradients[i] * hidden[j]
				// Apply additional safety: clip the weight update
				update = clipGradient(update, 0.1)
				n.weightsHiddenOutput[i][j] -= update
			}
			n.biasesOutput[i] -= learningRate * outputGradients[i]
		}

		// Hidden layer gradients
		hiddenGradients := make([]float64, n.hiddenSize)
		for i := 0; i < n.hiddenSize; i++ {
			for j := 0; j < n.outputSize; j++ {
				hiddenGradients[i] += outputGradients[j] * n.weightsHiddenOutput[j][i]
			}
			// Apply ReLU gradient
			if hidden[i] <= 0 {
				hiddenGradients[i] = 0
			}
			// Apply gradient clipping
			hiddenGradients[i] = clipGradient(hiddenGradients[i], gradientThreshold)
		}

		// Update input->hidden weights and biases
		for i := 0; i < n.hiddenSize; i++ {
			for j := 0; j < n.inputSize; j++ {
				update := learningRate * hiddenGradients[i] * input[j]
				// Apply additional safety: clip the weight update
				update = clipGradient(update, 0.1)
				n.weightsInputHidden[i][j] -= update
			}
			n.biasesHidden[i] -= learningRate * hiddenGradients[i]
		}
	}

	return totalLoss / float64(batchSize)
}

// SaveToFile saves the network weights and biases to a file
func (n *RPSPolicyNetwork) SaveToFile(filename string) error {
	// Create a serializable representation of the network
	data := map[string]interface{}{
		"inputSize":           n.inputSize,
		"hiddenSize":          n.hiddenSize,
		"outputSize":          n.outputSize,
		"weightsInputHidden":  n.weightsInputHidden,
		"biasesHidden":        n.biasesHidden,
		"weightsHiddenOutput": n.weightsHiddenOutput,
		"biasesOutput":        n.biasesOutput,
	}

	// Marshal and save to file using the helper function
	return saveToJSON(filename, data)
}

// LoadFromFile loads the network weights and biases from a file
func (n *RPSPolicyNetwork) LoadFromFile(filename string) error {
	// Load data from file
	var data map[string]interface{}
	err := loadFromJSON(filename, &data)
	if err != nil {
		return err
	}

	// Extract structure and size information
	inputSize, ok1 := data["inputSize"].(float64)
	hiddenSize, ok2 := data["hiddenSize"].(float64)
	outputSize, ok3 := data["outputSize"].(float64)

	if !ok1 || !ok2 || !ok3 {
		return errors.New("invalid network structure in file")
	}

	// Check compatibility
	if int(inputSize) != n.inputSize || int(outputSize) != n.outputSize {
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
		for i := 0; i < n.outputSize; i++ {
			n.weightsHiddenOutput[i] = make([]float64, n.hiddenSize)
		}
	}

	// Load weights and biases
	loadWeightsMatrix(data["weightsInputHidden"], &n.weightsInputHidden)
	loadWeightsVector(data["biasesHidden"], &n.biasesHidden)
	loadWeightsMatrix(data["weightsHiddenOutput"], &n.weightsHiddenOutput)
	loadWeightsVector(data["biasesOutput"], &n.biasesOutput)

	return nil
}

// GetHiddenSize returns the hidden layer size
func (n *RPSPolicyNetwork) GetHiddenSize() int {
	return n.hiddenSize
}

// GetWeights returns flattened network weights (input->hidden, hidden->output)
func (n *RPSPolicyNetwork) GetWeights() []float64 {
	total := n.hiddenSize*n.inputSize + n.outputSize*n.hiddenSize
	weights := make([]float64, total)
	idx := 0
	// input->hidden weights
	for i := 0; i < n.hiddenSize; i++ {
		for j := 0; j < n.inputSize; j++ {
			weights[idx] = n.weightsInputHidden[i][j]
			idx++
		}
	}
	// hidden->output weights
	for i := 0; i < n.outputSize; i++ {
		for j := 0; j < n.hiddenSize; j++ {
			weights[idx] = n.weightsHiddenOutput[i][j]
			idx++
		}
	}
	return weights
}

// SetWeights assigns flattened weight values into the policy network
func (n *RPSPolicyNetwork) SetWeights(weights []float64) error {
	expected := n.hiddenSize*n.inputSize + n.outputSize*n.hiddenSize
	if len(weights) != expected {
		return fmt.Errorf("policy weights length mismatch: expected %d, got %d", expected, len(weights))
	}
	idx := 0
	// input->hidden
	for i := 0; i < n.hiddenSize; i++ {
		for j := 0; j < n.inputSize; j++ {
			n.weightsInputHidden[i][j] = weights[idx]
			idx++
		}
	}
	// hidden->output
	for i := 0; i < n.outputSize; i++ {
		for j := 0; j < n.hiddenSize; j++ {
			n.weightsHiddenOutput[i][j] = weights[idx]
			idx++
		}
	}
	return nil
}
