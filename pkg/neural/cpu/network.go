package cpu

import (
	"errors"
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/pkg/common"
)

// Network implements a CPU-based feed-forward neural network
type Network struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	Weights1   [][]float64
	Bias1      []float64
	Weights2   [][]float64
	Bias2      []float64
}

// NewNetwork creates a new neural network with the given layer sizes
func NewNetwork(inputSize, hiddenSize, outputSize int) *Network {
	nn := &Network{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
		Weights1:   make([][]float64, inputSize),
		Bias1:      make([]float64, hiddenSize),
		Weights2:   make([][]float64, hiddenSize),
		Bias2:      make([]float64, outputSize),
	}

	// Initialize with Xavier/Glorot initialization
	limit1 := math.Sqrt(6.0 / float64(inputSize+hiddenSize))
	for i := 0; i < inputSize; i++ {
		nn.Weights1[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			nn.Weights1[i][j] = (rand.Float64() * 2 * limit1) - limit1
		}
	}

	limit2 := math.Sqrt(6.0 / float64(hiddenSize+outputSize))
	for i := 0; i < hiddenSize; i++ {
		nn.Weights2[i] = make([]float64, outputSize)
		for j := 0; j < outputSize; j++ {
			nn.Weights2[i][j] = (rand.Float64() * 2 * limit2) - limit2
		}
	}

	return nn
}

// Forward runs a forward pass through the network
func (nn *Network) Forward(input []float64) ([]float64, error) {
	if len(input) != nn.InputSize {
		return nil, errors.New("input size mismatch")
	}

	// Hidden layer with ReLU activation
	hidden := make([]float64, nn.HiddenSize)
	for j := 0; j < nn.HiddenSize; j++ {
		sum := nn.Bias1[j]
		for i := 0; i < nn.InputSize; i++ {
			sum += input[i] * nn.Weights1[i][j]
		}
		// ReLU activation
		if sum > 0 {
			hidden[j] = sum
		} else {
			hidden[j] = 0
		}
	}

	// Output layer with softmax activation
	output := make([]float64, nn.OutputSize)
	var sum float64
	var maxVal float64 = -math.MaxFloat64

	// First find the maximum value for numerical stability
	for k := 0; k < nn.OutputSize; k++ {
		val := nn.Bias2[k]
		for j := 0; j < nn.HiddenSize; j++ {
			val += hidden[j] * nn.Weights2[j][k]
		}
		if val > maxVal {
			maxVal = val
		}
		output[k] = val
	}

	// Apply softmax with the numerical stability trick
	for k := 0; k < nn.OutputSize; k++ {
		output[k] = math.Exp(output[k] - maxVal)
		sum += output[k]
	}

	// Normalize
	for k := 0; k < nn.OutputSize; k++ {
		output[k] /= sum
	}

	return output, nil
}

// Predict returns the index of the highest output value
func (nn *Network) Predict(input []float64) (int, error) {
	output, err := nn.Forward(input)
	if err != nil {
		return -1, err
	}

	// Find the index of the maximum value
	maxIdx := 0
	maxVal := output[0]
	for i := 1; i < len(output); i++ {
		if output[i] > maxVal {
			maxVal = output[i]
			maxIdx = i
		}
	}

	return maxIdx, nil
}

// ForwardBatch runs a forward pass for a batch of inputs
func (nn *Network) ForwardBatch(inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, errors.New("empty batch")
	}

	outputs := make([][]float64, len(inputs))
	for i, input := range inputs {
		output, err := nn.Forward(input)
		if err != nil {
			return nil, err
		}
		outputs[i] = output
	}

	return outputs, nil
}

// PredictBatch returns the index of the highest output value for a batch of inputs
func (nn *Network) PredictBatch(inputs [][]float64) ([]int, error) {
	if len(inputs) == 0 {
		return nil, errors.New("empty batch")
	}

	predictions := make([]int, len(inputs))
	for i, input := range inputs {
		prediction, err := nn.Predict(input)
		if err != nil {
			return nil, err
		}
		predictions[i] = prediction
	}

	return predictions, nil
}

// GetInputSize returns the input size of the network
func (nn *Network) GetInputSize() int {
	return nn.InputSize
}

// GetOutputSize returns the output size of the network
func (nn *Network) GetOutputSize() int {
	return nn.OutputSize
}

// Close is a no-op for CPU networks but satisfies the NeuralNetwork interface
func (nn *Network) Close() error {
	return nil
}

// Ensure Network implements the BatchedNeuralNetwork interface
var _ common.BatchedNeuralNetwork = (*Network)(nil)
