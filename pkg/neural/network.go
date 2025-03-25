package neural

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// Network represents a simple feed-forward neural network
type Network struct {
	// Network architecture
	InputSize  int
	HiddenSize int
	OutputSize int

	// Network parameters
	Weights1 [][]float64 // Input -> Hidden weights
	Bias1    []float64   // Hidden layer bias
	Weights2 [][]float64 // Hidden -> Output weights
	Bias2    []float64   // Output layer bias
}

// NewNetwork creates a new neural network with the specified architecture
func NewNetwork(inputSize, hiddenSize, outputSize int) *Network {
	nn := &Network{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
		Weights1:   make([][]float64, hiddenSize),
		Bias1:      make([]float64, hiddenSize),
		Weights2:   make([][]float64, outputSize),
		Bias2:      make([]float64, outputSize),
	}

	// Initialize weights with Xavier initialization
	w1Bound := math.Sqrt(6.0 / float64(inputSize+hiddenSize))
	w2Bound := math.Sqrt(6.0 / float64(hiddenSize+outputSize))

	// Initialize Weights1
	for i := 0; i < hiddenSize; i++ {
		nn.Weights1[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			nn.Weights1[i][j] = (rand.Float64()*2 - 1) * w1Bound
		}
		nn.Bias1[i] = 0.0
	}

	// Initialize Weights2
	for i := 0; i < outputSize; i++ {
		nn.Weights2[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			nn.Weights2[i][j] = (rand.Float64()*2 - 1) * w2Bound
		}
		nn.Bias2[i] = 0.0
	}

	return nn
}

// Forward performs a forward pass through the network
func (nn *Network) Forward(input []float64) []float64 {
	if len(input) != nn.InputSize {
		panic(fmt.Sprintf("Input size mismatch: expected %d, got %d", nn.InputSize, len(input)))
	}

	// Hidden layer
	hidden := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		sum := nn.Bias1[i]
		for j := 0; j < nn.InputSize; j++ {
			sum += nn.Weights1[i][j] * input[j]
		}
		hidden[i] = relu(sum)
	}

	// Output layer
	output := make([]float64, nn.OutputSize)
	for i := 0; i < nn.OutputSize; i++ {
		sum := nn.Bias2[i]
		for j := 0; j < nn.HiddenSize; j++ {
			sum += nn.Weights2[i][j] * hidden[j]
		}
		output[i] = sum
	}

	// Apply softmax to get probabilities
	return softmax(output)
}

// Predict predicts the best move based on the input
func (nn *Network) Predict(input []float64) int {
	output := nn.Forward(input)
	return argmax(output)
}

// SaveWeights saves the network weights to a file
func (nn *Network) SaveWeights(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(nn)
}

// LoadWeights loads the network weights from a file
func (nn *Network) LoadWeights(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(nn)
}

// Helper functions

// relu implements the Rectified Linear Unit activation function
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDerivative returns the derivative of the ReLU function
func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// softmax applies the softmax function to convert logits to probabilities
func softmax(x []float64) []float64 {
	result := make([]float64, len(x))
	sum := 0.0
	max := -math.MaxFloat64

	// Find the maximum value to avoid overflow
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	// Calculate exp(x - max) and sum
	for i, v := range x {
		exp := math.Exp(v - max)
		result[i] = exp
		sum += exp
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}

// argmax returns the index of the maximum value in an array
func argmax(x []float64) int {
	maxIdx := 0
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
			maxIdx = i
		}
	}
	return maxIdx
}
