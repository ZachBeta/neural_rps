package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// RPSCPUPolicyNetwork is a simple neural network implementation in pure Go
type RPSCPUPolicyNetwork struct {
	// Network dimensions
	InputSize  int
	HiddenSize int
	OutputSize int

	// Network weights and biases
	Weights1 [][]float64
	Bias1    []float64
	Weights2 [][]float64
	Bias2    []float64

	// Performance metrics
	totalTime      time.Duration
	totalCalls     int
	totalBatchSize int
}

// NewRPSCPUPolicyNetwork creates a new policy network with random weights
func NewRPSCPUPolicyNetwork(inputSize, hiddenSize, outputSize int) (*RPSCPUPolicyNetwork, error) {
	if inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0 {
		return nil, fmt.Errorf("invalid network dimensions: input=%d, hidden=%d, output=%d",
			inputSize, hiddenSize, outputSize)
	}

	// Initialize random weights with Xavier/Glorot initialization
	weights1 := make([][]float64, inputSize)
	weightsScale1 := math.Sqrt(2.0 / float64(inputSize+hiddenSize))

	for i := 0; i < inputSize; i++ {
		weights1[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			weights1[i][j] = (rand.Float64()*2 - 1) * weightsScale1
		}
	}

	bias1 := make([]float64, hiddenSize)

	weights2 := make([][]float64, hiddenSize)
	weightsScale2 := math.Sqrt(2.0 / float64(hiddenSize+outputSize))

	for i := 0; i < hiddenSize; i++ {
		weights2[i] = make([]float64, outputSize)
		for j := 0; j < outputSize; j++ {
			weights2[i][j] = (rand.Float64()*2 - 1) * weightsScale2
		}
	}

	bias2 := make([]float64, outputSize)

	return &RPSCPUPolicyNetwork{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
		Weights1:   weights1,
		Bias1:      bias1,
		Weights2:   weights2,
		Bias2:      bias2,
	}, nil
}

// Forward performs a forward pass through the neural network
func (n *RPSCPUPolicyNetwork) Forward(input []float64) ([]float64, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize++

	if len(input) != n.InputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", n.InputSize, len(input))
	}

	// First layer
	hidden := make([]float64, n.HiddenSize)
	for j := 0; j < n.HiddenSize; j++ {
		sum := 0.0
		for i := 0; i < n.InputSize; i++ {
			sum += input[i] * n.Weights1[i][j]
		}
		hidden[j] = relu(sum + n.Bias1[j])
	}

	// Output layer
	output := make([]float64, n.OutputSize)
	for j := 0; j < n.OutputSize; j++ {
		sum := 0.0
		for i := 0; i < n.HiddenSize; i++ {
			sum += hidden[i] * n.Weights2[i][j]
		}
		output[j] = sum + n.Bias2[j]
	}

	// Apply softmax
	softmax(output)

	n.totalTime += time.Since(start)

	return output, nil
}

// Predict returns the index of the highest probability
func (n *RPSCPUPolicyNetwork) Predict(input []float64) (int, error) {
	output, err := n.Forward(input)
	if err != nil {
		return -1, err
	}

	return argmax(output), nil
}

// ForwardBatch performs forward passes for multiple inputs
func (n *RPSCPUPolicyNetwork) ForwardBatch(inputs [][]float64) ([][]float64, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize += len(inputs)

	outputs := make([][]float64, len(inputs))
	for i, input := range inputs {
		output, err := n.Forward(input)
		if err != nil {
			return nil, err
		}
		outputs[i] = output
	}

	n.totalTime += time.Since(start)

	return outputs, nil
}

// PredictBatch performs predictions for multiple inputs
func (n *RPSCPUPolicyNetwork) PredictBatch(inputs [][]float64) ([]int, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize += len(inputs)

	predictions := make([]int, len(inputs))
	for i, input := range inputs {
		pred, err := n.Predict(input)
		if err != nil {
			return nil, err
		}
		predictions[i] = pred
	}

	n.totalTime += time.Since(start)

	return predictions, nil
}

// Helper functions
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func softmax(values []float64) {
	// Find the maximum value to avoid numerical instability
	max := values[0]
	for _, val := range values {
		if val > max {
			max = val
		}
	}

	// Calculate exponentials and sum
	sum := 0.0
	for i, val := range values {
		exp := math.Exp(val - max)
		values[i] = exp
		sum += exp
	}

	// Normalize
	for i := range values {
		values[i] /= sum
	}
}

func argmax(values []float64) int {
	maxIdx := 0
	maxVal := values[0]

	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}

// Close is a no-op for CPU networks but required by the interface
func (n *RPSCPUPolicyNetwork) Close() error {
	return nil
}

// GetInputSize returns the input size of the network
func (n *RPSCPUPolicyNetwork) GetInputSize() int {
	return n.InputSize
}

// GetOutputSize returns the output size of the network
func (n *RPSCPUPolicyNetwork) GetOutputSize() int {
	return n.OutputSize
}
