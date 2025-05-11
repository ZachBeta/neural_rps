package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Simple neural network implementation
type Network struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	Weights1   [][]float64
	Bias1      []float64
	Weights2   [][]float64
	Bias2      []float64
}

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

	// Initialize weights with small random values
	for i := 0; i < inputSize; i++ {
		nn.Weights1[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			nn.Weights1[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	for i := 0; i < hiddenSize; i++ {
		nn.Weights2[i] = make([]float64, outputSize)
		for j := 0; j < outputSize; j++ {
			nn.Weights2[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	return nn
}

func (nn *Network) Forward(input []float64) []float64 {
	// Hidden layer
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

	// Output layer
	output := make([]float64, nn.OutputSize)
	for k := 0; k < nn.OutputSize; k++ {
		sum := nn.Bias2[k]
		for j := 0; j < nn.HiddenSize; j++ {
			sum += hidden[j] * nn.Weights2[j][k]
		}
		// Softmax will be applied later if needed
		output[k] = sum
	}

	return output
}

func main() {
	// Set random seed
	rand.Seed(time.Now().UnixNano())

	fmt.Println("Simple Neural Network CPU Benchmark")
	fmt.Println("=================================")

	// Network dimensions
	inputSize := 120
	hiddenSize := 256
	outputSize := 45

	// Create network
	fmt.Println("\nInitializing network...")
	start := time.Now()
	network := NewNetwork(inputSize, hiddenSize, outputSize)
	fmt.Printf("Network creation: %v\n", time.Since(start))

	// Benchmark inference
	benchmarkInference(network, inputSize)

	fmt.Println("\nBenchmark complete!")
}

func benchmarkInference(network *Network, inputSize int) {
	fmt.Println("\nBenchmarking Single Inference...")
	numSamples := 1000

	// Generate random inputs
	inputs := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			inputs[i][j] = rand.Float64()*2 - 1
		}
	}

	// Single inference
	fmt.Println("Single Inference:")
	start := time.Now()
	for i := 0; i < numSamples; i++ {
		network.Forward(inputs[i])
	}
	elapsed := time.Since(start)
	fmt.Printf("  Total: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n",
		elapsed,
		elapsed/time.Duration(numSamples),
		float64(numSamples)/elapsed.Seconds())

	// Simulated batch processing
	batchSizes := []int{1, 16, 64, 128, 256}
	for _, batchSize := range batchSizes {
		if batchSize > numSamples {
			continue
		}

		fmt.Printf("Manual Batch (size = %d):\n", batchSize)
		numBatches := (numSamples + batchSize - 1) / batchSize
		start := time.Now()
		totalProcessed := 0

		for i := 0; i < numBatches; i++ {
			startIdx := i * batchSize
			endIdx := min(startIdx+batchSize, numSamples)

			for j := startIdx; j < endIdx; j++ {
				network.Forward(inputs[j])
			}

			totalProcessed += (endIdx - startIdx)
		}

		elapsed := time.Since(start)
		fmt.Printf("  Total: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n",
			elapsed,
			elapsed/time.Duration(totalProcessed),
			float64(totalProcessed)/elapsed.Seconds())
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
