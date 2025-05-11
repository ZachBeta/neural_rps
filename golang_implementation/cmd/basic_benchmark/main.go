package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/zachbeta/neural_rps/pkg/neural"
)

func main() {
	// Parse command line flags
	flag.Parse()

	fmt.Println("Neural RPS Basic CPU Benchmark")
	fmt.Println("===========================")

	// Initialize networks
	inputSize := 120 // Adjust based on your feature representation
	hiddenSize := 256
	outputSize := 9 * 5 // 9 positions * 5 cards (example)

	// Create CPU network
	fmt.Println("\nInitializing CPU network...")
	cpuStart := time.Now()
	cpuNetwork := neural.NewNetwork(inputSize, hiddenSize, outputSize)
	fmt.Printf("CPU network creation: %v\n", time.Since(cpuStart))

	// Benchmark single forward passes
	benchmarkCPUSingleInference(cpuNetwork, inputSize)

	fmt.Println("\nBenchmark complete!")
}

// benchmarkCPUSingleInference tests CPU inference performance
func benchmarkCPUSingleInference(cpuNetwork *neural.Network, inputSize int) {
	fmt.Println("\nBenchmarking CPU Single Inference...")
	numSamples := 1000

	// Generate random inputs
	inputs := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = generateRandomInput(inputSize)
	}

	// Benchmark CPU single inference
	fmt.Println("CPU Single Inference:")
	cpuStart := time.Now()
	for i := 0; i < numSamples; i++ {
		cpuNetwork.Forward(inputs[i])
	}
	cpuElapsed := time.Since(cpuStart)
	fmt.Printf("  Total: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n",
		cpuElapsed,
		cpuElapsed/time.Duration(numSamples),
		float64(numSamples)/cpuElapsed.Seconds())

	// Benchmark CPU batch inference - manually (without batched implementation)
	batchSizes := []int{1, 16, 64, 128, 256}

	for _, batchSize := range batchSizes {
		if batchSize > numSamples {
			continue
		}

		fmt.Printf("CPU Manual Batch (size = %d):\n", batchSize)
		numBatches := (numSamples + batchSize - 1) / batchSize

		start := time.Now()
		totalProcessed := 0

		for i := 0; i < numBatches; i++ {
			startIdx := i * batchSize
			endIdx := min(startIdx+batchSize, numSamples)

			// Process each input individually (since we don't have true batching)
			for j := startIdx; j < endIdx; j++ {
				cpuNetwork.Forward(inputs[j])
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

// Helper function to generate random input
func generateRandomInput(size int) []float64 {
	input := make([]float64, size)
	for i := range input {
		input[i] = rand.Float64()*2 - 1
	}
	return input
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
