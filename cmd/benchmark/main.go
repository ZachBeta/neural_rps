package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/zachbeta/neural_rps/pkg/common"
	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/neural/cpu"
)

// NetworkFactory is a function that creates neural networks
type NetworkFactory func(inputSize, hiddenSize, outputSize int) (common.BatchedNeuralNetwork, error)

// cpuNetworkFactory creates a CPU-based neural network
func cpuNetworkFactory(inputSize, hiddenSize, outputSize int) (common.BatchedNeuralNetwork, error) {
	return cpu.NewNetwork(inputSize, hiddenSize, outputSize), nil
}

// getNetworkFactory returns the appropriate network factory based on build tags
func getNetworkFactory(useGPU bool) NetworkFactory {
	if useGPU {
		// This function will be defined in the gpu build
		if factory := getGPUNetworkFactory(); factory != nil {
			return factory
		}
		fmt.Println("Warning: GPU support not available, falling back to CPU")
	}
	return cpuNetworkFactory
}

// Default empty implementation for CPU build
var getGPUNetworkFactory = func() NetworkFactory {
	return nil
}

func main() {
	// Parse command line flags
	useGPU := flag.Bool("gpu", false, "Use GPU acceleration if available")
	batchSize := flag.Int("batch", 64, "Batch size for inference")
	numSamples := flag.Int("samples", 10000, "Number of samples to process")
	hiddenSize := flag.Int("hidden", 128, "Hidden layer size")
	flag.Parse()

	// Create a game for feature extraction
	game := game.NewRPSCardGame(15, 5, 10)

	// Get input and output sizes
	features := game.GetBoardAsFeatures()
	inputSize := len(features)
	outputSize := 27 // 9 positions * 3 card types

	fmt.Printf("Running benchmark with %d samples (batch size: %d)\n", *numSamples, *batchSize)
	fmt.Printf("Network architecture: %d -> %d -> %d\n", inputSize, *hiddenSize, outputSize)

	// Get the appropriate network factory
	factory := getNetworkFactory(*useGPU)

	// Create the network
	start := time.Now()
	network, err := factory(inputSize, *hiddenSize, outputSize)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		return
	}
	defer network.Close()

	fmt.Printf("Network creation time: %v\n", time.Since(start))

	// Prepare input data
	inputs := make([][]float64, *batchSize)
	for i := range inputs {
		// Randomize the game state a bit for variation
		game.GetRandomMove()
		inputs[i] = game.GetBoardAsFeatures()
	}

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		_, err := network.ForwardBatch(inputs[:10])
		if err != nil {
			fmt.Printf("Error during warmup: %v\n", err)
			return
		}
	}

	// Run benchmark
	fmt.Println("Running benchmark...")
	start = time.Now()

	var totalBatches int
	for processed := 0; processed < *numSamples; processed += *batchSize {
		// Adjust batch size for the last iteration if needed
		currentBatchSize := *batchSize
		if processed+currentBatchSize > *numSamples {
			currentBatchSize = *numSamples - processed
		}

		// Run forward pass
		_, err := network.ForwardBatch(inputs[:currentBatchSize])
		if err != nil {
			fmt.Printf("Error during benchmark: %v\n", err)
			return
		}

		totalBatches++
	}

	duration := time.Since(start)

	// Print results
	totalSamples := float64(*numSamples)
	samplesPerSecond := totalSamples / duration.Seconds()

	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("  Mode:               %s\n", getModeName(*useGPU))
	fmt.Printf("  Total samples:      %d\n", *numSamples)
	fmt.Printf("  Total batches:      %d\n", totalBatches)
	fmt.Printf("  Total time:         %v\n", duration)
	fmt.Printf("  Samples per second: %.2f\n", samplesPerSecond)
	fmt.Printf("  Time per sample:    %.2f µs\n", (duration.Seconds()*1000000)/totalSamples)
}

// getModeName returns a descriptive name for the current mode
func getModeName(useGPU bool) string {
	if useGPU {
		// This will be overridden in the GPU build
		if getModeNameGPU != nil {
			return getModeNameGPU()
		}
		return "GPU (unavailable, using CPU)"
	}
	return "CPU"
}

// Default empty implementation for CPU build
var getModeNameGPU func() string
