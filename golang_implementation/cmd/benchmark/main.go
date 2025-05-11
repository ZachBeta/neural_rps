package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/mcts"
	"github.com/zachbeta/neural_rps/pkg/neural"
)

func main() {
	// Parse command line flags
	batchSize := flag.Int("batch", 128, "Batch size for GPU operations")
	simulations := flag.Int("sims", 800, "Number of MCTS simulations")
	flag.Parse()

	fmt.Println("Neural RPS GPU Benchmark")
	fmt.Println("=======================")

	// Create a random game state for benchmarking
	gameState := game.NewRPSCardGame(15, 5, 10)

	// Initialize networks
	inputSize := 120 // Adjust based on your feature representation
	hiddenSize := 256
	outputSize := 9 * 5 // 9 positions * 5 cards (example)

	// Create CPU network
	fmt.Println("\nInitializing CPU network...")
	cpuStart := time.Now()
	cpuNetwork := neural.NewNetwork(inputSize, hiddenSize, outputSize)
	fmt.Printf("CPU network creation: %v\n", time.Since(cpuStart))

	// Create GPU network
	fmt.Println("\nInitializing GPU network...")
	gpuStart := time.Now()
	gpuNetwork, err := neural.NewRPSTFPolicyNetwork(inputSize, hiddenSize, outputSize)
	if err != nil {
		fmt.Printf("Error creating GPU network: %v\n", err)
		return
	}
	fmt.Printf("GPU network creation: %v\n", time.Since(gpuStart))
	defer gpuNetwork.Close()

	// Load weights from CPU to GPU
	fmt.Println("\nTransferring weights from CPU to GPU...")
	transferStart := time.Now()
	err = gpuNetwork.LoadFromCPUNetwork(cpuNetwork)
	if err != nil {
		fmt.Printf("Error transferring weights: %v\n", err)
		return
	}
	fmt.Printf("Weight transfer: %v\n", time.Since(transferStart))

	// Benchmark single forward passes
	benchmarkSingleInference(cpuNetwork, gpuNetwork, inputSize)

	// Benchmark batched forward passes
	benchmarkBatchedInference(gpuNetwork, inputSize, *batchSize)

	// Benchmark MCTS
	benchmarkMCTS(gameState, cpuNetwork, gpuNetwork, *simulations, *batchSize)
}

// benchmarkSingleInference compares CPU vs GPU for single inference
func benchmarkSingleInference(cpuNetwork *neural.Network, gpuNetwork *neural.RPSTFPolicyNetwork, inputSize int) {
	fmt.Println("\nBenchmarking Single Inference...")
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

	// Benchmark GPU single inference
	fmt.Println("GPU Single Inference:")
	gpuStart := time.Now()
	for i := 0; i < numSamples; i++ {
		_, err := gpuNetwork.Forward(inputs[i])
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}
	}
	gpuElapsed := time.Since(gpuStart)
	fmt.Printf("  Total: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n",
		gpuElapsed,
		gpuElapsed/time.Duration(numSamples),
		float64(numSamples)/gpuElapsed.Seconds())

	// Calculate speedup
	speedup := float64(cpuElapsed) / float64(gpuElapsed)
	fmt.Printf("GPU Speedup for single inference: %.2fx\n", speedup)
}

// benchmarkBatchedInference tests GPU batched inference with different batch sizes
func benchmarkBatchedInference(gpuNetwork *neural.RPSTFPolicyNetwork, inputSize, batchSize int) {
	fmt.Println("\nBenchmarking Batched Inference...")
	numSamples := 10000

	// Generate random inputs
	inputs := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = generateRandomInput(inputSize)
	}

	// Test different batch sizes
	batchSizes := []int{1, 16, 64, 128, 256, 512}
	for _, size := range batchSizes {
		if size > numSamples {
			continue
		}

		fmt.Printf("GPU Batch Inference (batch size = %d):\n", size)

		// Create batches
		numBatches := (numSamples + size - 1) / size
		batches := make([][][]float64, numBatches)

		for i := 0; i < numBatches; i++ {
			startIdx := i * size
			endIdx := min(startIdx+size, numSamples)
			batches[i] = inputs[startIdx:endIdx]
		}

		// Run batched inference
		start := time.Now()
		totalProcessed := 0

		for i := 0; i < numBatches; i++ {
			batch := batches[i]
			_, err := gpuNetwork.ForwardBatch(batch)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				return
			}
			totalProcessed += len(batch)
		}

		elapsed := time.Since(start)
		fmt.Printf("  Total: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n",
			elapsed,
			elapsed/time.Duration(totalProcessed),
			float64(totalProcessed)/elapsed.Seconds())
	}
}

// benchmarkMCTS compares traditional MCTS vs batched MCTS
func benchmarkMCTS(gameState *game.RPSCardGame, cpuNetwork *neural.Network, gpuNetwork *neural.RPSTFPolicyNetwork, simulations, batchSize int) {
	fmt.Println("\nBenchmarking MCTS...")

	// Standard MCTS settings
	mctsParams := mcts.MCTSParams{
		NumSimulations:   simulations,
		ExplorationConst: 1.0,
	}

	// Standard MCTS with CPU network
	fmt.Println("Standard MCTS with CPU network:")
	start := time.Now()
	standardMCTS := mcts.NewMCTS(cpuNetwork, cpuNetwork, mctsParams)
	standardMCTS.SetRootState(gameState.Clone())
	cpuMove := standardMCTS.Search()
	cpuElapsed := time.Since(start)
	fmt.Printf("  Time: %v, Nodes per second: %.2f\n",
		cpuElapsed,
		float64(simulations)/cpuElapsed.Seconds())
	fmt.Printf("  Selected move: %+v\n", cpuMove)

	// Batched MCTS with GPU network
	fmt.Println("Batched MCTS with GPU network:")
	start = time.Now()
	batchedMCTS := mcts.NewBatchedMCTS(gpuNetwork, gpuNetwork, mctsParams, batchSize)
	batchedMCTS.SetRootState(gameState.Clone())
	gpuMove := batchedMCTS.Search()
	gpuElapsed := time.Since(start)
	fmt.Printf("  Time: %v, Nodes per second: %.2f\n",
		gpuElapsed,
		float64(simulations)/gpuElapsed.Seconds())
	fmt.Printf("  Selected move: %+v\n", gpuMove)

	// Calculate speedup
	speedup := float64(cpuElapsed) / float64(gpuElapsed)
	fmt.Printf("GPU Speedup for MCTS: %.2fx\n", speedup)
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
