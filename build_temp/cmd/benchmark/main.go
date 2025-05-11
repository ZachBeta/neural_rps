package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/zachbeta/neural_rps/pkg/neural/cpu"
	"github.com/zachbeta/neural_rps/pkg/neural/gpu"
)

const (
	defaultInputSize  = 64
	defaultHiddenSize = 128
	defaultOutputSize = 8
	defaultAddr       = "localhost:50052"
)

func generateRandomInput(size int) []float64 {
	input := make([]float64, size)
	for i := range input {
		input[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
	}
	return input
}

func generateRandomBatch(batchSize, inputSize int) [][]float64 {
	batch := make([][]float64, batchSize)
	for i := range batch {
		batch[i] = generateRandomInput(inputSize)
	}
	return batch
}

func benchmarkCPUSingle(network *cpu.RPSCPUPolicyNetwork, inputSize, iterations int) time.Duration {
	input := generateRandomInput(inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.Predict(input)
		if err != nil {
			log.Fatalf("Error during CPU prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkCPUBatch(network *cpu.RPSCPUPolicyNetwork, inputSize, batchSize, iterations int) time.Duration {
	batch := generateRandomBatch(batchSize, inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.PredictBatch(batch)
		if err != nil {
			log.Fatalf("Error during CPU batch prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkGPUSingle(network *gpu.RPSGPUPolicyNetwork, inputSize, iterations int) time.Duration {
	input := generateRandomInput(inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.Predict(input)
		if err != nil {
			log.Fatalf("Error during GPU prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkGPUBatch(network *gpu.RPSGPUPolicyNetwork, inputSize, batchSize, iterations int) time.Duration {
	batch := generateRandomBatch(batchSize, inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.BatchPredict(batch)
		if err != nil {
			log.Fatalf("Error during GPU batch prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func main() {
	// Define command line flags
	inputSize := flag.Int("input-size", defaultInputSize, "Size of the input layer")
	hiddenSize := flag.Int("hidden-size", defaultHiddenSize, "Size of the hidden layer")
	outputSize := flag.Int("output-size", defaultOutputSize, "Size of the output layer")
	iterations := flag.Int("iterations", 100, "Number of iterations for each benchmark")
	batchSize := flag.Int("batch-size", 32, "Batch size for batch processing")
	addr := flag.String("addr", defaultAddr, "The address of the gRPC server")
	cpuOnly := flag.Bool("cpu-only", false, "Run only CPU benchmarks")
	gpuOnly := flag.Bool("gpu-only", false, "Run only GPU benchmarks")

	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Printf("Benchmarking neural networks with:\n")
	fmt.Printf("  Input size: %d\n", *inputSize)
	fmt.Printf("  Hidden size: %d\n", *hiddenSize)
	fmt.Printf("  Output size: %d\n", *outputSize)
	fmt.Printf("  Iterations: %d\n", *iterations)
	fmt.Printf("  Batch size: %d\n", *batchSize)
	fmt.Println()

	// Run CPU benchmarks
	if !*gpuOnly {
		fmt.Println("CPU Benchmarks:")
		cpuNetwork, err := cpu.NewRPSCPUPolicyNetwork(*inputSize, *hiddenSize, *outputSize)
		if err != nil {
			log.Fatalf("Failed to create CPU network: %v", err)
		}

		// Single prediction benchmark
		cpuSingleTime := benchmarkCPUSingle(cpuNetwork, *inputSize, *iterations)
		cpuSingleAvg := float64(cpuSingleTime.Microseconds()) / float64(*iterations)
		fmt.Printf("  Single prediction: %v (avg %.2f µs/prediction)\n", cpuSingleTime, cpuSingleAvg)

		// Batch prediction benchmark
		cpuBatchTime := benchmarkCPUBatch(cpuNetwork, *inputSize, *batchSize, *iterations)
		cpuBatchAvg := float64(cpuBatchTime.Microseconds()) / float64(*iterations*(*batchSize))
		fmt.Printf("  Batch prediction:  %v (avg %.2f µs/prediction)\n", cpuBatchTime, cpuBatchAvg)
		fmt.Println()
	}

	// Run GPU benchmarks
	if !*cpuOnly {
		fmt.Println("GPU Benchmarks:")
		gpuNetwork, err := gpu.NewRPSGPUPolicyNetwork(*addr)
		if err != nil {
			log.Fatalf("Failed to create GPU network: %v", err)
		}
		defer gpuNetwork.Close()

		// Single prediction benchmark
		gpuSingleTime := benchmarkGPUSingle(gpuNetwork, *inputSize, *iterations)
		gpuSingleAvg := float64(gpuSingleTime.Microseconds()) / float64(*iterations)
		fmt.Printf("  Single prediction: %v (avg %.2f µs/prediction)\n", gpuSingleTime, gpuSingleAvg)

		// Batch prediction benchmark
		gpuBatchTime := benchmarkGPUBatch(gpuNetwork, *inputSize, *batchSize, *iterations)
		gpuBatchAvg := float64(gpuBatchTime.Microseconds()) / float64(*iterations*(*batchSize))
		fmt.Printf("  Batch prediction:  %v (avg %.2f µs/prediction)\n", gpuBatchTime, gpuBatchAvg)

		// Print network stats
		stats := gpuNetwork.GetStats()
		fmt.Printf("  Total calls: %d, Total positions: %d\n", stats.TotalCalls, stats.TotalBatchSize)
		fmt.Printf("  Avg latency: %.2f µs, Avg batch size: %.2f\n", stats.AvgLatencyUs, stats.AvgBatchSize)
		fmt.Println()
	}

	// Print comparison if both CPU and GPU were benchmarked
	if !*cpuOnly && !*gpuOnly {
		// Note: These values are calculated in the blocks above, but would need to be returned
		// or made accessible to actually compare here. This is just placeholder code.
		fmt.Println("Performance Comparison:")
		fmt.Println("  Single prediction speedup: GPU is X times faster than CPU")
		fmt.Println("  Batch prediction speedup: GPU is Y times faster than CPU")
	}
}
