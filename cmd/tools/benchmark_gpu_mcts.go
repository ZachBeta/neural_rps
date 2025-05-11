package main

import (
	"context"
	"flag"
	"log"
	"time"

	"github.com/zachbeta/neural_rps/pkg/agents/mcts"
	"github.com/zachbeta/neural_rps/pkg/agents/mcts_adapter"
)

func main() {
	// Command line arguments
	useCPU := flag.Bool("cpu", false, "Use CPU-only MCTS (default: GPU)")
	iterations := flag.Int("n", 1000, "Number of MCTS iterations")
	batchSize := flag.Int("batch", 64, "Batch size for GPU operations")
	serviceAddr := flag.String("service", "localhost:50052", "GPU neural service address")
	flag.Parse()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	var agent mcts.MCTSAgent
	var state mcts.GameState
	var err error

	// Create the appropriate agent using our factory
	if *useCPU {
		log.Println("Using CPU-only MCTS...")
		agent, state, err = mcts_adapter.CreateCPUAgent(*iterations, *batchSize)
	} else {
		log.Printf("Using GPU-accelerated MCTS with service at %s...\n", *serviceAddr)
		agent, state, err = mcts_adapter.CreateGPUAgent(*serviceAddr, *iterations, *batchSize)
	}

	if err != nil {
		log.Fatalf("Failed to create MCTS agent: %v", err)
	}

	// Set the root state
	agent.SetRootState(state)

	// Run benchmark
	log.Printf("Running benchmark with %d iterations...\n", *iterations)
	start := time.Now()

	// Run search
	bestMove := agent.Search(ctx)

	elapsed := time.Since(start)

	// Print results
	iterationsPerSecond := float64(*iterations) / elapsed.Seconds()
	log.Printf("Search completed in %v", elapsed)
	log.Printf("Iterations per second: %.2f", iterationsPerSecond)
	log.Printf("Best move: %v", bestMove)

	// Print performance stats
	stats := agent.GetStats()
	log.Println("\nPerformance Statistics:")
	for key, value := range stats {
		log.Printf("  %s: %v", key, value)
	}

	// Close agent to release resources
	agent.Close()
}
