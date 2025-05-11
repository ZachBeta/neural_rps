package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"
	"time"

	"github.com/zachbeta/neural_rps/pkg/agents/mcts"
	"github.com/zachbeta/neural_rps/pkg/game"
)

func main() {
	// Parse command line arguments
	iterations := flag.Int("iterations", 1000, "Number of MCTS iterations per move")
	games := flag.Int("games", 100, "Number of self-play games")
	batchSize := flag.Int("batch-size", 64, "Batch size for GPU operations")
	serviceAddr := flag.String("service", "localhost:50052", "GPU neural service address")
	outputDir := flag.String("output", "output/gpu_training", "Directory for training output")
	enableProfile := flag.Bool("profile", false, "Enable CPU profiling")
	flag.Parse()

	// Set up logging
	logFile, err := os.Create(filepath.Join(*outputDir, "training.log"))
	if err != nil {
		log.Fatalf("Failed to create log file: %v", err)
	}
	defer logFile.Close()
	log.SetOutput(logFile)

	// Start CPU profiling if enabled
	if *enableProfile {
		profFile, err := os.Create(filepath.Join(*outputDir, "cpu.prof"))
		if err != nil {
			log.Fatalf("Failed to create profile file: %v", err)
		}
		defer profFile.Close()
		if err := pprof.StartCPUProfile(profFile); err != nil {
			log.Fatalf("Failed to start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Record start time
	startTime := time.Now()

	// Create GPU-accelerated MCTS agent
	params := mcts.DefaultMCTSParams()
	params.NumSimulations = *iterations

	agent, err := mcts.NewGPUBatchedMCTS(*serviceAddr, params)
	if err != nil {
		log.Fatalf("Failed to create GPU-accelerated MCTS: %v", err)
	}
	defer agent.Close()

	// Set batch size
	agent.SetBatchSize(*batchSize)

	// Run self-play games
	log.Printf("Starting %d self-play games with %d iterations per move\n", *games, *iterations)

	totalNodes := 0
	totalMoves := 0
	gameResults := make([]string, 0, *games)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	defer cancel()

	for i := 0; i < *games; i++ {
		gameStart := time.Now()

		// Initialize game
		state := game.NewRPSCardGame(9, 3, 9)

		// Play game
		moveCount := 0
		for !state.IsGameOver() {
			// Set root state for MCTS
			adaptedState := mcts.NewRPSGameStateAdapter(state)
			agent.SetRootState(adaptedState)

			// Get best move
			move := agent.Search(ctx)

			// Apply move
			state.ApplyMove(move)
			moveCount++
		}

		// Record game result
		winner := state.GetWinner()
		gameResults = append(gameResults, fmt.Sprintf("Game %d: Winner=%v, Moves=%d", i+1, winner, moveCount))

		// Update statistics
		stats := agent.GetStats()
		totalNodes += stats["total_nodes"].(int)
		totalMoves += moveCount

		gameDuration := time.Since(gameStart)
		log.Printf("Game %d completed in %v (%d moves, %v/move)\n",
			i+1, gameDuration, moveCount, gameDuration/time.Duration(moveCount))
	}

	// Compute statistics
	totalDuration := time.Since(startTime)

	// Write results to output file
	resultsFile, err := os.Create(filepath.Join(*outputDir, "results.txt"))
	if err != nil {
		log.Fatalf("Failed to create results file: %v", err)
	}
	defer resultsFile.Close()

	fmt.Fprintf(resultsFile, "GPU Training Results\n")
	fmt.Fprintf(resultsFile, "=====================\n\n")
	fmt.Fprintf(resultsFile, "Configuration:\n")
	fmt.Fprintf(resultsFile, "- MCTS Iterations: %d\n", *iterations)
	fmt.Fprintf(resultsFile, "- Self-play Games: %d\n", *games)
	fmt.Fprintf(resultsFile, "- Batch Size: %d\n", *batchSize)
	fmt.Fprintf(resultsFile, "- Service: %s\n\n", *serviceAddr)

	fmt.Fprintf(resultsFile, "Summary Statistics:\n")
	fmt.Fprintf(resultsFile, "- Total Duration: %v\n", totalDuration)
	fmt.Fprintf(resultsFile, "- Avg. Game Duration: %v\n", totalDuration/time.Duration(*games))
	fmt.Fprintf(resultsFile, "- Total Nodes: %d\n", totalNodes)
	fmt.Fprintf(resultsFile, "- Total Moves: %d\n", totalMoves)
	fmt.Fprintf(resultsFile, "- Nodes/Second: %.2f\n", float64(totalNodes)/totalDuration.Seconds())
	fmt.Fprintf(resultsFile, "- Moves/Second: %.2f\n\n", float64(totalMoves)/totalDuration.Seconds())

	fmt.Fprintf(resultsFile, "Game Results:\n")
	for _, result := range gameResults {
		fmt.Fprintf(resultsFile, "%s\n", result)
	}

	log.Printf("Training completed in %v\n", totalDuration)
	log.Printf("Results saved to %s\n", filepath.Join(*outputDir, "results.txt"))
}
