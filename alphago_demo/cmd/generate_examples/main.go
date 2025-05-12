package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts" // Needed for default params
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

const (
	// Default Game parameters (should match training if possible)
	defaultDeckSize  = 21
	defaultHandSize  = 5
	defaultMaxRounds = 10

	// Default Generation parameters
	defaultHiddenSize  = 64
	defaultNumGames    = 500
	defaultSims        = 100
	defaultExploration = 1.5
)

func main() {
	// --- Flags ---
	hiddenSize := flag.Int("hidden", defaultHiddenSize, "Hidden neurons for the placeholder networks")
	numGames := flag.Int("games", defaultNumGames, "Number of self-play games to generate")
	sims := flag.Int("sims", defaultSims, "MCTS simulations per move during self-play")
	exploration := flag.Float64("exploration", defaultExploration, "MCTS exploration constant during self-play")
	outputPath := flag.String("output", "", "Path to save generated self-play examples (JSON format)")
	parallel := flag.Bool("parallel", false, "Use parallel execution for self-play generation")
	threads := flag.Int("threads", 0, "Specific number of threads to use for parallel generation (0 = auto)")

	flag.Parse()

	// --- Validation ---
	if *outputPath == "" {
		log.Fatal("Error: Output path must be specified using --output")
	}
	if *hiddenSize <= 0 {
		log.Fatal("Error: Hidden size must be positive")
	}
	if *numGames <= 0 {
		log.Fatal("Error: Number of games must be positive")
	}
	if *sims <= 0 {
		log.Fatal("Error: Number of MCTS simulations must be positive")
	}

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Printf("=== Generating Self-Play Examples ===\n")
	fmt.Printf("Parameters:\n")
	fmt.Printf("  Hidden Size: %d\n", *hiddenSize)
	fmt.Printf("  Num Games:   %d\n", *numGames)
	fmt.Printf("  MCTS Sims:   %d\n", *sims)
	fmt.Printf("  Exploration: %.2f\n", *exploration)
	fmt.Printf("  Output Path: %s\n", *outputPath)
	fmt.Printf("  Parallel:    %v\n", *parallel)
	fmt.Printf("  Threads:     %d (0=auto)\n", *threads)
	fmt.Println("------------------------------------")

	// --- Initialization ---
	// Create placeholder networks with random weights.
	// These are only used by MCTS during self-play to guide the search.
	// The actual training will happen later (e.g., in Python).
	policyNet := neural.NewRPSPolicyNetwork(*hiddenSize)
	valueNet := neural.NewRPSValueNetwork(*hiddenSize)

	// Configure self-play parameters
	spParams := training.DefaultRPSSelfPlayParams()
	spParams.NumGames = *numGames
	spParams.DeckSize = defaultDeckSize
	spParams.HandSize = defaultHandSize
	spParams.MaxRounds = defaultMaxRounds
	spParams.ForceParallel = *parallel
	spParams.NumThreads = *threads

	// Set MCTS parameters from flags
	mctsParams := mcts.DefaultRPSMCTSParams() // Start with defaults
	mctsParams.NumSimulations = *sims
	mctsParams.ExplorationConst = *exploration
	// Keep other MCTS defaults like Dirichlet noise settings
	spParams.MCTSParams = mctsParams

	sp := training.NewRPSSelfPlay(policyNet, valueNet, spParams)

	// --- Generation ---
	fmt.Printf("Generating %d self-play games...\n", *numGames)
	startTime := time.Now()
	examples := sp.GenerateGames(true) // verbose=true
	genTime := time.Since(startTime)
	fmt.Printf("Generation complete in %s. Generated %d examples.\n", genTime, len(examples))

	// --- Saving Examples ---
	fmt.Printf("Saving generated examples to %s...\n", *outputPath)
	saveStartTime := time.Now()
	jsonData, err := json.MarshalIndent(examples, "", "  ")
	if err != nil {
		log.Fatalf("Error: Failed to marshal examples to JSON: %v", err)
	}

	// Ensure output directory exists
	os.MkdirAll("output", 0755)

	err = os.WriteFile(*outputPath, jsonData, 0644)
	if err != nil {
		log.Fatalf("Error: Failed to write examples to file %s: %v", *outputPath, err)
	}
	saveTime := time.Since(saveStartTime)

	fmt.Printf("Successfully saved %d examples to %s (Save time: %s).\n", len(examples), *outputPath, saveTime)
	fmt.Println("Done.")
}
