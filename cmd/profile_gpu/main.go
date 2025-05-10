package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

func main() {
	// Parse command line flags
	cpuProfile := flag.String("cpuprofile", "", "write cpu profile to file")
	memProfile := flag.String("memprofile", "", "write memory profile to file")
	modelPath := flag.String("model", "", "path to policy network model file")
	useGPU := flag.Bool("gpu", false, "use GPU acceleration when possible")
	batchSize := flag.Int("batch", 64, "batch size for GPU operations")
	task := flag.String("task", "all", "task to profile (prediction, mcts, tournament, neat, all)")
	flag.Parse()

	// Setup CPU profiling if requested
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			fmt.Printf("Could not create CPU profile: %v\n", err)
			return
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Printf("Could not start CPU profile: %v\n", err)
			return
		}
		defer pprof.StopCPUProfile()
	}

	// Run requested profiling tasks
	switch *task {
	case "prediction":
		profilePrediction(*useGPU, *batchSize)
	case "mcts":
		profileMCTS(*useGPU, *batchSize, *modelPath)
	case "tournament":
		profileTournament(*useGPU, *modelPath)
	case "neat":
		profileNEAT(*useGPU, *batchSize)
	case "all":
		fmt.Println("=== Profiling Neural Network Prediction ===")
		profilePrediction(*useGPU, *batchSize)

		fmt.Println("\n=== Profiling MCTS Search ===")
		profileMCTS(*useGPU, *batchSize, *modelPath)

		fmt.Println("\n=== Profiling Tournament Play ===")
		profileTournament(*useGPU, *modelPath)

		fmt.Println("\n=== Profiling NEAT Generation ===")
		profileNEAT(*useGPU, *batchSize)
	default:
		fmt.Printf("Unknown task: %s\n", *task)
		flag.Usage()
		return
	}

	// Write memory profile if requested
	if *memProfile != "" {
		f, err := os.Create(*memProfile)
		if err != nil {
			fmt.Printf("Could not create memory profile: %v\n", err)
			return
		}
		defer f.Close()
		// Force garbage collection before taking memory profile
		// runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Printf("Could not write memory profile: %v\n", err)
			return
		}
	}
}

// profilePrediction benchmarks neural network forward pass performance
func profilePrediction(useGPU bool, batchSize int) {
	// Create policy network (CPU version always needed for comparison)
	cpuNet := neural.NewRPSPolicyNetwork(128)

	// Load weights if model path provided
	if len(os.Args) > 2 {
		modelPath := os.Args[2]
		if err := cpuNet.LoadFromFile(modelPath); err != nil {
			fmt.Printf("Error loading model: %v\n", err)
			return
		}
	}

	var gpuNet *neural.RPSTFPolicyNetwork
	if useGPU {
		gpuNet = neural.NewRPSTFPolicyNetwork(128)
		// In a real implementation, we would transfer weights from cpuNet to gpuNet
	}

	// Generate random game positions for benchmarking
	fmt.Println("Generating benchmark positions...")
	var positions []*game.RPSGame
	for i := 0; i < 1000; i++ {
		g := game.NewRPSGame(21, 5, 10)
		// Make some random moves to create diverse positions
		movesCount := i % 9 // Between 0-8 moves
		for j := 0; j < movesCount; j++ {
			moves := g.GetValidMoves()
			if len(moves) == 0 {
				break
			}
			g.MakeMove(moves[j%len(moves)])
		}
		positions = append(positions, g)
	}

	// Extract features for batch processing
	features := make([][]float64, len(positions))
	for i, pos := range positions {
		features[i] = pos.GetBoardAsFeatures()
	}

	// Profile CPU one-by-one
	fmt.Println("Profiling CPU (one-by-one)...")
	startCPU := time.Now()
	for _, pos := range positions {
		cpuNet.Predict(pos)
	}
	cpuTime := time.Since(startCPU)

	// Profile GPU if requested
	var gpuSingleTime, gpuBatchedTime time.Duration
	if useGPU && gpuNet != nil {
		// Profile GPU (single inference)
		fmt.Println("Profiling GPU (one-by-one)...")
		startGPU := time.Now()
		for _, pos := range positions {
			gpuNet.Predict(pos)
		}
		gpuSingleTime = time.Since(startGPU)

		// Profile GPU (batched)
		fmt.Printf("Profiling GPU (batched, size=%d)...\n", batchSize)
		startBatched := time.Now()
		for i := 0; i < len(positions); i += batchSize {
			end := i + batchSize
			if end > len(positions) {
				end = len(positions)
			}
			batch := features[i:end]
			gpuNet.PredictBatch(batch)
		}
		gpuBatchedTime = time.Since(startBatched)
	}

	// Print results
	fmt.Printf("\nResults for %d positions:\n", len(positions))
	fmt.Printf("CPU time: %v (%.1f pos/sec)\n", cpuTime, float64(len(positions))/cpuTime.Seconds())

	if useGPU && gpuNet != nil {
		fmt.Printf("GPU single time: %v (%.1f pos/sec)\n",
			gpuSingleTime, float64(len(positions))/gpuSingleTime.Seconds())
		fmt.Printf("GPU batched time: %v (%.1f pos/sec)\n",
			gpuBatchedTime, float64(len(positions))/gpuBatchedTime.Seconds())

		fmt.Printf("\nSpeedup factors:\n")
		fmt.Printf("GPU single vs CPU: %.1fx\n", cpuTime.Seconds()/gpuSingleTime.Seconds())
		fmt.Printf("GPU batched vs CPU: %.1fx\n", cpuTime.Seconds()/gpuBatchedTime.Seconds())
		fmt.Printf("GPU batched vs GPU single: %.1fx\n", gpuSingleTime.Seconds()/gpuBatchedTime.Seconds())
	}
}

// profileMCTS benchmarks MCTS search performance
func profileMCTS(useGPU bool, batchSize int, modelPath string) {
	// Load policy and value networks
	policyNet, valueNet, err := loadNetworks(modelPath)
	if err != nil {
		fmt.Printf("Error loading networks: %v\n", err)
		return
	}

	// Setup MCTS parameters
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = 400

	// Create game state
	g := game.NewRPSGame(21, 5, 10)
	// Make a couple of moves to create an interesting position
	moves := g.GetValidMoves()
	if len(moves) > 0 {
		g.MakeMove(moves[0])
	}
	if len(g.GetValidMoves()) > 0 {
		g.MakeMove(g.GetValidMoves()[0])
	}

	// Create CPU MCTS
	cpuMCTS := mcts.NewRPSMCTS(policyNet, valueNet, mctsParams)

	// Profile CPU MCTS
	fmt.Println("Profiling CPU MCTS search...")
	startCPU := time.Now()
	cpuMCTS.SetRootState(g)
	cpuNode := cpuMCTS.Search()
	cpuTime := time.Since(startCPU)
	cpuNodesEvaluated := cpuMCTS.GetNodesEvaluated()

	// In a real implementation, we would have GPU MCTS to compare
	if useGPU {
		fmt.Println("GPU MCTS profiling not yet implemented")
		// Here we would:
		// 1. Create GPU networks
		// 2. Create GPU-accelerated MCTS
		// 3. Run the same search
		// 4. Compare performance
	}

	// Print results
	fmt.Printf("\nMCTS Search Results:\n")
	fmt.Printf("CPU time: %v for %d nodes (%.1f nodes/sec)\n",
		cpuTime, cpuNodesEvaluated, float64(cpuNodesEvaluated)/cpuTime.Seconds())

	if cpuNode != nil && cpuNode.Move != nil {
		fmt.Printf("Best move: %v (value: %.3f)\n", *cpuNode.Move, cpuNode.Value)
	}
}

// profileTournament benchmarks tournament play
func profileTournament(useGPU bool, modelPath string) {
	fmt.Println("Tournament profiling requires running a separate command:")
	fmt.Println("For CPU: go run cmd/tournament/minimax_vs_neural.go --model", modelPath, "--depth 5 --games 10")

	if useGPU {
		fmt.Println("For GPU: Not yet implemented, would require GPU integration into tournament code")
	}
}

// profileNEAT benchmarks NEAT training
func profileNEAT(useGPU bool, batchSize int) {
	fmt.Println("NEAT training profiling requires running a separate command:")
	fmt.Println("For CPU: go run cmd/train_models/main.go --method neat --generations 1")

	if useGPU {
		fmt.Println("For GPU: Not yet implemented, would require GPU integration into NEAT evaluator")
	}
}

// loadNetworks loads policy and value networks from files
func loadNetworks(modelPath string) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork, error) {
	if modelPath == "" {
		// Create new networks with random weights
		return neural.NewRPSPolicyNetwork(128), neural.NewRPSValueNetwork(128), nil
	}

	// Load policy network
	policyNet, err := neural.LoadPolicyNetwork(modelPath)
	if err != nil {
		return nil, nil, err
	}

	// Try to load value network from corresponding file
	valuePath := modelPath
	// If path ends with _policy.model, change to _value.model
	if len(modelPath) > 13 && modelPath[len(modelPath)-13:] == "_policy.model" {
		valuePath = modelPath[:len(modelPath)-13] + "_value.model"
	}

	valueNet, err := neural.LoadValueNetwork(valuePath)
	if err != nil {
		// If value network can't be loaded, create a new one
		fmt.Printf("Could not load value network from %s, creating new one\n", valuePath)
		valueNet = neural.NewRPSValueNetwork(policyNet.GetHiddenSize())
	}

	return policyNet, valueNet, nil
}
