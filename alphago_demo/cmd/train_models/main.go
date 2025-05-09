package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10

	// Training parameters for model 1 (baseline)
	model1SelfPlayGames = 100
	model1Epochs        = 5
	model1HiddenSize    = 64 // Smaller network size

	// Training parameters for model 2 (trained longer)
	model2SelfPlayGames = 1000 // 10x more games
	model2Epochs        = 10   // 2x more epochs
	model2HiddenSize    = 128  // Larger network size

	// Tournament parameters
	tournamentGames = 30
	mctsSimulations = 200
)

func main() {
	// Parse command line flags
	smallRun := flag.Bool("small-run", false, "Run with reduced parameters for quick testing")
	parallel := flag.Bool("parallel", false, "Use parallel execution for training")
	optimizeThreads := flag.Bool("optimize-threads", false, "Find optimal thread count for current hardware")
	threads := flag.Int("threads", 0, "Specific number of threads to use (0 = auto)")
	profile := flag.Bool("profile", false, "Enable CPU profiling")

	// Model hyperparameter flags (defaults from constants)
	m1Games := flag.Int("m1-games", model1SelfPlayGames, "Self-play games for Model 1")
	m1Epochs := flag.Int("m1-epochs", model1Epochs, "Training epochs for Model 1")
	m1Hidden := flag.Int("m1-hidden", model1HiddenSize, "Hidden neurons for Model 1")
	m1Sims := flag.Int("m1-sims", mctsSimulations*3/2, "MCTS simulations for Model 1")
	m1Exploration := flag.Float64("m1-exploration", 1.5, "Exploration constant for Model 1")

	m2Games := flag.Int("m2-games", model2SelfPlayGames, "Self-play games for Model 2")
	m2Epochs := flag.Int("m2-epochs", model2Epochs, "Training epochs for Model 2")
	m2Hidden := flag.Int("m2-hidden", model2HiddenSize, "Hidden neurons for Model 2")
	m2Sims := flag.Int("m2-sims", mctsSimulations, "MCTS simulations for Model 2")
	m2Exploration := flag.Float64("m2-exploration", 1.0, "Exploration constant for Model 2")

	tourGames := flag.Int("tournament-games", tournamentGames, "Number of head-to-head games")
	flag.Parse()

	// Setup CPU profiling if requested
	if *profile {
		// Ensure directory exists
		os.MkdirAll("output/profiles", 0755)

		// Create profile file
		timestamp := time.Now().Format("20060102-150405")
		profilePath := fmt.Sprintf("output/profiles/cpu_%s.prof", timestamp)
		f, err := os.Create(profilePath)
		if err != nil {
			log.Fatalf("Could not create CPU profile: %v", err)
		}

		fmt.Printf("CPU profiling enabled. Profile will be written to %s\n", profilePath)
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("Could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Handle thread optimization if requested
	if *optimizeThreads {
		findOptimalThreadCount()
		return
	}

	// Read hyperparameters from flags
	m1G := *m1Games
	m1E := *m1Epochs
	h1 := *m1Hidden
	s1 := *m1Sims
	x1 := *m1Exploration

	m2G := *m2Games
	m2E := *m2Epochs
	h2 := *m2Hidden
	s2 := *m2Sims
	x2 := *m2Exploration

	tG := *tourGames

	// Handle small-run override
	if *smallRun {
		fmt.Println("Running in small test mode with reduced parameters")
		m1G = 10
		m1E = 3
		m2G = 20
		m2E = 6
		tG = 50
	}

	if *parallel {
		fmt.Println("Using parallel execution for faster training")
	}

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create output directory if it doesn't exist
	os.Mkdir("output", 0755)

	// Initialize neural networks for model 1 (smaller network, fewer games)
	fmt.Println("=== Training Model 1 (Small Network) ===")
	policy1, value1 := trainModel("output/rps_policy1.model", "output/rps_value1.model",
		m1G, m1E, h1, *parallel, *threads)

	// Initialize neural networks for model 2 (larger network, more games)
	fmt.Println("\n=== Training Model 2 (Large Network) ===")
	policy2, value2 := trainModel("output/rps_policy2.model", "output/rps_value2.model",
		m2G, m2E, h2, *parallel, *threads)

	model1Name := fmt.Sprintf("H%d-G%d-E%d-S%d-X%.1f",
		h1, m1G, m1E, s1, x1)

	model2Name := fmt.Sprintf("H%d-G%d-E%d-S%d-X%.1f",
		h2, m2G, m2E, s2, x2)

	// Create agents for tournament with different MCTS parameters
	// Give the smaller model more simulations to compensate for less training
	// Model 1: More search but less neural network knowledge (more exploration)
	// Model 2: Less search but more neural network knowledge (more exploitation)
	agent1 := NewCustomAlphaGoAgent(model1Name, policy1, value1,
		s1, x1)

	agent2 := NewCustomAlphaGoAgent(model2Name, policy2, value2,
		s2, x2)

	// Display model comparison information
	fmt.Println("\n=== Model Comparison ===")
	fmt.Printf("Model 1: %s\n", agent1.Name())
	fmt.Printf("  Self-play games: %d\n", m1G)
	fmt.Printf("  Training epochs: %d\n", m1E)
	fmt.Printf("  MCTS simulations: %d\n", s1)
	fmt.Printf("  Exploration constant: %.1f\n", x1)
	stats1Policy := neural.CalculatePolicyNetworkStats(policy1)
	stats1Value := neural.CalculateValueNetworkStats(value1)
	totalParams1 := stats1Policy.TotalParameters + stats1Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", h1)
	fmt.Printf("  Total parameters: %d\n", totalParams1)

	fmt.Printf("\nModel 2: %s\n", agent2.Name())
	fmt.Printf("  Self-play games: %d\n", m2G)
	fmt.Printf("  Training epochs: %d\n", m2E)
	fmt.Printf("  MCTS simulations: %d\n", s2)
	fmt.Printf("  Exploration constant: %.1f\n", x2)
	stats2Policy := neural.CalculatePolicyNetworkStats(policy2)
	stats2Value := neural.CalculateValueNetworkStats(value2)
	totalParams2 := stats2Policy.TotalParameters + stats2Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", h2)
	fmt.Printf("  Total parameters: %d\n", totalParams2)

	paramRatio := float64(totalParams2) / float64(totalParams1)
	gameRatio := float64(m2G) / float64(m1G)
	fmt.Printf("\nModel 2 has %.1fx more parameters and %.1fx more training games than Model 1\n",
		paramRatio, gameRatio)
	fmt.Printf("Model 1 has %.1fx more MCTS simulations and %.1fx higher exploration constant than Model 2\n",
		1.5, 1.5)
	fmt.Printf("This sets up a classic quality vs. quantity tradeoff:\n")
	fmt.Printf("- Model 1: Weaker neural network but more search\n")
	fmt.Printf("- Model 2: Stronger neural network but less search\n")

	// Run tournament
	fmt.Println("\n=== Starting Tournament (Model 1 vs Model 2) ===")
	model1Wins, model2Wins, draws := runTournament(agent1, agent2, tG)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", tG)
	fmt.Printf("Model 1 (%s) wins: %d (%.1f%%)\n", agent1.Name(), model1Wins, float64(model1Wins)/float64(tG)*100)
	fmt.Printf("Model 2 (%s) wins: %d (%.1f%%)\n", agent2.Name(), model2Wins, float64(model2Wins)/float64(tG)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(tG)*100)

	// Calculate statistical significance
	winDiff := math.Abs(float64(model1Wins) - float64(model2Wins))
	pValue := calculatePValue(model1Wins, model2Wins, tG)
	fmt.Printf("\nWin difference: %.1f%%\n", winDiff/float64(tG)*100)
	fmt.Printf("Statistical significance: p-value %.3f ", pValue)

	if pValue < 0.05 {
		fmt.Println("(statistically significant)")
	} else {
		fmt.Println("(not statistically significant)")
	}

	model1Desc := "Small network with more search"
	model2Desc := "Large network with less search"

	if model2Wins > model1Wins {
		fmt.Printf("\nModel 2 (%s) outperformed Model 1!\n", model2Desc)
		fmt.Println("Neural network quality appears more important than search quantity.")
	} else if model1Wins > model2Wins {
		fmt.Printf("\nModel 1 (%s) outperformed Model 2!\n", model1Desc)
		fmt.Println("Search quantity appears more important than neural network quality.")
	} else {
		fmt.Println("\nThe models performed equally!")
		fmt.Println("The tradeoff between neural network quality and search quantity is balanced.")
	}
}

// calculatePValue calculates a simple p-value for the win difference
func calculatePValue(wins1, wins2, total int) float64 {
	// Using binomial distribution to test if win rate is different from 0.5
	// This is a simplification, but gives a rough idea of statistical significance
	observed := math.Abs(float64(wins1) - float64(wins2))

	// Simple approximation using normal distribution for large samples
	stdDev := math.Sqrt(float64(total) * 0.5 * 0.5)
	z := observed / stdDev

	// Calculate two-tailed p-value (simplified)
	return 2 * (1 - math.Erf(z/math.Sqrt(2)))
}

// trainModel trains a policy and value network with self-play
func trainModel(policyPath, valuePath string, selfPlayGames, epochs, hiddenSize int, forceParallel bool, threads int) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork) {
	// Get timestamp for model naming
	timestamp := time.Now().Format("20060102-150405")

	// Create descriptive model names
	modelName := fmt.Sprintf("rps_h%d_g%d_e%d_%s", hiddenSize, selfPlayGames, epochs, timestamp)
	policyPath = fmt.Sprintf("output/%s_policy.model", modelName)
	valuePath = fmt.Sprintf("output/%s_value.model", modelName)

	// Initialize neural networks with specified hidden size
	policyNetwork := neural.NewRPSPolicyNetwork(hiddenSize)
	valueNetwork := neural.NewRPSValueNetwork(hiddenSize)

	// Display network complexity information
	fmt.Println("\n--- Network Architecture Details ---")
	neural.DisplayNetworkComplexity(policyNetwork, valueNetwork)
	fmt.Println("")

	// Create self-play parameters
	selfPlayParams := training.DefaultRPSSelfPlayParams()
	selfPlayParams.NumGames = selfPlayGames
	selfPlayParams.DeckSize = deckSize
	selfPlayParams.HandSize = handSize
	selfPlayParams.MaxRounds = maxRounds
	selfPlayParams.NumThreads = threads

	// Force parallel execution if requested
	if forceParallel {
		// Set a minimum game count to ensure parallel execution
		if selfPlayParams.NumGames < 5 {
			fmt.Println("Warning: Game count too low for effective parallelization, increasing to 5")
			selfPlayParams.NumGames = 5
		}
		selfPlayParams.ForceParallel = true
		fmt.Println("Forced parallel execution enabled")

		// Print thread information
		if threads > 0 {
			fmt.Printf("Using %d worker threads as specified\n", threads)
		} else {
			fmt.Printf("Using auto thread selection (up to %d workers)\n", runtime.NumCPU()-1)
		}
	}

	// Print MCTS simulation parameters
	fmt.Printf("MCTS Parameters: %d simulations per move\n", selfPlayParams.MCTSParams.NumSimulations)
	fmt.Printf("Exploration constant: %.2f\n", selfPlayParams.MCTSParams.ExplorationConst)

	// Create self-play instance
	selfPlay := training.NewRPSSelfPlay(policyNetwork, valueNetwork, selfPlayParams)

	// Generate training examples through self-play
	fmt.Printf("\n--- Self-Play Phase ---\n")
	fmt.Printf("Generating %d self-play games with %d cards per player (%d max rounds)...\n",
		selfPlayGames, handSize, maxRounds)
	startTime := time.Now()
	examples := selfPlay.GenerateGames(true) // Enable verbose mode for more updates
	genTime := time.Since(startTime)

	// Calculate examples per game
	examplesPerGame := float64(len(examples)) / float64(selfPlayGames)
	gamesPerSecond := float64(selfPlayGames) / genTime.Seconds()

	fmt.Printf("Generated %d training examples in %s (%.1f examples/game, %.2f games/sec)\n",
		len(examples), genTime, examplesPerGame, gamesPerSecond)

	// Train networks with adjusted learning rate for larger networks
	fmt.Printf("\n--- Training Phase ---\n")

	// Use lower learning rate for larger networks to prevent instability
	baseLR := 0.01
	learningRate := baseLR
	if hiddenSize >= 100 {
		learningRate = baseLR * 0.5
		fmt.Printf("Using reduced learning rate (%.4f) for large network\n", learningRate)
	} else {
		fmt.Printf("Using standard learning rate (%.4f)\n", learningRate)
	}

	fmt.Printf("Training networks for %d epochs (Batch size: %d)...\n",
		epochs, 32)
	startTime = time.Now()
	policyLosses, valueLosses := selfPlay.TrainNetworks(epochs, 32, learningRate, true)
	trainTime := time.Since(startTime)

	// Calculate training speed
	examplesPerSecond := float64(len(examples)*epochs) / trainTime.Seconds()
	fmt.Printf("Training completed in %s (%.2f examples/sec)\n", trainTime, examplesPerSecond)

	// Display final losses if available
	if len(policyLosses) > 0 && len(valueLosses) > 0 {
		finalPolicyLoss := policyLosses[len(policyLosses)-1]
		finalValueLoss := valueLosses[len(valueLosses)-1]
		fmt.Printf("Final losses - Policy: %.4f, Value: %.4f\n", finalPolicyLoss, finalValueLoss)

		// Calculate total improvement
		if len(policyLosses) > 1 {
			policyImprovement := (policyLosses[0] - finalPolicyLoss) / policyLosses[0] * 100
			valueImprovement := (valueLosses[0] - finalValueLoss) / valueLosses[0] * 100
			fmt.Printf("Total improvement - Policy: %.1f%%, Value: %.1f%%\n",
				policyImprovement, valueImprovement)
		}
	}

	// Save the trained models
	fmt.Printf("\n--- Saving Models ---\n")
	err := policyNetwork.SaveToFile(policyPath)
	if err != nil {
		log.Fatalf("Failed to save policy network: %v", err)
	}
	err = valueNetwork.SaveToFile(valuePath)
	if err != nil {
		log.Fatalf("Failed to save value network: %v", err)
	}
	fmt.Printf("Models saved to %s and %s\n", policyPath, valuePath)

	return policyNetwork, valueNetwork
}

// AlphaGoAgent wraps the AlphaGo-style MCTS + neural network agent
type AlphaGoAgent struct {
	name          string
	policyNetwork *neural.RPSPolicyNetwork
	valueNetwork  *neural.RPSValueNetwork
	mctsEngine    *mcts.RPSMCTS
}

func NewAlphaGoAgent(name string, policyNet *neural.RPSPolicyNetwork, valueNet *neural.RPSValueNetwork) *AlphaGoAgent {
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = mctsSimulations // Adjust as needed

	return &AlphaGoAgent{
		name:          name,
		policyNetwork: policyNet,
		valueNetwork:  valueNet,
		mctsEngine:    mcts.NewRPSMCTS(policyNet, valueNet, mctsParams),
	}
}

// Create an agent with custom MCTS parameters
func NewCustomAlphaGoAgent(name string, policyNet *neural.RPSPolicyNetwork, valueNet *neural.RPSValueNetwork,
	simulations int, explorationConst float64) *AlphaGoAgent {

	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = simulations
	mctsParams.ExplorationConst = explorationConst

	return &AlphaGoAgent{
		name:          name,
		policyNetwork: policyNet,
		valueNetwork:  valueNet,
		mctsEngine:    mcts.NewRPSMCTS(policyNet, valueNet, mctsParams),
	}
}

func (a *AlphaGoAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	// Use MCTS to find the best move
	a.mctsEngine.SetRootState(state)
	bestNode := a.mctsEngine.Search()

	if bestNode == nil || bestNode.Move == nil {
		// Fallback to random move if MCTS fails
		validMoves := state.GetValidMoves()
		if len(validMoves) == 0 {
			return game.RPSMove{}, fmt.Errorf("no valid moves")
		}
		return validMoves[rand.Intn(len(validMoves))], nil
	}

	return *bestNode.Move, nil
}

func (a *AlphaGoAgent) Name() string {
	return a.name
}

// runTournament runs a tournament between two agents
func runTournament(agent1, agent2 *AlphaGoAgent, numGames int) (agent1Wins, agent2Wins, draws int) {
	fmt.Println("\nDetailed tournament results:")
	fmt.Println("----------------------------")

	// Get short names for display
	agent1ShortName := "Model1"
	agent2ShortName := "Model2"

	// Track win streaks for analysis
	currentStreak := 0
	maxStreak := 0
	streakHolder := ""

	// Track position-based wins
	positionWins := make(map[int]map[string]int)
	for pos := 0; pos < 9; pos++ {
		positionWins[pos] = make(map[string]int)
	}

	for i := 0; i < numGames; i++ {
		// Print progress
		if (i+1)%10 == 0 || i == 0 {
			fmt.Printf("Playing game %d of %d...\n", i+1, numGames)
		}

		// Create a new game
		gameInstance := game.NewRPSGame(deckSize, handSize, maxRounds)

		// Alternate who goes first to ensure fairness
		var player1Agent, player2Agent *AlphaGoAgent
		if i%2 == 0 {
			player1Agent = agent1
			player2Agent = agent2
		} else {
			player1Agent = agent2
			player2Agent = agent1
		}

		// Track decisive moves for analysis
		decisiveMoves := make([]game.RPSMove, 0)
		moveCount := 0

		// Play the game
		for !gameInstance.IsGameOver() {
			var currentAgent *AlphaGoAgent
			if gameInstance.CurrentPlayer == game.Player1 {
				currentAgent = player1Agent
			} else {
				currentAgent = player2Agent
			}

			move, err := currentAgent.GetMove(gameInstance.Copy())
			if err != nil {
				log.Fatalf("Agent %s failed to make a move: %v", currentAgent.Name(), err)
			}

			move.Player = gameInstance.CurrentPlayer
			err = gameInstance.MakeMove(move)
			if err != nil {
				log.Fatalf("Invalid move from agent %s: %v", currentAgent.Name(), err)
			}

			// Track moves
			moveCount++
			decisiveMoves = append(decisiveMoves, move)
		}

		// Determine winner
		winner := gameInstance.GetWinner()
		var winnerAgent *AlphaGoAgent
		var winnerName string

		if winner == game.Player1 {
			if player1Agent == agent1 {
				agent1Wins++
				winnerAgent = agent1
				winnerName = agent1.Name()
			} else {
				agent2Wins++
				winnerAgent = agent2
				winnerName = agent2.Name()
			}
		} else if winner == game.Player2 {
			if player2Agent == agent1 {
				agent1Wins++
				winnerAgent = agent1
				winnerName = agent1.Name()
			} else {
				agent2Wins++
				winnerAgent = agent2
				winnerName = agent2.Name()
			}
		} else {
			draws++
			winnerName = "Draw"
			winnerAgent = nil
		}

		// Update win streak analysis
		if winnerAgent != nil {
			if streakHolder == winnerName {
				currentStreak++
				if currentStreak > maxStreak {
					maxStreak = currentStreak
				}
			} else {
				streakHolder = winnerName
				currentStreak = 1
			}

			// Record position-based wins
			if len(decisiveMoves) > 0 {
				lastMove := decisiveMoves[len(decisiveMoves)-1]
				positionWins[lastMove.Position][winnerName]++
			}
		} else {
			// Reset streak on draw
			currentStreak = 0
			streakHolder = ""
		}

		// Print result for every 10th game or the final game
		if (i+1)%10 == 0 || i == numGames-1 {
			fmt.Printf("Game %d result: %s (moves: %d)\n", i+1, winnerName, moveCount)
		}
	}

	// Print analysis
	fmt.Printf("\nMax win streak: %d games by %s\n", maxStreak, streakHolder)

	// Analyze position-based winning rates
	fmt.Println("\nPosition-based winning rates:")
	for pos := 0; pos < 9; pos++ {
		row := pos / 3
		col := pos % 3
		a1Wins := positionWins[pos][agent1.Name()]
		a2Wins := positionWins[pos][agent2.Name()]
		fmt.Printf("Position (%d,%d): %s: %d wins, %s: %d wins\n",
			row, col, agent1ShortName, a1Wins, agent2ShortName, a2Wins)
	}

	return agent1Wins, agent2Wins, draws
}

// findOptimalThreadCount determines the optimal number of threads for the current hardware
func findOptimalThreadCount() {
	fmt.Println("Finding optimal thread count for your hardware...")
	fmt.Printf("CPU cores available: %d\n", runtime.NumCPU())

	// Create test parameters
	const testGames = 50
	const hiddenSize = 64
	const maxWorkers = 32 // Don't test beyond 32 threads

	// Create output file
	resultsFile, err := os.Create("output/thread_optimization.txt")
	if err != nil {
		log.Fatalf("Failed to create results file: %v", err)
	}
	defer resultsFile.Close()

	// Write header
	fmt.Fprintf(resultsFile, "Thread Count,Games Per Second,Speedup Factor\n")

	// Initialize base networks for testing
	policyNetwork := neural.NewRPSPolicyNetwork(hiddenSize)
	valueNetwork := neural.NewRPSValueNetwork(hiddenSize)

	// Run tests with varying thread counts
	var baselineSpeed float64
	var bestThreads int
	var bestSpeed float64

	// Test thread counts from 1 to min(maxWorkers, numCPU*2)
	maxTestThreads := runtime.NumCPU() * 2
	if maxTestThreads > maxWorkers {
		maxTestThreads = maxWorkers
	}

	fmt.Println("\nTesting performance with different thread counts:")
	fmt.Println("------------------------------------------------")

	for threads := 1; threads <= maxTestThreads; threads++ {
		// Configure parameters
		selfPlayParams := training.DefaultRPSSelfPlayParams()
		selfPlayParams.NumGames = testGames
		selfPlayParams.ForceParallel = threads > 1

		// Set thread count in global runtime
		originalMaxProcs := runtime.GOMAXPROCS(0)
		runtime.GOMAXPROCS(threads)

		// Create self-play instance
		selfPlay := training.NewRPSSelfPlay(policyNetwork, valueNetwork, selfPlayParams)

		// Measure performance
		start := time.Now()
		selfPlay.GenerateGames(false)
		elapsed := time.Since(start)

		// Calculate metrics
		gamesPerSecond := float64(testGames) / elapsed.Seconds()
		speedupFactor := 1.0

		if threads == 1 {
			baselineSpeed = gamesPerSecond
		} else {
			speedupFactor = gamesPerSecond / baselineSpeed
		}

		// Update best if applicable
		if gamesPerSecond > bestSpeed {
			bestSpeed = gamesPerSecond
			bestThreads = threads
		}

		// Print and save results
		fmt.Printf("Threads: %2d | Games/sec: %6.2f | Speedup: %5.2fx\n",
			threads, gamesPerSecond, speedupFactor)
		fmt.Fprintf(resultsFile, "%d,%.2f,%.2f\n",
			threads, gamesPerSecond, speedupFactor)

		// Reset max procs to original value
		runtime.GOMAXPROCS(originalMaxProcs)
	}

	// Print recommendation
	fmt.Println("\nResults:")
	fmt.Printf("Optimal thread count for your hardware: %d threads\n", bestThreads)
	fmt.Printf("Peak performance: %.2f games/second\n", bestSpeed)
	fmt.Printf("Maximum speedup: %.2fx over single-threaded execution\n", bestSpeed/baselineSpeed)

	if bestThreads == runtime.NumCPU() {
		fmt.Println("Recommendation: Use default parallel execution for optimal performance")
	} else if bestThreads < runtime.NumCPU() {
		fmt.Printf("Recommendation: Use %d threads for optimal performance (fewer than CPU cores)\n", bestThreads)
	} else {
		fmt.Printf("Recommendation: Use %d threads for optimal performance (more than CPU cores)\n", bestThreads)
	}

	fmt.Printf("\nDetailed results saved to output/thread_optimization.txt\n")
}
