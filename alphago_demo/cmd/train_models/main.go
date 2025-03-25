package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
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
	flag.Parse()

	// Adjust parameters for small test runs
	m1Games := model1SelfPlayGames
	m1Epochs := model1Epochs
	m2Games := model2SelfPlayGames
	m2Epochs := model2Epochs
	tGames := tournamentGames

	if *smallRun {
		fmt.Println("Running in small test mode with reduced parameters")
		m1Games = 10 // 10% of normal
		m1Epochs = 3 // Fewer epochs
		m2Games = 20 // 2% of normal
		m2Epochs = 6 // Fewer epochs
		tGames = 50  // More tournament games to get better statistical significance
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
		m1Games, m1Epochs, model1HiddenSize, *parallel)

	// Initialize neural networks for model 2 (larger network, more games)
	fmt.Println("\n=== Training Model 2 (Large Network) ===")
	policy2, value2 := trainModel("output/rps_policy2.model", "output/rps_value2.model",
		m2Games, m2Epochs, model2HiddenSize, *parallel)

	// Extract model names from saved paths for agent naming
	model1Name := fmt.Sprintf("H%d-G%d-E%d-S%d-X%.1f",
		model1HiddenSize, m1Games, m1Epochs, mctsSimulations*3/2, 1.5)

	model2Name := fmt.Sprintf("H%d-G%d-E%d-S%d-X%.1f",
		model2HiddenSize, m2Games, m2Epochs, mctsSimulations, 1.0)

	// Create agents for tournament with different MCTS parameters
	// Give the smaller model more simulations to compensate for less training
	// Model 1: More search but less neural network knowledge (more exploration)
	// Model 2: Less search but more neural network knowledge (more exploitation)
	agent1 := NewCustomAlphaGoAgent(model1Name, policy1, value1,
		mctsSimulations*3/2, 1.5) // 50% more simulations, higher exploration

	agent2 := NewCustomAlphaGoAgent(model2Name, policy2, value2,
		mctsSimulations, 1.0) // Standard parameters

	// Display model comparison information
	fmt.Println("\n=== Model Comparison ===")
	fmt.Printf("Model 1: %s\n", agent1.Name())
	fmt.Printf("  Self-play games: %d\n", m1Games)
	fmt.Printf("  Training epochs: %d\n", m1Epochs)
	fmt.Printf("  MCTS simulations: %d\n", mctsSimulations*3/2)
	fmt.Printf("  Exploration constant: %.1f\n", 1.5)
	stats1Policy := neural.CalculatePolicyNetworkStats(policy1)
	stats1Value := neural.CalculateValueNetworkStats(value1)
	totalParams1 := stats1Policy.TotalParameters + stats1Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", model1HiddenSize)
	fmt.Printf("  Total parameters: %d\n", totalParams1)

	fmt.Printf("\nModel 2: %s\n", agent2.Name())
	fmt.Printf("  Self-play games: %d\n", m2Games)
	fmt.Printf("  Training epochs: %d\n", m2Epochs)
	fmt.Printf("  MCTS simulations: %d\n", mctsSimulations)
	fmt.Printf("  Exploration constant: %.1f\n", 1.0)
	stats2Policy := neural.CalculatePolicyNetworkStats(policy2)
	stats2Value := neural.CalculateValueNetworkStats(value2)
	totalParams2 := stats2Policy.TotalParameters + stats2Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", model2HiddenSize)
	fmt.Printf("  Total parameters: %d\n", totalParams2)

	paramRatio := float64(totalParams2) / float64(totalParams1)
	gameRatio := float64(m2Games) / float64(m1Games)
	fmt.Printf("\nModel 2 has %.1fx more parameters and %.1fx more training games than Model 1\n",
		paramRatio, gameRatio)
	fmt.Printf("Model 1 has %.1fx more MCTS simulations and %.1fx higher exploration constant than Model 2\n",
		1.5, 1.5)
	fmt.Printf("This sets up a classic quality vs. quantity tradeoff:\n")
	fmt.Printf("- Model 1: Weaker neural network but more search\n")
	fmt.Printf("- Model 2: Stronger neural network but less search\n")

	// Run tournament
	fmt.Println("\n=== Starting Tournament (Model 1 vs Model 2) ===")
	model1Wins, model2Wins, draws := runTournament(agent1, agent2, tGames)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", tGames)
	fmt.Printf("Model 1 (%s) wins: %d (%.1f%%)\n", agent1.Name(), model1Wins, float64(model1Wins)/float64(tGames)*100)
	fmt.Printf("Model 2 (%s) wins: %d (%.1f%%)\n", agent2.Name(), model2Wins, float64(model2Wins)/float64(tGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(tGames)*100)

	// Calculate statistical significance
	winDiff := math.Abs(float64(model1Wins) - float64(model2Wins))
	pValue := calculatePValue(model1Wins, model2Wins, tGames)
	fmt.Printf("\nWin difference: %.1f%%\n", winDiff/float64(tGames)*100)
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
func trainModel(policyPath, valuePath string, selfPlayGames, epochs, hiddenSize int, forceParallel bool) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork) {
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

	// Force parallel execution if requested
	if forceParallel {
		// Set a minimum game count to ensure parallel execution
		if selfPlayParams.NumGames < 5 {
			fmt.Println("Warning: Game count too low for effective parallelization, increasing to 5")
			selfPlayParams.NumGames = 5
		}
		selfPlayParams.ForceParallel = true
		fmt.Println("Forced parallel execution enabled")
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
