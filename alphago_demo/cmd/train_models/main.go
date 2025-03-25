package main

import (
	"fmt"
	"log"
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
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create output directory if it doesn't exist
	os.Mkdir("output", 0755)

	// Initialize neural networks for model 1 (smaller network, fewer games)
	fmt.Println("=== Training Model 1 (100 Games, Small Network) ===")
	policy1, value1 := trainModel("output/rps_policy1.model", "output/rps_value1.model",
		model1SelfPlayGames, model1Epochs, model1HiddenSize)

	// Initialize neural networks for model 2 (larger network, more games)
	fmt.Println("\n=== Training Model 2 (1000 Games, Large Network) ===")
	policy2, value2 := trainModel("output/rps_policy2.model", "output/rps_value2.model",
		model2SelfPlayGames, model2Epochs, model2HiddenSize)

	// Create agents for tournament
	agent1 := NewAlphaGoAgent("Model1-100Games", policy1, value1)
	agent2 := NewAlphaGoAgent("Model2-1000Games", policy2, value2)

	// Display model comparison information
	fmt.Println("\n=== Model Comparison ===")
	fmt.Printf("Model 1: %s\n", agent1.Name())
	fmt.Printf("  Self-play games: %d\n", model1SelfPlayGames)
	fmt.Printf("  Training epochs: %d\n", model1Epochs)
	stats1Policy := neural.CalculatePolicyNetworkStats(policy1)
	stats1Value := neural.CalculateValueNetworkStats(value1)
	totalParams1 := stats1Policy.TotalParameters + stats1Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", model1HiddenSize)
	fmt.Printf("  Total parameters: %d\n", totalParams1)

	fmt.Printf("\nModel 2: %s\n", agent2.Name())
	fmt.Printf("  Self-play games: %d\n", model2SelfPlayGames)
	fmt.Printf("  Training epochs: %d\n", model2Epochs)
	stats2Policy := neural.CalculatePolicyNetworkStats(policy2)
	stats2Value := neural.CalculateValueNetworkStats(value2)
	totalParams2 := stats2Policy.TotalParameters + stats2Value.TotalParameters
	fmt.Printf("  Hidden size: %d neurons\n", model2HiddenSize)
	fmt.Printf("  Total parameters: %d\n", totalParams2)

	paramRatio := float64(totalParams2) / float64(totalParams1)
	gameRatio := float64(model2SelfPlayGames) / float64(model1SelfPlayGames)
	fmt.Printf("\nModel 2 has %.1fx more parameters and %.1fx more training games than Model 1\n",
		paramRatio, gameRatio)

	// Run tournament
	fmt.Println("\n=== Starting Tournament (Model 1 vs Model 2) ===")
	model1Wins, model2Wins, draws := runTournament(agent1, agent2, tournamentGames)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", tournamentGames)
	fmt.Printf("Model 1 (100 Games, Small) wins: %d (%.1f%%)\n", model1Wins, float64(model1Wins)/float64(tournamentGames)*100)
	fmt.Printf("Model 2 (1000 Games, Large) wins: %d (%.1f%%)\n", model2Wins, float64(model2Wins)/float64(tournamentGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(tournamentGames)*100)

	if model2Wins > model1Wins {
		fmt.Println("\nModel 2 (1000 Games, Large) outperformed Model 1!")
	} else if model1Wins > model2Wins {
		fmt.Println("\nModel 1 (100 Games, Small) outperformed Model 2!")
	} else {
		fmt.Println("\nThe models performed equally!")
	}
}

// trainModel trains a policy and value network with self-play
func trainModel(policyPath, valuePath string, selfPlayGames, epochs, hiddenSize int) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork) {
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

	// Train networks
	fmt.Printf("\n--- Training Phase ---\n")
	fmt.Printf("Training networks for %d epochs (Learning rate: %.4f, Batch size: %d)...\n",
		epochs, 0.01, 32)
	startTime = time.Now()
	selfPlay.TrainNetworks(epochs, 32, 0.01, true)
	trainTime := time.Since(startTime)

	// Calculate training speed
	examplesPerSecond := float64(len(examples)*epochs) / trainTime.Seconds()
	fmt.Printf("Training completed in %s (%.2f examples/sec)\n", trainTime, examplesPerSecond)

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
		}

		// Determine winner
		winner := gameInstance.GetWinner()
		var winnerName string

		if winner == game.Player1 {
			if player1Agent == agent1 {
				agent1Wins++
				winnerName = agent1.Name()
			} else {
				agent2Wins++
				winnerName = agent2.Name()
			}
		} else if winner == game.Player2 {
			if player2Agent == agent1 {
				agent1Wins++
				winnerName = agent1.Name()
			} else {
				agent2Wins++
				winnerName = agent2.Name()
			}
		} else {
			draws++
			winnerName = "Draw"
		}

		// Print result for every 10th game
		if (i+1)%10 == 0 {
			fmt.Printf("Game %d result: %s\n", i+1, winnerName)
		}
	}

	return agent1Wins, agent2Wins, draws
}
