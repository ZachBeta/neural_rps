package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10

	// MCTS parameters
	mctsSimulations = 200
)

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

func main() {
	// Define command line flags
	model1Policy := flag.String("model1-policy", "output/rps_policy1.model", "Path to model 1 policy network file")
	model1Value := flag.String("model1-value", "output/rps_value1.model", "Path to model 1 value network file")
	model1Name := flag.String("model1-name", "Model1", "Name for model 1")

	model2Policy := flag.String("model2-policy", "output/rps_policy2.model", "Path to model 2 policy network file")
	model2Value := flag.String("model2-value", "output/rps_value2.model", "Path to model 2 value network file")
	model2Name := flag.String("model2-name", "Model2", "Name for model 2")

	numGames := flag.Int("games", 30, "Number of games to play")
	verbose := flag.Bool("verbose", false, "Show each move during games")
	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Load policy networks from files
	policy1 := neural.NewRPSPolicyNetwork(128)
	err := policy1.LoadFromFile(*model1Policy)
	if err != nil {
		log.Fatalf("Failed to load model 1 policy from %s: %v", *model1Policy, err)
	}
	fmt.Printf("Loaded model 1 policy from %s\n", *model1Policy)

	policy2 := neural.NewRPSPolicyNetwork(128)
	err = policy2.LoadFromFile(*model2Policy)
	if err != nil {
		log.Fatalf("Failed to load model 2 policy from %s: %v", *model2Policy, err)
	}
	fmt.Printf("Loaded model 2 policy from %s\n", *model2Policy)

	// Load value networks from files
	value1 := neural.NewRPSValueNetwork(128)
	err = value1.LoadFromFile(*model1Value)
	if err != nil {
		log.Fatalf("Failed to load model 1 value from %s: %v", *model1Value, err)
	}
	fmt.Printf("Loaded model 1 value from %s\n", *model1Value)

	value2 := neural.NewRPSValueNetwork(128)
	err = value2.LoadFromFile(*model2Value)
	if err != nil {
		log.Fatalf("Failed to load model 2 value from %s: %v", *model2Value, err)
	}
	fmt.Printf("Loaded model 2 value from %s\n", *model2Value)

	// Create agents
	agent1 := NewAlphaGoAgent(*model1Name, policy1, value1)
	agent2 := NewAlphaGoAgent(*model2Name, policy2, value2)

	// Display model network complexity comparison
	fmt.Println("\n=== Model Complexity Comparison ===")
	fmt.Printf("Model 1: %s\n", agent1.Name())
	stats1Policy := neural.CalculatePolicyNetworkStats(policy1)
	stats1Value := neural.CalculateValueNetworkStats(value1)
	totalParams1 := stats1Policy.TotalParameters + stats1Value.TotalParameters
	fmt.Printf("  Architecture: %d-%d-%d (policy), %d-%d-%d (value)\n",
		stats1Policy.InputSize, stats1Policy.HiddenSize, stats1Policy.OutputSize,
		stats1Value.InputSize, stats1Value.HiddenSize, stats1Value.OutputSize)
	fmt.Printf("  Total neurons: %d\n", stats1Policy.TotalNeurons+stats1Value.TotalNeurons)
	fmt.Printf("  Total parameters: %d (%.2f KB)\n", totalParams1,
		stats1Policy.MemoryFootprint+stats1Value.MemoryFootprint)

	fmt.Printf("\nModel 2: %s\n", agent2.Name())
	stats2Policy := neural.CalculatePolicyNetworkStats(policy2)
	stats2Value := neural.CalculateValueNetworkStats(value2)
	totalParams2 := stats2Policy.TotalParameters + stats2Value.TotalParameters
	fmt.Printf("  Architecture: %d-%d-%d (policy), %d-%d-%d (value)\n",
		stats2Policy.InputSize, stats2Policy.HiddenSize, stats2Policy.OutputSize,
		stats2Value.InputSize, stats2Value.HiddenSize, stats2Value.OutputSize)
	fmt.Printf("  Total neurons: %d\n", stats2Policy.TotalNeurons+stats2Value.TotalNeurons)
	fmt.Printf("  Total parameters: %d (%.2f KB)\n", totalParams2,
		stats2Policy.MemoryFootprint+stats2Value.MemoryFootprint)

	// Compare sizes
	sizeRatio := float64(totalParams2) / float64(totalParams1)
	fmt.Printf("\nModel size comparison: Model 2 is %.2fx the size of Model 1\n", sizeRatio)
	fmt.Println("===================================")

	// Run tournament
	fmt.Printf("\n=== Starting Tournament (%s vs %s) ===\n", agent1.Name(), agent2.Name())
	model1Wins, model2Wins, draws := runTournament(agent1, agent2, *numGames, *verbose)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", *numGames)
	fmt.Printf("%s wins: %d (%.1f%%)\n", agent1.Name(), model1Wins, float64(model1Wins)/float64(*numGames)*100)
	fmt.Printf("%s wins: %d (%.1f%%)\n", agent2.Name(), model2Wins, float64(model2Wins)/float64(*numGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(*numGames)*100)

	if model2Wins > model1Wins {
		fmt.Printf("\n%s outperformed %s!\n", agent2.Name(), agent1.Name())
	} else if model1Wins > model2Wins {
		fmt.Printf("\n%s outperformed %s!\n", agent1.Name(), agent2.Name())
	} else {
		fmt.Println("\nThe models performed equally!")
	}

	// Save results to file
	resultStr := fmt.Sprintf("Tournament: %s vs %s\nGames: %d\n%s wins: %d (%.1f%%)\n%s wins: %d (%.1f%%)\nDraws: %d (%.1f%%)\n",
		agent1.Name(), agent2.Name(), *numGames,
		agent1.Name(), model1Wins, float64(model1Wins)/float64(*numGames)*100,
		agent2.Name(), model2Wins, float64(model2Wins)/float64(*numGames)*100,
		draws, float64(draws)/float64(*numGames)*100)

	// Create results directory if it doesn't exist
	os.MkdirAll("results", 0755)

	// Write results to file
	filename := fmt.Sprintf("results/tournament_%s_vs_%s_%s.txt", agent1.Name(), agent2.Name(), time.Now().Format("20060102_150405"))
	err = os.WriteFile(filename, []byte(resultStr), 0644)
	if err != nil {
		log.Printf("Warning: Failed to save results to file: %v", err)
	} else {
		fmt.Printf("Results saved to %s\n", filename)
	}
}

// runTournament runs a tournament between two agents
func runTournament(agent1, agent2 *AlphaGoAgent, numGames int, verbose bool) (agent1Wins, agent2Wins, draws int) {
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

			if verbose {
				fmt.Printf("Agent %s plays card %d at position %d\n",
					currentAgent.Name(), move.CardIndex, move.Position)
				fmt.Println(gameInstance.String())
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
