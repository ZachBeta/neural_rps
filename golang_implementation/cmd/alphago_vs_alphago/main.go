package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/pkg/agent"
	"github.com/zachbeta/neural_rps/pkg/game"
)

// This program runs a tournament between two AlphaGo agents with different configurations.
// It uses neural networks trained in the alphago_demo package and loaded here.
//
// Integration with alphago_demo:
// - The neural networks (policy and value) are defined in alphago_demo/pkg/neural
// - The networks are trained using alphago_demo/pkg/training
// - The trained networks are saved in alphago_demo/output
// - This program loads those networks and uses them with the AlphaGoAgent adapter
//
// The AlphaGoAgent adapter in pkg/agent/alphago_agent.go converts between the
// golang_implementation game state and the alphago_demo game state, allowing
// for the AlphaGo networks to be used with the tournament infrastructure here.

const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10
)

func main() {
	// Parse command-line flags
	numGames := flag.Int("games", 50, "Number of tournament games to play")
	smallPolicyPath := flag.String("small-policy", "../alphago_demo/output/rps_policy1.model", "Path to small AlphaGo policy network model")
	smallValuePath := flag.String("small-value", "../alphago_demo/output/rps_value1.model", "Path to small AlphaGo value network model")
	largePolicyPath := flag.String("large-policy", "../alphago_demo/output/rps_policy2.model", "Path to large AlphaGo policy network model")
	largeValuePath := flag.String("large-value", "../alphago_demo/output/rps_value2.model", "Path to large AlphaGo value network model")
	smallSims := flag.Int("small-sims", 100, "Number of MCTS simulations for small AlphaGo agent")
	largeSims := flag.Int("large-sims", 300, "Number of MCTS simulations for large AlphaGo agent")
	smallExploration := flag.Float64("small-exploration", 1.0, "Exploration constant for small AlphaGo agent")
	largeExploration := flag.Float64("large-exploration", 1.5, "Exploration constant for large AlphaGo agent")
	verbose := flag.Bool("verbose", false, "Show detailed game information")
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== AlphaGo Small vs AlphaGo Large Tournament ===")
	fmt.Printf("Playing %d games of RPS Card Game\n", *numGames)

	// Create output directory if it doesn't exist
	os.MkdirAll("results", 0755)

	// Load AlphaGo networks
	fmt.Println("\nLoading AlphaGo Small networks...")
	smallPolicyNet, smallValueNet, err := agent.LoadAlphaGoNetworksFromFile(*smallPolicyPath, *smallValuePath)
	if err != nil {
		log.Fatalf("Failed to load AlphaGo Small networks: %v", err)
	}

	fmt.Println("Loading AlphaGo Large networks...")
	largePolicyNet, largeValueNet, err := agent.LoadAlphaGoNetworksFromFile(*largePolicyPath, *largeValuePath)
	if err != nil {
		log.Fatalf("Failed to load AlphaGo Large networks: %v", err)
	}

	// Create agents
	smallAgent := agent.NewAlphaGoAgent("AlphaGo-Small", smallPolicyNet, smallValueNet, *smallSims, *smallExploration)
	largeAgent := agent.NewAlphaGoAgent("AlphaGo-Large", largePolicyNet, largeValueNet, *largeSims, *largeExploration)

	// Display agent information
	fmt.Println("\n=== Agent Information ===")
	fmt.Printf("AlphaGo Small Agent:\n")
	fmt.Printf("  Policy Network: %d-%d-%d\n", 81, smallPolicyNet.GetHiddenSize(), 9)
	fmt.Printf("  Value Network: %d-%d-%d\n", 81, smallValueNet.GetHiddenSize(), 1)
	fmt.Printf("  MCTS Simulations: %d\n", *smallSims)
	fmt.Printf("  Exploration Constant: %.2f\n", *smallExploration)

	fmt.Printf("\nAlphaGo Large Agent:\n")
	fmt.Printf("  Policy Network: %d-%d-%d\n", 81, largePolicyNet.GetHiddenSize(), 9)
	fmt.Printf("  Value Network: %d-%d-%d\n", 81, largeValueNet.GetHiddenSize(), 1)
	fmt.Printf("  MCTS Simulations: %d\n", *largeSims)
	fmt.Printf("  Exploration Constant: %.2f\n", *largeExploration)

	// Run tournament
	fmt.Println("\n=== Starting Tournament ===")
	smallWins, largeWins, draws := runTournament(smallAgent, largeAgent, *numGames, *verbose)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", *numGames)
	fmt.Printf("AlphaGo Small wins: %d (%.1f%%)\n", smallWins, float64(smallWins)/float64(*numGames)*100)
	fmt.Printf("AlphaGo Large wins: %d (%.1f%%)\n", largeWins, float64(largeWins)/float64(*numGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(*numGames)*100)

	// Save results to file
	resultStr := fmt.Sprintf("Tournament: AlphaGo Small vs AlphaGo Large\nGames: %d\nAlphaGo Small wins: %d (%.1f%%)\nAlphaGo Large wins: %d (%.1f%%)\nDraws: %d (%.1f%%)\n",
		*numGames,
		smallWins, float64(smallWins)/float64(*numGames)*100,
		largeWins, float64(largeWins)/float64(*numGames)*100,
		draws, float64(draws)/float64(*numGames)*100)

	filename := fmt.Sprintf("results/tournament_AlphaGo_Small_vs_Large_%s.txt", time.Now().Format("20060102_150405"))
	err = os.WriteFile(filename, []byte(resultStr), 0644)
	if err != nil {
		log.Printf("Warning: Failed to save results to file: %v", err)
	} else {
		fmt.Printf("Results saved to %s\n", filename)
	}
}

// runTournament runs a tournament between two agents
func runTournament(agent1, agent2 interface{}, numGames int, verbose bool) (agent1Wins, agent2Wins, draws int) {
	// Define agent wrappers to handle GetMove interface differences
	type AgentWrapper interface {
		Name() string
		GetMove(*game.RPSCardGame) (game.RPSCardMove, error)
	}

	// Define move counter
	moveCount := 0

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
		gameInstance := game.NewRPSCardGame(deckSize, handSize, maxRounds)

		// Alternate who goes first to ensure fairness
		var currentAgents [2]AgentWrapper
		if i%2 == 0 {
			currentAgents[0] = agent1.(AgentWrapper)
			currentAgents[1] = agent2.(AgentWrapper)
		} else {
			currentAgents[0] = agent2.(AgentWrapper)
			currentAgents[1] = agent1.(AgentWrapper)
		}

		// Track moves for analysis
		gameMoves := make([]game.RPSCardMove, 0)
		gameMovesCount := 0

		// Play the game
		for !gameInstance.IsGameOver() {
			var currentAgent AgentWrapper
			if gameInstance.CurrentPlayer == game.Player1 {
				currentAgent = currentAgents[0]
			} else {
				currentAgent = currentAgents[1]
			}

			if verbose {
				fmt.Printf("\nTurn: %s (Player %d)\n", currentAgent.Name(), gameInstance.CurrentPlayer+1)
				fmt.Println(gameInstance.String())
			}

			// Get move from current agent
			move, err := currentAgent.GetMove(gameInstance.Copy())
			if err != nil {
				log.Fatalf("Agent %s failed to make a move: %v", currentAgent.Name(), err)
			}

			// Make the move
			move.Player = gameInstance.CurrentPlayer
			err = gameInstance.MakeMove(move)
			if err != nil {
				log.Fatalf("Invalid move from agent %s: %v", currentAgent.Name(), err)
			}

			// Track moves
			gameMoves = append(gameMoves, move)
			gameMovesCount++
			moveCount++

			if verbose {
				fmt.Printf("%s places %v at position %d\n",
					currentAgent.Name(), move.CardIndex, move.Position)
			}
		}

		// Determine winner
		winner := gameInstance.GetWinner()
		var winnerName string

		if winner == game.Player1 {
			if currentAgents[0] == agent1 {
				agent1Wins++
				winnerName = agent1.(AgentWrapper).Name()
			} else {
				agent2Wins++
				winnerName = agent2.(AgentWrapper).Name()
			}
		} else if winner == game.Player2 {
			if currentAgents[1] == agent1 {
				agent1Wins++
				winnerName = agent1.(AgentWrapper).Name()
			} else {
				agent2Wins++
				winnerName = agent2.(AgentWrapper).Name()
			}
		} else {
			draws++
			winnerName = "Draw"
		}

		// Record position-based wins
		if winnerName != "Draw" && len(gameMoves) > 0 {
			lastMove := gameMoves[len(gameMoves)-1]
			positionWins[lastMove.Position][winnerName]++
		}

		// Print result for every game if verbose
		if verbose || (i+1)%10 == 0 {
			fmt.Printf("Game %d result: %s (moves: %d)\n", i+1, winnerName, gameMovesCount)
		}
	}

	// Print analysis of positions
	fmt.Println("\nPosition-based winning rates:")
	for pos := 0; pos < 9; pos++ {
		row := pos / 3
		col := pos % 3
		a1Wins := positionWins[pos][agent1.(AgentWrapper).Name()]
		a2Wins := positionWins[pos][agent2.(AgentWrapper).Name()]
		fmt.Printf("Position (%d,%d): %s: %d wins, %s: %d wins\n",
			row, col, agent1.(AgentWrapper).Name(), a1Wins, agent2.(AgentWrapper).Name(), a2Wins)
	}

	// Calculate average moves per game
	fmt.Printf("\nAverage moves per game: %.1f\n", float64(moveCount)/float64(numGames))

	return agent1Wins, agent2Wins, draws
}
