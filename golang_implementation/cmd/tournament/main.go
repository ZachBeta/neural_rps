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

const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10
)

func main() {
	// Parse command-line flags
	numGames := flag.Int("games", 50, "Number of tournament games to play")
	policyPath := flag.String("policy", "../alphago_demo/output/rps_policy2.model", "Path to AlphaGo policy network model")
	valuePath := flag.String("value", "../alphago_demo/output/rps_value2.model", "Path to AlphaGo value network model")
	alphagoSims := flag.Int("alphago-sims", 200, "Number of MCTS simulations for AlphaGo agent")
	ppoHidden := flag.Int("ppo-hidden", 128, "Hidden layer size for PPO agent")
	verbose := flag.Bool("verbose", false, "Show detailed game information")
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== AlphaGo vs PPO Tournament ===")
	fmt.Printf("Playing %d games of RPS Card Game\n", *numGames)

	// Create output directory if it doesn't exist
	os.MkdirAll("results", 0755)

	// Load AlphaGo networks
	fmt.Println("\nLoading AlphaGo networks...")
	policyNet, valueNet, err := agent.LoadAlphaGoNetworksFromFile(*policyPath, *valuePath)
	if err != nil {
		log.Fatalf("Failed to load AlphaGo networks: %v", err)
	}

	// Create agents
	alphagoAgent := agent.NewAlphaGoAgent("AlphaGo", policyNet, valueNet, *alphagoSims, 1.0)
	ppoAgent := agent.NewRPSPPOAgent("PPO", *ppoHidden)

	// Display agent information
	fmt.Println("\n=== Agent Information ===")
	fmt.Printf("AlphaGo Agent: %s\n", alphagoAgent.Name())
	fmt.Printf("  Policy Network: %d-%d-%d\n", 81, policyNet.GetHiddenSize(), 9)
	fmt.Printf("  Value Network: %d-%d-%d\n", 81, valueNet.GetHiddenSize(), 1)
	fmt.Printf("  MCTS Simulations: %d\n", *alphagoSims)

	fmt.Printf("\nPPO Agent: %s\n", ppoAgent.Name())
	fmt.Printf("  Network: %d-%d-%d\n", 81, *ppoHidden, 9)

	// Run tournament
	fmt.Println("\n=== Starting Tournament ===")
	alphagoWins, ppoWins, draws := runTournament(alphagoAgent, ppoAgent, *numGames, *verbose)

	// Print results
	fmt.Println("\n=== Tournament Results ===")
	fmt.Printf("Games played: %d\n", *numGames)
	fmt.Printf("AlphaGo wins: %d (%.1f%%)\n", alphagoWins, float64(alphagoWins)/float64(*numGames)*100)
	fmt.Printf("PPO wins: %d (%.1f%%)\n", ppoWins, float64(ppoWins)/float64(*numGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n", draws, float64(draws)/float64(*numGames)*100)

	// Save results to file
	resultStr := fmt.Sprintf("Tournament: AlphaGo vs PPO\nGames: %d\nAlphaGo wins: %d (%.1f%%)\nPPO wins: %d (%.1f%%)\nDraws: %d (%.1f%%)\n",
		*numGames,
		alphagoWins, float64(alphagoWins)/float64(*numGames)*100,
		ppoWins, float64(ppoWins)/float64(*numGames)*100,
		draws, float64(draws)/float64(*numGames)*100)

	filename := fmt.Sprintf("results/tournament_AlphaGo_vs_PPO_%s.txt", time.Now().Format("20060102_150405"))
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
