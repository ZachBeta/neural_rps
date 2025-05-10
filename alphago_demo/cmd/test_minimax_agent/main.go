package main

import (
	"fmt"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/agents"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

func main() {
	fmt.Println("=== Minimax Agent Test ===")

	// Create the game
	g := game.NewRPSGame(21, 5, 10)

	// Create minimax agent with different depths for testing
	minimaxAgents := []*agents.MinimaxAgent{
		agents.NewMinimaxAgent("Minimax-3", 3, 1*time.Second, true),
		agents.NewMinimaxAgent("Minimax-5", 5, 2*time.Second, true),
	}

	// Create a neural agent with a newly initialized network
	neuralAgent := neural.NewNeuralAgent("NewNeural", neural.NewRPSPolicyNetwork(64))

	// Set up the initial board with a few cards
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Player 1's Rock at (0,0)
	g.Board[8] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Player 2's Scissors at (2,2)

	// Make sure player 1 has at least one of each card type in hand
	p1Hand := []game.RPSCard{
		{Type: game.Rock, Owner: game.Player1},
		{Type: game.Paper, Owner: game.Player1},
		{Type: game.Scissors, Owner: game.Player1},
		{Type: game.Rock, Owner: game.Player1},
	}
	g.Player1Hand = p1Hand

	// Make sure player 2 has at least one of each card type in hand
	p2Hand := []game.RPSCard{
		{Type: game.Rock, Owner: game.Player2},
		{Type: game.Paper, Owner: game.Player2},
		{Type: game.Scissors, Owner: game.Player2},
		{Type: game.Paper, Owner: game.Player2},
	}
	g.Player2Hand = p2Hand

	// Set current player to Player 1
	g.CurrentPlayer = game.Player1

	// Display the game state
	fmt.Println("\nInitial game state:")
	fmt.Println(g.String())

	// Test each minimax agent
	for _, minimaxAgent := range minimaxAgents {
		fmt.Printf("\nTesting %s...\n", minimaxAgent.Name())

		// Get minimax move
		start := time.Now()
		minimaxMove, err := minimaxAgent.GetMove(g)
		if err != nil {
			fmt.Printf("Error getting minimax move: %v\n", err)
			continue
		}
		elapsed := time.Since(start)

		fmt.Printf("%s recommends move: %v (time: %v)\n",
			minimaxAgent.Name(), minimaxMove, elapsed)

		// Compare with neural prediction
		neuralMove, err := neuralAgent.GetMove(g)
		if err != nil {
			fmt.Printf("Error getting neural move: %v\n", err)
			continue
		}

		// Check if they match
		match := minimaxMove.Position == neuralMove.Position
		fmt.Printf("Neural move: %v\n", neuralMove)
		fmt.Printf("Moves match: %v\n", match)
	}

	// Print minimax agent stats
	for _, agent := range minimaxAgents {
		avgTime, totalPositions, avgPositionsPerMove := agent.GetStats()
		fmt.Printf("\n%s stats:\n", agent.Name())
		fmt.Printf("  Average time per move: %v\n", avgTime)
		fmt.Printf("  Total positions evaluated: %d\n", totalPositions)
		fmt.Printf("  Average positions per move: %.1f\n", avgPositionsPerMove)
	}
}
