package main

import (
	"fmt"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func main() {
	fmt.Println("RPS Card Game Winner Logic Test")
	fmt.Println("==============================")

	// Create a test game
	g := game.NewRPSGame(21, 5, 10)

	// Set up a specific board configuration similar to what was reported
	// Board:
	//   0 1 2
	// 0 s P R
	// 1 s R R
	// 2 p r P
	// Player1 (uppercase): 5 cards
	// Player2 (lowercase): 4 cards

	// Clear the board first
	for i := range g.Board {
		g.Board[i] = game.RPSCard{Type: game.Rock, Owner: game.NoPlayer}
	}

	// Set up the board
	g.Board[0] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 0,0: s
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 0,1: P
	g.Board[2] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 0,2: R
	g.Board[3] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 1,0: s
	g.Board[4] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 1,1: R
	g.Board[5] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 1,2: R
	g.Board[6] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Position 2,0: p
	g.Board[7] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 2,1: r
	g.Board[8] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 2,2: P

	// Empty hands to simulate end of game
	g.Player1Hand = []game.RPSCard{}
	g.Player2Hand = []game.RPSCard{}

	// Manually set Round to be greater than MaxRounds to ensure game is over
	g.Round = g.MaxRounds + 1

	// Print the game state
	fmt.Println("\nGame State:")
	fmt.Println(g.String())

	// Count cards manually
	var player1Count, player2Count int
	for _, card := range g.Board {
		if card.Owner == game.Player1 {
			player1Count++
		} else if card.Owner == game.Player2 {
			player2Count++
		}
	}

	fmt.Printf("\nManual Card Count:\n")
	fmt.Printf("Player 1: %d cards\n", player1Count)
	fmt.Printf("Player 2: %d cards\n", player2Count)

	// Check if game is over
	fmt.Printf("\nGame Over Check:\n")
	fmt.Printf("Is Game Over: %v\n", g.IsGameOver())
	fmt.Printf("Round: %d/%d\n", g.Round, g.MaxRounds)

	// Get the winner
	winner := g.GetWinner()

	fmt.Printf("\nWinner Determination:\n")
	fmt.Printf("Winner: %v\n", winner)

	if winner == game.Player1 {
		fmt.Println("Player 1 wins")
	} else if winner == game.Player2 {
		fmt.Println("Player 2 wins")
	} else {
		fmt.Println("Draw")
	}

	// Now let's try a board where Player 2 has more cards
	fmt.Println("\n\n=== SECOND TEST ===")
	fmt.Println("Testing with Player 2 having more cards")

	// Clear the board
	for i := range g.Board {
		g.Board[i] = game.RPSCard{Type: game.Rock, Owner: game.NoPlayer}
	}

	// Set up a board where Player 2 has more cards
	// Board:
	//   0 1 2
	// 0 s P r
	// 1 s r r
	// 2 p R P
	// Player1 (uppercase): 3 cards
	// Player2 (lowercase): 6 cards

	g.Board[0] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 0,0: s
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 0,1: P
	g.Board[2] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 0,2: r
	g.Board[3] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 1,0: s
	g.Board[4] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 1,1: r
	g.Board[5] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 1,2: r
	g.Board[6] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Position 2,0: p
	g.Board[7] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 2,1: R
	g.Board[8] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 2,2: P

	// Print the game state
	fmt.Println("\nGame State:")
	fmt.Println(g.String())

	// Count cards manually
	player1Count, player2Count = 0, 0
	for _, card := range g.Board {
		if card.Owner == game.Player1 {
			player1Count++
		} else if card.Owner == game.Player2 {
			player2Count++
		}
	}

	fmt.Printf("\nManual Card Count:\n")
	fmt.Printf("Player 1: %d cards\n", player1Count)
	fmt.Printf("Player 2: %d cards\n", player2Count)

	// Get the winner
	winner = g.GetWinner()

	fmt.Printf("\nWinner Determination:\n")
	fmt.Printf("Winner: %v\n", winner)

	if winner == game.Player1 {
		fmt.Println("Player 1 wins")
	} else if winner == game.Player2 {
		fmt.Println("Player 2 wins")
	} else {
		fmt.Println("Draw")
	}
}
