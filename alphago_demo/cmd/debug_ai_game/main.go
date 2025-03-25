package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

// Global parameters - reduced for quick testing
const (
	// Game parameters
	deckSize  = 21 // 7 of each card type
	handSize  = 5
	maxRounds = 10

	// Neural network parameters
	hiddenSize = 64

	// Training parameters - minimized for quick testing
	trainNetworks  = true
	selfPlayGames  = 5  // Reduced
	trainingEpochs = 2  // Reduced
	batchSize      = 16 // Reduced
	learningRate   = 0.01
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Println("RPS Card Game Debug - AI vs AI with Winner Determination Check")
	fmt.Println("===========================================================")

	// Initialize neural networks
	fmt.Println("Initializing neural networks...")
	policyNetwork := neural.NewRPSPolicyNetwork(hiddenSize)
	valueNetwork := neural.NewRPSValueNetwork(hiddenSize)

	if trainNetworks {
		// Train networks through self-play
		fmt.Println("Doing minimal training through self-play...")

		// Create self-play parameters
		selfPlayParams := training.DefaultRPSSelfPlayParams()
		selfPlayParams.NumGames = selfPlayGames
		selfPlayParams.DeckSize = deckSize
		selfPlayParams.HandSize = handSize
		selfPlayParams.MaxRounds = maxRounds
		selfPlayParams.MCTSParams.NumSimulations = 50 // Reduced for speed

		// Create self-play instance
		selfPlay := training.NewRPSSelfPlay(policyNetwork, valueNetwork, selfPlayParams)

		// Generate training examples through self-play
		fmt.Printf("Generating %d self-play games...\n", selfPlayGames)
		examples := selfPlay.GenerateGames(false)
		fmt.Printf("Generated %d training examples.\n", len(examples))

		// Train networks
		fmt.Printf("Training networks for %d epochs...\n", trainingEpochs)
		selfPlay.TrainNetworks(trainingEpochs, batchSize, learningRate, true)
		fmt.Println("Training complete.")
	}

	// Simulate AI vs AI game
	fmt.Println("\nRunning AI vs AI Debug Game...")
	runAIvsAIGame(policyNetwork, valueNetwork)

	fmt.Println("\nDebug run complete.")
}

func runAIvsAIGame(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) {
	// Create a new game
	gameInstance := game.NewRPSGame(deckSize, handSize, maxRounds)

	// Create MCTS for AI
	mctsParams := mcts.DefaultRPSMCTSParams()
	// Reduce simulation count for faster demo
	mctsParams.NumSimulations = 100
	mctsEngine := mcts.NewRPSMCTS(policyNetwork, valueNetwork, mctsParams)

	// Game loop
	fmt.Println("Initial state:")
	fmt.Println(gameInstance.String())

	moveCount := 0
	for !gameInstance.IsGameOver() {
		fmt.Printf("\nMove %d\n", moveCount+1)

		// Use MCTS to find the best move
		mctsEngine.SetRootState(gameInstance)
		bestNode := mctsEngine.Search()

		if bestNode != nil && bestNode.Move != nil {
			// Make the move
			move := *bestNode.Move
			err := gameInstance.MakeMove(move)
			if err != nil {
				fmt.Printf("Error making move: %v\n", err)
				return
			}

			// Display the move
			playerName := "Player 1"
			if move.Player == game.Player2 {
				playerName = "Player 2"
			}

			cardType := gameInstance.Board[move.Position].Type
			var cardTypeStr string
			switch cardType {
			case game.Rock:
				cardTypeStr = "Rock"
			case game.Paper:
				cardTypeStr = "Paper"
			case game.Scissors:
				cardTypeStr = "Scissors"
			}

			row := move.Position / 3
			col := move.Position % 3
			fmt.Printf("%s plays %s at position (%d,%d)\n", playerName, cardTypeStr, row, col)
		} else {
			// Use random move if MCTS fails
			randomMove, err := gameInstance.GetRandomMove()
			if err != nil {
				fmt.Printf("No valid moves: %v\n", err)
				break
			}

			err = gameInstance.MakeMove(randomMove)
			if err != nil {
				fmt.Printf("Error making random move: %v\n", err)
				return
			}

			// Display the move
			playerName := "Player 1"
			if randomMove.Player == game.Player2 {
				playerName = "Player 2"
			}

			cardType := gameInstance.Board[randomMove.Position].Type
			var cardTypeStr string
			switch cardType {
			case game.Rock:
				cardTypeStr = "Rock"
			case game.Paper:
				cardTypeStr = "Paper"
			case game.Scissors:
				cardTypeStr = "Scissors"
			}

			row := randomMove.Position / 3
			col := randomMove.Position % 3
			fmt.Printf("%s plays %s at position (%d,%d) (random)\n", playerName, cardTypeStr, row, col)
		}

		// Display game state after each move
		fmt.Println(gameInstance.String())

		moveCount++

		// Add a short pause to make it easier to follow
		time.Sleep(1 * time.Second)
	}

	// Game over
	fmt.Println("\n*** GAME OVER ***")
	fmt.Println(gameInstance.String())

	// Manual card count for verification
	var player1Count, player2Count int
	for _, card := range gameInstance.Board {
		if card.Owner == game.Player1 {
			player1Count++
		} else if card.Owner == game.Player2 {
			player2Count++
		}
	}

	fmt.Printf("\nFinal Manual Card Count Verification:\n")
	fmt.Printf("Player 1: %d cards\n", player1Count)
	fmt.Printf("Player 2: %d cards\n", player2Count)

	winner := gameInstance.GetWinner()
	fmt.Printf("\nWinner Determination Logic Result: ")
	if winner == game.NoPlayer {
		fmt.Println("Draw")
	} else if winner == game.Player1 {
		fmt.Println("Player 1 wins")
	} else {
		fmt.Println("Player 2 wins")
	}

	// Verify winner determination is correct
	if player1Count > player2Count && winner != game.Player1 {
		fmt.Printf("ERROR: Player 1 has more cards but is not declared the winner!\n")
	} else if player2Count > player1Count && winner != game.Player2 {
		fmt.Printf("ERROR: Player 2 has more cards but is not declared the winner!\n")
	} else if player1Count == player2Count && winner != game.NoPlayer {
		fmt.Printf("ERROR: Card count is tied but a winner was declared!\n")
	} else {
		fmt.Println("Winner determination is correct!")
	}
}
