package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

// Global parameters
const (
	// Game parameters
	deckSize  = 21 // 7 of each card type
	handSize  = 5
	maxRounds = 10

	// Neural network parameters
	hiddenSize = 128

	// Training parameters
	trainNetworks  = true
	selfPlayGames  = 100
	trainingEpochs = 10
	batchSize      = 32
	learningRate   = 0.01
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Println("AlphaGo-Style Rock-Paper-Scissors Card Game")
	fmt.Println("===========================================")

	// Initialize neural networks
	policyNetwork := neural.NewRPSPolicyNetwork(hiddenSize)
	valueNetwork := neural.NewRPSValueNetwork(hiddenSize)

	if trainNetworks {
		// Train networks through self-play
		fmt.Println("Training neural networks through self-play...")

		// Create self-play parameters
		selfPlayParams := training.DefaultRPSSelfPlayParams()
		selfPlayParams.NumGames = selfPlayGames
		selfPlayParams.DeckSize = deckSize
		selfPlayParams.HandSize = handSize
		selfPlayParams.MaxRounds = maxRounds

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

	// Main menu
	for {
		fmt.Println("\nMain Menu:")
		fmt.Println("1. Play against AI")
		fmt.Println("2. Watch AI vs AI demonstration")
		fmt.Println("3. Exit")
		fmt.Print("Select an option: ")

		var choice int
		fmt.Scanln(&choice)

		switch choice {
		case 1:
			playAgainstAI(policyNetwork, valueNetwork)
		case 2:
			aiDemonstration(policyNetwork, valueNetwork)
		case 3:
			fmt.Println("Goodbye!")
			return
		default:
			fmt.Println("Invalid choice. Please try again.")
		}
	}
}

func playAgainstAI(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) {
	fmt.Println("\nPlaying against AI")
	fmt.Println("=================")

	// Create a new game
	gameInstance := game.NewRPSGame(deckSize, handSize, maxRounds)

	// Create MCTS for AI
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsEngine := mcts.NewRPSMCTS(policyNetwork, valueNetwork, mctsParams)

	scanner := bufio.NewScanner(os.Stdin)

	// Game loop
	for !gameInstance.IsGameOver() {
		// Display game state
		fmt.Println("\n" + gameInstance.String())

		if gameInstance.CurrentPlayer == game.Player1 {
			// Human player's turn
			fmt.Println("Your turn:")

			// Get valid moves
			validMoves := gameInstance.GetValidMoves()
			if len(validMoves) == 0 {
				fmt.Println("No valid moves. Skipping turn.")
				break
			}

			// Group moves by card
			movesByCard := make(map[int][]game.RPSMove)
			for _, move := range validMoves {
				movesByCard[move.CardIndex] = append(movesByCard[move.CardIndex], move)
			}

			// Display available cards
			fmt.Println("Available cards:")
			for i, card := range gameInstance.Player1Hand {
				var cardType string
				switch card.Type {
				case game.Rock:
					cardType = "Rock"
				case game.Paper:
					cardType = "Paper"
				case game.Scissors:
					cardType = "Scissors"
				}
				fmt.Printf("%d: %s\n", i, cardType)
			}

			// Get card selection
			var cardIndex int
			for {
				fmt.Print("Select card (0-" + strconv.Itoa(len(gameInstance.Player1Hand)-1) + "): ")
				var input string
				scanner.Scan()
				input = scanner.Text()

				var err error
				cardIndex, err = strconv.Atoi(input)
				if err != nil || cardIndex < 0 || cardIndex >= len(gameInstance.Player1Hand) {
					fmt.Println("Invalid selection. Try again.")
					continue
				}
				break
			}

			// Get valid positions for this card
			var positions []int
			for _, move := range movesByCard[cardIndex] {
				positions = append(positions, move.Position)
			}

			// Display board positions
			fmt.Println("Select a position (row,col):")
			fmt.Println("  0 1 2")
			for row := 0; row < 3; row++ {
				fmt.Print(row, " ")
				for col := 0; col < 3; col++ {
					pos := row*3 + col
					isValid := false
					for _, validPos := range positions {
						if validPos == pos {
							isValid = true
							break
						}
					}

					if isValid {
						fmt.Print("? ")
					} else {
						card := gameInstance.Board[pos]
						if card.Owner == game.NoPlayer {
							fmt.Print(". ")
						} else {
							var symbol string
							switch card.Type {
							case game.Rock:
								symbol = "R"
							case game.Paper:
								symbol = "P"
							case game.Scissors:
								symbol = "S"
							}

							if card.Owner == game.Player1 {
								symbol = strings.ToUpper(symbol)
							} else {
								symbol = strings.ToLower(symbol)
							}

							fmt.Print(symbol + " ")
						}
					}
				}
				fmt.Println()
			}

			// Get position selection
			var row, col int
			for {
				fmt.Print("Enter position as row,col (e.g., 1,2): ")
				var input string
				scanner.Scan()
				input = scanner.Text()

				parts := strings.Split(input, ",")
				if len(parts) != 2 {
					fmt.Println("Invalid format. Use row,col (e.g., 1,2)")
					continue
				}

				var err error
				row, err = strconv.Atoi(strings.TrimSpace(parts[0]))
				if err != nil || row < 0 || row >= 3 {
					fmt.Println("Invalid row. Must be 0-2.")
					continue
				}

				col, err = strconv.Atoi(strings.TrimSpace(parts[1]))
				if err != nil || col < 0 || col >= 3 {
					fmt.Println("Invalid column. Must be 0-2.")
					continue
				}

				position := row*3 + col
				isValid := false
				for _, validPos := range positions {
					if validPos == position {
						isValid = true
						break
					}
				}

				if !isValid {
					fmt.Println("Invalid position. Try again.")
					continue
				}

				break
			}

			// Make the move
			position := row*3 + col
			move := game.RPSMove{
				CardIndex: cardIndex,
				Position:  position,
				Player:    game.Player1,
			}

			err := gameInstance.MakeMove(move)
			if err != nil {
				fmt.Printf("Error making move: %v\n", err)
				return
			}

		} else {
			// AI's turn
			fmt.Println("AI is thinking...")

			// Use MCTS to find the best move
			mctsEngine.SetRootState(gameInstance)
			bestNode := mctsEngine.Search()

			if bestNode != nil && bestNode.Move != nil {
				// Make the move
				err := gameInstance.MakeMove(*bestNode.Move)
				if err != nil {
					fmt.Printf("Error making AI move: %v\n", err)
					return
				}

				// Display the move
				cardType := gameInstance.Board[bestNode.Move.Position].Type
				var cardTypeStr string
				switch cardType {
				case game.Rock:
					cardTypeStr = "Rock"
				case game.Paper:
					cardTypeStr = "Paper"
				case game.Scissors:
					cardTypeStr = "Scissors"
				}

				row := bestNode.Move.Position / 3
				col := bestNode.Move.Position % 3
				fmt.Printf("AI plays %s at position (%d,%d)\n", cardTypeStr, row, col)

			} else {
				// Use random move if MCTS fails
				randomMove, err := gameInstance.GetRandomMove()
				if err != nil {
					fmt.Printf("No valid moves for AI: %v\n", err)
					break
				}

				err = gameInstance.MakeMove(randomMove)
				if err != nil {
					fmt.Printf("Error making random AI move: %v\n", err)
					return
				}

				// Display the move
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
				fmt.Printf("AI plays %s at position (%d,%d)\n", cardTypeStr, row, col)
			}
		}
	}

	// Game over
	fmt.Println("\nGame Over!")
	fmt.Println(gameInstance.String())

	winner := gameInstance.GetWinner()
	if winner == game.NoPlayer {
		fmt.Println("It's a draw!")
	} else if winner == game.Player1 {
		fmt.Println("You win!")
	} else {
		fmt.Println("AI wins!")
	}
}

func aiDemonstration(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) {
	fmt.Println("\nAI vs AI Demonstration")
	fmt.Println("=====================")

	// Create a new game
	gameInstance := game.NewRPSGame(deckSize, handSize, maxRounds)

	// Create MCTS for AI
	mctsParams := mcts.DefaultRPSMCTSParams()
	// Reduce simulation count for faster demo
	mctsParams.NumSimulations = 400
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

			// Display game state
			fmt.Println(gameInstance.String())

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

			// Display game state
			fmt.Println(gameInstance.String())
		}

		moveCount++

		// Pause between moves
		time.Sleep(1 * time.Second)
	}

	// Game over
	fmt.Println("\nGame Over!")

	winner := gameInstance.GetWinner()
	if winner == game.NoPlayer {
		fmt.Println("It's a draw!")
	} else if winner == game.Player1 {
		fmt.Println("Player 1 wins!")
	} else {
		fmt.Println("Player 2 wins!")
	}
}
