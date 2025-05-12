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
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
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

	fmt.Println("Balanced AlphaGo-Style Rock-Paper-Scissors Card Game")
	fmt.Println("====================================================")
	fmt.Println("This version implements a two-round system to balance first-mover advantage.")
	fmt.Println("Players switch positions after the first round, and the final winner is")
	fmt.Println("determined by the combined score from both rounds.")

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
		fmt.Println("1. Play balanced match against AI (two rounds)")
		fmt.Println("2. Watch balanced AI vs AI demonstration (two rounds)")
		fmt.Println("3. Exit")
		fmt.Print("Select an option: ")

		var choice int
		fmt.Scanln(&choice)

		switch choice {
		case 1:
			playBalancedMatchAgainstAI(policyNetwork, valueNetwork)
		case 2:
			balancedAIDemonstration(policyNetwork, valueNetwork)
		case 3:
			fmt.Println("Goodbye!")
			return
		default:
			fmt.Println("Invalid choice. Please try again.")
		}
	}
}

func playBalancedMatchAgainstAI(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) {
	fmt.Println("\nPlaying balanced match against AI (two rounds)")
	fmt.Println("=============================================")

	// Scores for the two rounds
	humanScore := 0
	aiScore := 0

	// Play first round (human as Player 1)
	fmt.Println("\n=== ROUND 1: You play as Player 1 ===")
	humanRound1Cards, aiRound1Cards := playRoundAgainstAI(policyNetwork, valueNetwork, true)

	// Determine winner of first round
	if humanRound1Cards > aiRound1Cards {
		fmt.Printf("Round 1 result: You win with %d cards vs AI's %d cards\n", humanRound1Cards, aiRound1Cards)
		humanScore++
	} else if aiRound1Cards > humanRound1Cards {
		fmt.Printf("Round 1 result: AI wins with %d cards vs your %d cards\n", aiRound1Cards, humanRound1Cards)
		aiScore++
	} else {
		fmt.Printf("Round 1 result: Draw with %d cards each\n", humanRound1Cards)
		// No score change for a draw
	}

	// Play second round (human as Player 2)
	fmt.Println("\n=== ROUND 2: You play as Player 2 ===")
	aiRound2Cards, humanRound2Cards := playRoundAgainstAI(policyNetwork, valueNetwork, false)

	// Determine winner of second round
	if humanRound2Cards > aiRound2Cards {
		fmt.Printf("Round 2 result: You win with %d cards vs AI's %d cards\n", humanRound2Cards, aiRound2Cards)
		humanScore++
	} else if aiRound2Cards > humanRound2Cards {
		fmt.Printf("Round 2 result: AI wins with %d cards vs your %d cards\n", aiRound2Cards, humanRound2Cards)
		aiScore++
	} else {
		fmt.Printf("Round 2 result: Draw with %d cards each\n", humanRound2Cards)
		// No score change for a draw
	}

	// Overall match result
	fmt.Println("\n=== MATCH RESULT ===")
	fmt.Printf("Your total score: %d\n", humanScore)
	fmt.Printf("AI's total score: %d\n", aiScore)

	if humanScore > aiScore {
		fmt.Println("You win the match!")
	} else if aiScore > humanScore {
		fmt.Println("AI wins the match!")
	} else {
		fmt.Println("The match is a draw!")

		// Optional: Offer tiebreaker
		fmt.Print("\nWould you like to play a tiebreaker round? (y/n): ")
		var choice string
		fmt.Scanln(&choice)
		if strings.ToLower(choice) == "y" {
			fmt.Println("\n=== TIEBREAKER ROUND ===")
			// Randomly assign positions for tiebreaker
			humanIsPlayer1 := rand.Intn(2) == 0
			if humanIsPlayer1 {
				fmt.Println("You play as Player 1 (randomly assigned)")
				humanTieCards, aiTieCards := playRoundAgainstAI(policyNetwork, valueNetwork, true)
				if humanTieCards > aiTieCards {
					fmt.Printf("Tiebreaker result: You win with %d cards vs AI's %d cards\n", humanTieCards, aiTieCards)
					fmt.Println("You win the match!")
				} else if aiTieCards > humanTieCards {
					fmt.Printf("Tiebreaker result: AI wins with %d cards vs your %d cards\n", aiTieCards, humanTieCards)
					fmt.Println("AI wins the match!")
				} else {
					fmt.Printf("Tiebreaker result: Another draw with %d cards each\n", humanTieCards)
					fmt.Println("The match remains a draw after tiebreaker!")
				}
			} else {
				fmt.Println("You play as Player 2 (randomly assigned)")
				aiTieCards, humanTieCards := playRoundAgainstAI(policyNetwork, valueNetwork, false)
				if humanTieCards > aiTieCards {
					fmt.Printf("Tiebreaker result: You win with %d cards vs AI's %d cards\n", humanTieCards, aiTieCards)
					fmt.Println("You win the match!")
				} else if aiTieCards > humanTieCards {
					fmt.Printf("Tiebreaker result: AI wins with %d cards vs your %d cards\n", aiTieCards, humanTieCards)
					fmt.Println("AI wins the match!")
				} else {
					fmt.Printf("Tiebreaker result: Another draw with %d cards each\n", humanTieCards)
					fmt.Println("The match remains a draw after tiebreaker!")
				}
			}
		}
	}
}

// playRoundAgainstAI plays a single round and returns the card counts (player1Cards, player2Cards)
func playRoundAgainstAI(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork, humanIsPlayer1 bool) (int, int) {
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

		isHumanTurn := (gameInstance.CurrentPlayer == game.Player1 && humanIsPlayer1) ||
			(gameInstance.CurrentPlayer == game.Player2 && !humanIsPlayer1)

		if isHumanTurn {
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
			playerHand := gameInstance.Player1Hand
			if gameInstance.CurrentPlayer == game.Player2 {
				playerHand = gameInstance.Player2Hand
			}

			for i, card := range playerHand {
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
				fmt.Print("Select card (0-" + strconv.Itoa(len(playerHand)-1) + "): ")
				var input string
				scanner.Scan()
				input = scanner.Text()

				var err error
				cardIndex, err = strconv.Atoi(input)
				if err != nil || cardIndex < 0 || cardIndex >= len(playerHand) {
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
				Player:    gameInstance.CurrentPlayer,
			}

			err := gameInstance.MakeMove(move)
			if err != nil {
				fmt.Printf("Error making move: %v\n", err)
				return 0, 0
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
					return 0, 0
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
					return 0, 0
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
	fmt.Println("\nRound Complete!")
	fmt.Println(gameInstance.String())

	// Count cards for each player
	player1Cards := 0
	player2Cards := 0
	for _, card := range gameInstance.Board {
		if card.Owner == game.Player1 {
			player1Cards++
		} else if card.Owner == game.Player2 {
			player2Cards++
		}
	}

	return player1Cards, player2Cards
}

func balancedAIDemonstration(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) {
	fmt.Println("\nBalanced AI vs AI Demonstration (Two Rounds)")
	fmt.Println("===========================================")

	// Scores for the two rounds
	ai1Score := 0
	ai2Score := 0

	// Play first round
	fmt.Println("\n=== ROUND 1 ===")
	ai1Round1Cards, ai2Round1Cards := playAIvsAIRound(policyNetwork, valueNetwork)

	// Determine winner of first round
	if ai1Round1Cards > ai2Round1Cards {
		fmt.Printf("Round 1 result: AI 1 wins with %d cards vs AI 2's %d cards\n", ai1Round1Cards, ai2Round1Cards)
		ai1Score++
	} else if ai2Round1Cards > ai1Round1Cards {
		fmt.Printf("Round 1 result: AI 2 wins with %d cards vs AI 1's %d cards\n", ai2Round1Cards, ai1Round1Cards)
		ai2Score++
	} else {
		fmt.Printf("Round 1 result: Draw with %d cards each\n", ai1Round1Cards)
		// No score change for a draw
	}

	// Play second round with positions swapped
	fmt.Println("\n=== ROUND 2 (positions swapped) ===")
	ai2Round2Cards, ai1Round2Cards := playAIvsAIRound(policyNetwork, valueNetwork)

	// Determine winner of second round
	if ai1Round2Cards > ai2Round2Cards {
		fmt.Printf("Round 2 result: AI 1 wins with %d cards vs AI 2's %d cards\n", ai1Round2Cards, ai2Round2Cards)
		ai1Score++
	} else if ai2Round2Cards > ai1Round2Cards {
		fmt.Printf("Round 2 result: AI 2 wins with %d cards vs AI 1's %d cards\n", ai2Round2Cards, ai1Round2Cards)
		ai2Score++
	} else {
		fmt.Printf("Round 2 result: Draw with %d cards each\n", ai1Round2Cards)
		// No score change for a draw
	}

	// Overall match result
	fmt.Println("\n=== MATCH RESULT ===")
	fmt.Printf("AI 1 total score: %d\n", ai1Score)
	fmt.Printf("AI 2 total score: %d\n", ai2Score)

	if ai1Score > ai2Score {
		fmt.Println("AI 1 wins the match!")
	} else if ai2Score > ai1Score {
		fmt.Println("AI 2 wins the match!")
	} else {
		fmt.Println("The match is a draw!")

		// Optional: Play tiebreaker
		fmt.Println("\n=== TIEBREAKER ROUND ===")
		// Randomly assign positions for tiebreaker
		if rand.Intn(2) == 0 {
			fmt.Println("AI 1 plays as Player 1 (randomly assigned)")
			ai1TieCards, ai2TieCards := playAIvsAIRound(policyNetwork, valueNetwork)
			if ai1TieCards > ai2TieCards {
				fmt.Printf("Tiebreaker result: AI 1 wins with %d cards vs AI 2's %d cards\n", ai1TieCards, ai2TieCards)
				fmt.Println("AI 1 wins the match!")
			} else if ai2TieCards > ai1TieCards {
				fmt.Printf("Tiebreaker result: AI 2 wins with %d cards vs AI 1's %d cards\n", ai2TieCards, ai1TieCards)
				fmt.Println("AI 2 wins the match!")
			} else {
				fmt.Printf("Tiebreaker result: Another draw with %d cards each\n", ai1TieCards)
				fmt.Println("The match remains a draw after tiebreaker!")
			}
		} else {
			fmt.Println("AI 2 plays as Player 1 (randomly assigned)")
			ai2TieCards, ai1TieCards := playAIvsAIRound(policyNetwork, valueNetwork)
			if ai1TieCards > ai2TieCards {
				fmt.Printf("Tiebreaker result: AI 1 wins with %d cards vs AI 2's %d cards\n", ai1TieCards, ai2TieCards)
				fmt.Println("AI 1 wins the match!")
			} else if ai2TieCards > ai1TieCards {
				fmt.Printf("Tiebreaker result: AI 2 wins with %d cards vs AI 1's %d cards\n", ai2TieCards, ai1TieCards)
				fmt.Println("AI 2 wins the match!")
			} else {
				fmt.Printf("Tiebreaker result: Another draw with %d cards each\n", ai1TieCards)
				fmt.Println("The match remains a draw after tiebreaker!")
			}
		}
	}
}

// playAIvsAIRound plays a single round with AI vs AI and returns card counts (player1Cards, player2Cards)
func playAIvsAIRound(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork) (int, int) {
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
				return 0, 0
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
				return 0, 0
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
	fmt.Println("\nRound Complete!")

	// Count cards for each player
	player1Cards := 0
	player2Cards := 0
	for _, card := range gameInstance.Board {
		if card.Owner == game.Player1 {
			player1Cards++
		} else if card.Owner == game.Player2 {
			player2Cards++
		}
	}

	return player1Cards, player2Cards
}
