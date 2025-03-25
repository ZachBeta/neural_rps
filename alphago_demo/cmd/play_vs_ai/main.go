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
)

const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10

	// MCTS parameters
	mctsSimulations = 200
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Get model file path from command-line arguments or use default
	modelPath := "output/rps_policy2.model"
	valueModelPath := "output/rps_value2.model"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		valueModelPath = os.Args[2]
	}

	// Load policy network from file
	policyNetwork := neural.NewRPSPolicyNetwork(128)
	err := policyNetwork.LoadFromFile(modelPath)
	if err != nil {
		fmt.Printf("Failed to load policy model from %s: %v\n", modelPath, err)
		fmt.Println("Starting with a new model instead.")
	} else {
		fmt.Printf("Loaded policy model from %s\n", modelPath)
	}

	// Load value network from file
	valueNetwork := neural.NewRPSValueNetwork(128)
	err = valueNetwork.LoadFromFile(valueModelPath)
	if err != nil {
		fmt.Printf("Failed to load value model from %s: %v\n", valueModelPath, err)
		fmt.Println("Starting with a new model instead.")
	} else {
		fmt.Printf("Loaded value model from %s\n", valueModelPath)
	}

	// Create MCTS engine for the AI
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = mctsSimulations
	mctsEngine := mcts.NewRPSMCTS(policyNetwork, valueNetwork, mctsParams)

	// Create the game
	gameInstance := game.NewRPSGame(deckSize, handSize, maxRounds)

	// Main game loop
	scanner := bufio.NewScanner(os.Stdin)
	for !gameInstance.IsGameOver() {
		// Print current game state
		fmt.Println(gameInstance.String())

		// Get the current player
		currentPlayer := gameInstance.CurrentPlayer

		// Human player is Player1, AI is Player2
		if currentPlayer == game.Player1 {
			// Human's turn
			fmt.Println("Your turn! Choose a card and position.")
			move, err := getHumanMove(scanner, gameInstance)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}

			// Make the move
			err = gameInstance.MakeMove(move)
			if err != nil {
				fmt.Printf("Invalid move: %v\n", err)
				continue
			}
		} else {
			// AI's turn
			fmt.Println("AI is thinking...")

			// Set the root state for MCTS
			mctsEngine.SetRootState(gameInstance)

			// Search for the best move
			bestNode := mctsEngine.Search()

			if bestNode == nil || bestNode.Move == nil {
				fmt.Println("AI couldn't find a valid move!")
				// Fallback to random move
				randomMove, err := gameInstance.GetRandomMove()
				if err != nil {
					fmt.Printf("Error: %v\n", err)
					break
				}
				randomMove.Player = currentPlayer
				err = gameInstance.MakeMove(randomMove)
				if err != nil {
					fmt.Printf("Error: %v\n", err)
					break
				}
				fmt.Printf("AI plays card %d at position %d\n", randomMove.CardIndex, randomMove.Position)
			} else {
				// Execute the best move found by MCTS
				aiMove := *bestNode.Move
				aiMove.Player = currentPlayer
				err := gameInstance.MakeMove(aiMove)
				if err != nil {
					fmt.Printf("Error: %v\n", err)
					break
				}
				fmt.Printf("AI plays card %d at position %d\n", aiMove.CardIndex, aiMove.Position)
			}
		}

		// Add a brief pause so human can see the AI's move
		time.Sleep(1 * time.Second)
	}

	// Print final game state
	fmt.Println(gameInstance.String())

	// Print game result
	winner := gameInstance.GetWinner()
	switch winner {
	case game.Player1:
		fmt.Println("You win!")
	case game.Player2:
		fmt.Println("AI wins!")
	default:
		fmt.Println("It's a draw!")
	}
}

// getHumanMove gets a move from the human player
func getHumanMove(scanner *bufio.Scanner, gameState *game.RPSGame) (game.RPSMove, error) {
	// Print the player's hand
	fmt.Println("Your hand:")
	for i, card := range gameState.Player1Hand {
		fmt.Printf("%d: %s\n", i, cardTypeToString(card.Type))
	}

	// Print the valid positions
	fmt.Println("Valid positions:")
	for i := 0; i < 9; i++ {
		// Check if position is empty
		if gameState.Board[i].Owner == game.NoPlayer {
			fmt.Printf("%d ", i)
		}
	}
	fmt.Println()

	// Get card index
	fmt.Print("Choose card index (0-4): ")
	if !scanner.Scan() {
		return game.RPSMove{}, fmt.Errorf("failed to read input")
	}
	cardIndexStr := scanner.Text()
	cardIndex, err := strconv.Atoi(strings.TrimSpace(cardIndexStr))
	if err != nil || cardIndex < 0 || cardIndex >= len(gameState.Player1Hand) {
		return game.RPSMove{}, fmt.Errorf("invalid card index")
	}

	// Get position
	fmt.Print("Choose position (0-8): ")
	if !scanner.Scan() {
		return game.RPSMove{}, fmt.Errorf("failed to read input")
	}
	positionStr := scanner.Text()
	position, err := strconv.Atoi(strings.TrimSpace(positionStr))
	if err != nil || position < 0 || position > 8 {
		return game.RPSMove{}, fmt.Errorf("invalid position")
	}

	// Check if the position is already occupied
	if gameState.Board[position].Owner != game.NoPlayer {
		return game.RPSMove{}, fmt.Errorf("position %d is already occupied", position)
	}

	return game.RPSMove{
		Player:    gameState.CurrentPlayer,
		CardIndex: cardIndex,
		Position:  position,
	}, nil
}

// Helper function to convert card type to string
func cardTypeToString(cardType game.RPSCardType) string {
	switch cardType {
	case game.Rock:
		return "Rock"
	case game.Paper:
		return "Paper"
	case game.Scissors:
		return "Scissors"
	default:
		return "Unknown"
	}
}
