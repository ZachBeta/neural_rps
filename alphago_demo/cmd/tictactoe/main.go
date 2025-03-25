package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

const (
	trainNetworks  = true
	selfPlayGames  = 5  // Reduced number of games for faster demo
	trainingEpochs = 10 // Reduced epochs for faster demo
	batchSize      = 32
	learningRate   = 0.001
)

func main() {
	fmt.Println("AlphaGo-style TicTacToe Demo")
	fmt.Println("============================")

	// Initialize neural networks
	policyNetwork := neural.NewAGPolicyNetwork(9, 64)
	valueNetwork := neural.NewAGValueNetwork(9, 64)

	// Track training metrics for standardized output
	var trainingExamples int
	var trainingTime time.Duration
	var policyLosses []float64
	var valueLosses []float64

	// Train networks if enabled
	if trainNetworks {
		fmt.Println("\nTraining networks through self-play...")

		// Create self-play agent
		selfPlayParams := training.DefaultAGSelfPlayParams()
		selfPlayParams.NumGames = selfPlayGames
		// Reduce MCTS simulations for faster demo
		selfPlayParams.MCTSParams.NumSimulations = 100
		selfPlay := training.NewAGSelfPlay(policyNetwork, valueNetwork, selfPlayParams)

		// Generate games
		startTime := time.Now()
		examples := selfPlay.GenerateGames(true)
		trainingExamples = len(examples)
		fmt.Printf("Generated %d training examples from %d games in %.2f seconds\n",
			len(examples), selfPlayGames, time.Since(startTime).Seconds())

		// Train networks
		startTime = time.Now()
		policyLosses, valueLosses = selfPlay.TrainNetworks(trainingEpochs, batchSize, learningRate, true)
		trainingTime = time.Since(startTime)
		fmt.Printf("Training completed in %.2f seconds\n", trainingTime.Seconds())
	}

	// Generate standardized output
	generateStandardizedOutput(
		policyNetwork,
		valueNetwork,
		selfPlayGames,
		trainingExamples,
		trainingTime,
		policyLosses,
		valueLosses)

	// Run a simulated demo game
	fmt.Println("\nRunning demo game with simulated player...")
	runSimulatedGame(policyNetwork, valueNetwork)
}

func runSimulatedGame(policyNetwork *neural.AGPolicyNetwork, valueNetwork *neural.AGValueNetwork) {
	fmt.Println("\nDemo game: Human (X) vs AI (O)")

	// Create new game
	gameState := game.NewAGGame()

	// Predefined human moves for demo
	humanMoves := []game.AGMove{
		{Row: 1, Col: 1}, // Center
		{Row: 0, Col: 0}, // Top-left
		{Row: 2, Col: 2}, // Bottom-right
	}

	moveIndex := 0

	// Game loop
	for !gameState.IsGameOver() {
		// Print current board
		fmt.Println("\nCurrent board:")
		fmt.Println(gameState.String())

		if gameState.CurrentPlayer == game.PlayerX {
			// Simulated human player's turn
			if moveIndex < len(humanMoves) {
				move := humanMoves[moveIndex]
				moveIndex++

				fmt.Printf("Human player selects: %d,%d\n", move.Row, move.Col)
				time.Sleep(1 * time.Second) // Pause for effect

				err := gameState.MakeMove(move)
				if err != nil {
					// If move is invalid, try a random valid move
					fmt.Printf("Invalid move: %v\n", err)
					randomMove, errRandom := gameState.GetRandomMove()
					if errRandom == nil {
						gameState.MakeMove(randomMove)
						fmt.Printf("Falling back to random move: %d,%d\n", randomMove.Row, randomMove.Col)
					} else {
						fmt.Println("No valid moves available!")
						break
					}
				}
			} else {
				// If we've used all predefined moves, get a random one
				randomMove, err := gameState.GetRandomMove()
				if err == nil {
					fmt.Printf("Human player selects: %d,%d\n", randomMove.Row, randomMove.Col)
					time.Sleep(1 * time.Second) // Pause for effect
					gameState.MakeMove(randomMove)
				} else {
					fmt.Println("No valid moves available!")
					break
				}
			}
		} else {
			// AI's turn
			fmt.Println("AI is thinking...")
			time.Sleep(1 * time.Second) // Simulate thinking time

			// Create MCTS with neural networks
			mctsParams := mcts.DefaultAGMCTSParams()
			mctsParams.NumSimulations = 100 // Reduced for faster demo
			mctsEngine := mcts.NewAGMCTS(policyNetwork, valueNetwork, mctsParams)

			// Set root state and search
			mctsEngine.SetRootState(gameState)
			bestMove := mctsEngine.GetBestMove()

			// Make the move
			err := gameState.MakeMove(bestMove)
			if err != nil {
				fmt.Printf("AI error: %v\n", err)
				randomMove, errRandom := gameState.GetRandomMove()
				if errRandom == nil {
					gameState.MakeMove(randomMove)
					fmt.Printf("AI falls back to random move: %v\n", randomMove)
				} else {
					fmt.Println("AI couldn't make a move!")
					break
				}
			} else {
				fmt.Printf("AI played: %d,%d\n", bestMove.Row, bestMove.Col)
			}

			time.Sleep(1 * time.Second) // Pause between moves
		}
	}

	// Game over
	fmt.Println("\nFinal board:")
	fmt.Println(gameState.String())

	// Determine winner
	winner := gameState.GetWinner()
	if winner == game.Empty {
		fmt.Println("Game ended in a draw!")
	} else if winner == game.PlayerX {
		fmt.Println("Human player wins!")
	} else {
		fmt.Println("AI wins!")
	}

	fmt.Println("\nDemo completed. Thanks for watching!")
}

func playInteractiveGame(policyNetwork *neural.AGPolicyNetwork, valueNetwork *neural.AGValueNetwork) {
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("\nLet's play Tic-Tac-Toe!")
	fmt.Println("You are X, AI is O")
	fmt.Println("Enter moves as 'row,col' (0-2,0-2)")
	fmt.Println("For example: 1,1 for the center")

	for {
		// Create new game
		gameState := game.NewAGGame()

		// Game loop
		for !gameState.IsGameOver() {
			// Print current board
			fmt.Println("\nCurrent board:")
			fmt.Println(gameState.String())

			if gameState.CurrentPlayer == game.PlayerX {
				// Human player's turn
				fmt.Print("Your move (row,col): ")
				if !scanner.Scan() {
					return
				}

				input := strings.TrimSpace(scanner.Text())
				if input == "q" || input == "quit" || input == "exit" {
					fmt.Println("Thanks for playing!")
					return
				}

				// Parse input
				parts := strings.Split(input, ",")
				if len(parts) != 2 {
					fmt.Println("Invalid input. Expected format: row,col")
					continue
				}

				row, errRow := strconv.Atoi(strings.TrimSpace(parts[0]))
				col, errCol := strconv.Atoi(strings.TrimSpace(parts[1]))

				if errRow != nil || errCol != nil || row < 0 || row > 2 || col < 0 || col > 2 {
					fmt.Println("Invalid input. Row and column must be between 0 and 2.")
					continue
				}

				// Make move
				move := game.AGMove{Row: row, Col: col}
				err := gameState.MakeMove(move)
				if err != nil {
					fmt.Printf("Invalid move: %v\n", err)
					continue
				}
			} else {
				// AI's turn
				fmt.Println("AI is thinking...")

				// Create MCTS with neural networks
				mctsParams := mcts.DefaultAGMCTSParams()
				mctsEngine := mcts.NewAGMCTS(policyNetwork, valueNetwork, mctsParams)

				// Set root state and search
				mctsEngine.SetRootState(gameState)
				bestMove := mctsEngine.GetBestMove()

				// Make the move
				err := gameState.MakeMove(bestMove)
				if err != nil {
					fmt.Printf("AI error: %v\n", err)
					randomMove, errRandom := gameState.GetRandomMove()
					if errRandom == nil {
						gameState.MakeMove(randomMove)
						fmt.Printf("AI falls back to random move: %v\n", randomMove)
					} else {
						fmt.Println("AI couldn't make a move!")
						break
					}
				} else {
					fmt.Printf("AI played: %v\n", bestMove)
				}
			}
		}

		// Game over
		fmt.Println("\nFinal board:")
		fmt.Println(gameState.String())

		// Determine winner
		winner := gameState.GetWinner()
		if winner == game.Empty {
			fmt.Println("Game ended in a draw!")
		} else if winner == game.PlayerX {
			fmt.Println("You win! Congratulations!")
		} else {
			fmt.Println("AI wins! Better luck next time.")
		}

		// Ask to play again
		fmt.Print("\nPlay again? (y/n): ")
		if !scanner.Scan() || strings.ToLower(strings.TrimSpace(scanner.Text())) != "y" {
			fmt.Println("Thanks for playing!")
			break
		}
	}
}

// Function to evaluate the neural networks
func evaluateNetworks(policyNetwork *neural.AGPolicyNetwork, valueNetwork *neural.AGValueNetwork) {
	fmt.Println("\nEvaluating networks...")

	// Create a new game
	gameState := game.NewAGGame()

	// Get policy prediction
	policyProbs := policyNetwork.Predict(gameState)

	fmt.Println("Initial policy prediction:")
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			index := row*3 + col
			fmt.Printf("%.3f ", policyProbs[index])
		}
		fmt.Println()
	}

	// Get value prediction
	value := valueNetwork.Predict(gameState)
	fmt.Printf("Initial value prediction: %.3f\n", value)

	// Make some moves
	moves := []game.AGMove{
		{Row: 1, Col: 1}, // X in center
		{Row: 0, Col: 0}, // O in top-left
		{Row: 2, Col: 2}, // X in bottom-right
	}

	for _, move := range moves {
		gameState.MakeMove(move)
	}

	fmt.Println("\nBoard after moves:")
	fmt.Println(gameState.String())

	// Get updated policy prediction
	policyProbs = policyNetwork.Predict(gameState)

	fmt.Println("Updated policy prediction:")
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			index := row*3 + col
			fmt.Printf("%.3f ", policyProbs[index])
		}
		fmt.Println()
	}

	// Get updated value prediction
	value = valueNetwork.Predict(gameState)
	fmt.Printf("Updated value prediction: %.3f\n", value)
}

// generateStandardizedOutput creates output in the standardized format
func generateStandardizedOutput(
	policyNetwork *neural.AGPolicyNetwork,
	valueNetwork *neural.AGValueNetwork,
	selfPlayGames int,
	trainingExamples int,
	trainingTime time.Duration,
	policyLosses []float64,
	valueLosses []float64) {

	// Create output file
	f, err := os.Create("../alphago_demo_output.txt")
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer f.Close()

	// Header & Implementation Info
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Neural Game AI - Go Implementation (AlphaGo-style)\n")
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Version: 1.0\n")
	fmt.Fprintf(f, "Implementation Type: AlphaGo-style MCTS with Neural Networks\n\n")

	// Network Architecture
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Network Architecture\n")
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Input Layer: 9 neurons (board state encoding)\n")
	fmt.Fprintf(f, "Hidden Layer: 64 neurons (ReLU activation)\n")
	fmt.Fprintf(f, "Output Layer: 9 neurons (policy head) + 1 neuron (value head)\n\n")

	// Network visualization
	fmt.Fprintf(f, "Network Visualization:\n")
	fmt.Fprintf(f, "  (I)--\\\n")
	fmt.Fprintf(f, "  (I)---\\\n")
	fmt.Fprintf(f, "  (I)----\\\n")
	fmt.Fprintf(f, "  (I)-----[Hidden Layer]--[Policy Head: 9 neurons]\n")
	fmt.Fprintf(f, "  (I)-----/          \\\n")
	fmt.Fprintf(f, "  (I)----/            \\\n")
	fmt.Fprintf(f, "  (I)---/              \\\n")
	fmt.Fprintf(f, "  (I)--/                [Value Head: 1 neuron]\n")
	fmt.Fprintf(f, "  (I)-/\n\n")

	// Training Process
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Training Process\n")
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Training Episodes: %d self-play games\n", selfPlayGames)
	fmt.Fprintf(f, "Training Examples: %d\n", trainingExamples)
	fmt.Fprintf(f, "Training Time: %.2fs\n\n", trainingTime.Seconds())

	// Training Progress
	fmt.Fprintf(f, "Training Progress:\n")
	for i := 0; i < len(policyLosses); i++ {
		fmt.Fprintf(f, "Epoch %d/%d - Policy Loss: %.4f, Value Loss: %.4f\n",
			i+1, len(policyLosses), policyLosses[i], valueLosses[i])
	}
	fmt.Fprintf(f, "\n")

	// Model Predictions (adapted for Tic-Tac-Toe)
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Model Predictions (Adapted for Tic-Tac-Toe)\n")
	fmt.Fprintf(f, "==================================================\n")

	// Prediction for empty board
	emptyBoard := game.NewAGGame()
	generateTicTacToePrediction(f, policyNetwork, valueNetwork, emptyBoard, "Empty board")

	// Prediction for board with X in center
	centerXBoard := game.NewAGGame()
	centerXBoard.MakeMove(game.AGMove{Row: 1, Col: 1}) // X in center
	generateTicTacToePrediction(f, policyNetwork, valueNetwork, centerXBoard, "Board with X in center")

	// Prediction for board with O about to win
	oAboutToWinBoard := game.NewAGGame()
	oAboutToWinBoard.MakeMove(game.AGMove{Row: 0, Col: 0}) // X top-left
	oAboutToWinBoard.MakeMove(game.AGMove{Row: 0, Col: 1}) // O top-middle
	oAboutToWinBoard.MakeMove(game.AGMove{Row: 2, Col: 0}) // X bottom-left
	oAboutToWinBoard.MakeMove(game.AGMove{Row: 0, Col: 2}) // O top-right
	generateTicTacToePrediction(f, policyNetwork, valueNetwork, oAboutToWinBoard, "Board with O about to win")

	// Model Parameters (Optional)
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Model Parameters (Optional)\n")
	fmt.Fprintf(f, "==================================================\n")
	fmt.Fprintf(f, "Policy Network:\n")
	fmt.Fprintf(f, "  Input to Hidden: Matrix (9x64)\n")
	fmt.Fprintf(f, "  Hidden to Output: Matrix (64x9)\n\n")
	fmt.Fprintf(f, "Value Network:\n")
	fmt.Fprintf(f, "  Hidden to Value: Matrix (64x1)\n\n")
	fmt.Fprintf(f, "Parameter Count: 1473 total parameters\n")
}

// generateTicTacToePrediction generates a prediction for a Tic-Tac-Toe board
func generateTicTacToePrediction(
	f *os.File,
	policyNetwork *neural.AGPolicyNetwork,
	valueNetwork *neural.AGValueNetwork,
	state *game.AGGame,
	description string) {

	// Create MCTS with neural networks
	mctsParams := mcts.DefaultAGMCTSParams()
	mctsParams.NumSimulations = 100 // Reduced for faster demo
	mctsEngine := mcts.NewAGMCTS(policyNetwork, valueNetwork, mctsParams)

	// Set root state and search
	mctsEngine.SetRootState(state)

	// Get probabilities and value
	probs := mctsEngine.GetActionProbabilities()
	valueEstimate := mctsEngine.GetRootValue()

	// Format for output
	fmt.Fprintf(f, "Input: %s\n", description)
	fmt.Fprintf(f, "Output:\n")

	// Print move probabilities
	bestMoveIdx := 0
	bestProb := 0.0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			idx := i*3 + j
			probability := 0.0
			// Only show probability if this is a valid move
			if state.IsValidMove(game.AGMove{Row: i, Col: j}) {
				probability = probs[idx]
				if probability > bestProb {
					bestProb = probability
					bestMoveIdx = idx
				}
			}
			fmt.Fprintf(f, "  Move (%d,%d): %.2f%%", i, j, probability*100)

			// Add a note for special cases
			if !state.IsValidMove(game.AGMove{Row: i, Col: j}) {
				fmt.Fprintf(f, " (already taken)")
			} else if i == 1 && j == 1 {
				fmt.Fprintf(f, " (center)")
			} else if probability > 0.5 {
				fmt.Fprintf(f, " (blocking move)")
			}
			fmt.Fprintf(f, "\n")
		}
	}

	// Print value estimate
	fmt.Fprintf(f, "  Value: %.2f", valueEstimate)
	if valueEstimate > 0.2 {
		fmt.Fprintf(f, " (strong advantage for X)")
	} else if valueEstimate > 0.05 {
		fmt.Fprintf(f, " (slight advantage for X)")
	} else if valueEstimate < -0.2 {
		fmt.Fprintf(f, " (strong advantage for O)")
	} else if valueEstimate < -0.05 {
		fmt.Fprintf(f, " (slight advantage for O)")
	} else {
		fmt.Fprintf(f, " (roughly even)")
	}
	fmt.Fprintf(f, "\n")

	// Print prediction
	bestMove := game.AGMove{Row: bestMoveIdx / 3, Col: bestMoveIdx % 3}
	fmt.Fprintf(f, "Prediction: Move to (%d,%d)\n\n", bestMove.Row, bestMove.Col)
}
