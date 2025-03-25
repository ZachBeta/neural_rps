package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/pkg/agent"
	"github.com/zachbeta/neural_rps/pkg/neural"
)

// GameState represents the state of the game
type GameState struct {
	Board         string
	Hand1         string
	Hand2         string
	CurrentPlayer int
}

// Move represents a move in the game
type Move struct {
	CardIndex int
	Position  int
}

func main() {
	// Parse command line flags
	modelPath := flag.String("model", "", "Path to the model file")
	stateStr := flag.String("state", "", "Game state string")
	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Check if state was provided
	if *stateStr == "" {
		fmt.Fprintln(os.Stderr, "Error: Game state not provided")
		os.Exit(1)
	}

	// Parse the game state
	state, err := parseGameState(*stateStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing game state: %v\n", err)
		os.Exit(1)
	}

	// Choose the best move
	var move Move
	if *modelPath != "" {
		// Try to use the model
		move, err = chooseBestMove(*modelPath, state)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Error using model: %v\n", err)
			fmt.Fprintln(os.Stderr, "Falling back to random move selection")
			move = chooseRandomMove(state)
		}
	} else {
		// No model provided, use random move
		move = chooseRandomMove(state)
	}

	// Output the move in the expected format
	fmt.Printf("%d:%d\n", move.CardIndex, move.Position)
}

// parseGameState parses a game state string into a GameState struct
func parseGameState(stateStr string) (GameState, error) {
	// Format: Board:R.S.P...|Hand1:RPS|Hand2:RPS|Current:1
	parts := make(map[string]string)
	for _, part := range strings.Split(stateStr, "|") {
		kv := strings.Split(part, ":")
		if len(kv) == 2 {
			parts[kv[0]] = kv[1]
		}
	}

	// Check required fields
	if parts["Board"] == "" || parts["Hand1"] == "" || parts["Hand2"] == "" || parts["Current"] == "" {
		return GameState{}, fmt.Errorf("missing required game state fields")
	}

	// Parse current player
	currentPlayer, err := strconv.Atoi(parts["Current"])
	if err != nil {
		return GameState{}, fmt.Errorf("invalid current player: %v", err)
	}

	return GameState{
		Board:         parts["Board"],
		Hand1:         parts["Hand1"],
		Hand2:         parts["Hand2"],
		CurrentPlayer: currentPlayer,
	}, nil
}

// gameStateToInput converts a game state to neural network input
func gameStateToInput(state GameState) []float64 {
	// Input structure: 9 positions * 3 features (R, P, S)
	input := make([]float64, 27)

	// Process board state
	for i, c := range state.Board {
		if i >= 9 {
			break
		}

		if c == '.' {
			continue // Empty space
		}

		// Determine card type and owner
		isPlayer1 := c == 'R' || c == 'P' || c == 'S'
		cardType := strings.ToUpper(string(c))
		var cardIndex int

		switch cardType {
		case "R":
			cardIndex = 0
		case "P":
			cardIndex = 1
		case "S":
			cardIndex = 2
		default:
			continue
		}

		// Set position for this card type
		val := 1.0
		if !isPlayer1 {
			val = -1.0
		}
		input[i*3+cardIndex] = val
	}

	return input
}

// getValidMoves returns all valid moves for the current state
func getValidMoves(state GameState) []Move {
	var moves []Move
	var hand string
	if state.CurrentPlayer == 1 {
		hand = state.Hand1
	} else {
		hand = state.Hand2
	}

	// Check each position on the board
	for pos := 0; pos < 9; pos++ {
		if pos < len(state.Board) && state.Board[pos] == '.' {
			// Position is empty, can play any card from hand
			for cardIdx := 0; cardIdx < len(hand); cardIdx++ {
				moves = append(moves, Move{
					CardIndex: cardIdx,
					Position:  pos,
				})
			}
		}
	}

	return moves
}

// chooseBestMove uses a neural network to choose the best move
func chooseBestMove(modelPath string, state GameState) (Move, error) {
	validMoves := getValidMoves(state)
	if len(validMoves) == 0 {
		return Move{}, fmt.Errorf("no valid moves available")
	}

	// If only one move, return it
	if len(validMoves) == 1 {
		return validMoves[0], nil
	}

	// Try to load the model
	network := neural.NewNetwork(27, 16, 9) // 27 inputs, 16 hidden, 9 outputs (board positions)
	err := network.LoadWeights(modelPath)
	if err != nil {
		return Move{}, fmt.Errorf("failed to load model: %v", err)
	}

	// Create a PPO agent with the loaded network
	ppoAgent := &agent.PPOAgent{
		Network:      network,
		StateSize:    27,
		ActionSize:   9,
		LearningRate: 0.01,
	}

	// Convert game state to network input
	input := gameStateToInput(state)

	// Get policy probabilities
	probs := ppoAgent.GetPolicyProbs(input)

	// Find the best valid move based on policy probabilities
	bestScore := -1.0
	bestMoveIdx := 0
	for i, move := range validMoves {
		score := probs[move.Position]
		if score > bestScore {
			bestScore = score
			bestMoveIdx = i
		}
	}

	return validMoves[bestMoveIdx], nil
}

// chooseRandomMove selects a random valid move
func chooseRandomMove(state GameState) Move {
	validMoves := getValidMoves(state)
	if len(validMoves) == 0 {
		// Should not happen if the game state is valid, but just in case
		return Move{CardIndex: 0, Position: 0}
	}
	return validMoves[rand.Intn(len(validMoves))]
}
