package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/agents"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

type TrainingExample struct {
	BoardState    []int   `json:"board_state"`    // Flattened board (9 positions)
	Player1Hand   []int   `json:"player1_hand"`   // Card types in P1's hand
	Player2Hand   []int   `json:"player2_hand"`   // Card types in P2's hand
	CurrentPlayer int     `json:"current_player"` // 1 or 2
	BestMove      int     `json:"best_move"`      // 0-8 position index
	Evaluation    float64 `json:"evaluation"`     // Minimax evaluation
	GamePhase     string  `json:"game_phase"`     // "opening", "midgame", "endgame"
	SearchDepth   int     `json:"search_depth"`   // Depth used for this position
}

func main() {
	// Parse command line flags
	numPositions := flag.Int("positions", 10000, "Number of positions to generate")
	minimaxDepth := flag.Int("depth", 5, "Minimax search depth")
	outputFile := flag.String("output", "training_data.json", "Output file path")
	timeLimit := flag.Duration("time-limit", 5*time.Second, "Time limit per move")
	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create output directory if it doesn't exist
	os.MkdirAll("data", 0755)
	outputPath := fmt.Sprintf("data/%s", *outputFile)

	// Open output file
	file, err := os.Create(outputPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to create output file: %v", err))
	}
	defer file.Close()

	// Create a minimax agent with specified depth and caching enabled
	minimaxAgent := agents.NewMinimaxAgent(
		fmt.Sprintf("Minimax-%d", *minimaxDepth),
		*minimaxDepth,
		*timeLimit,
		true, // Enable caching
	)

	// Statistics tracking
	startTime := time.Now()
	totalPositions := *numPositions
	positionsGenerated := 0

	// Game parameters
	deckSize := 21
	handSize := 5
	maxRounds := 10

	// Array to hold all examples
	examples := make([]TrainingExample, 0, totalPositions)

	fmt.Printf("Generating %d training examples using Minimax-%d...\n",
		totalPositions, *minimaxDepth)

	for positionsGenerated < totalPositions {
		// Create a new game
		g := game.NewRPSGame(deckSize, handSize, maxRounds)

		// Play a few random moves to get diverse positions
		playRandomMoves(g, 0, 4) // 0-4 random moves

		if g.IsGameOver() {
			continue // Skip completed games
		}

		// Get minimax move for this position
		move, err := minimaxAgent.GetMove(g)
		if err != nil {
			fmt.Printf("Error getting move: %v\n", err)
			continue
		}

		// Create training example
		example := createTrainingExample(g, move, *minimaxDepth)
		examples = append(examples, example)

		positionsGenerated++

		// Status update every 100 positions
		if positionsGenerated%100 == 0 {
			elapsed := time.Since(startTime)
			posPerSecond := float64(positionsGenerated) / elapsed.Seconds()
			fmt.Printf("Generated %d/%d positions (%.2f pos/sec)\n",
				positionsGenerated, totalPositions, posPerSecond)
		}
	}

	// Write data to file
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(examples); err != nil {
		panic(fmt.Sprintf("Failed to write training data: %v", err))
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\nCompleted! Generated %d positions in %v (%.2f pos/sec)\n",
		positionsGenerated, elapsed, float64(positionsGenerated)/elapsed.Seconds())
	fmt.Printf("Training data saved to %s\n", outputPath)
}

// playRandomMoves plays a random number of moves between min and max
func playRandomMoves(g *game.RPSGame, min, max int) {
	numMoves := min + rand.Intn(max-min+1)

	for i := 0; i < numMoves; i++ {
		moves := g.GetValidMoves()
		if len(moves) == 0 || g.IsGameOver() {
			return
		}

		move := moves[rand.Intn(len(moves))]
		g.MakeMove(move)
	}
}

// createTrainingExample converts a game state and minimax move to a training example
func createTrainingExample(g *game.RPSGame, move game.RPSMove, depth int) TrainingExample {
	// Create board state representation (flattened)
	boardState := make([]int, 9)
	for i, card := range g.Board {
		if card.Owner == game.NoPlayer {
			boardState[i] = 0 // Empty
		} else if card.Owner == game.Player1 {
			// Encode Player 1's cards as 1, 2, 3
			switch card.Type {
			case game.Rock:
				boardState[i] = 1
			case game.Paper:
				boardState[i] = 2
			case game.Scissors:
				boardState[i] = 3
			}
		} else {
			// Encode Player 2's cards as 4, 5, 6
			switch card.Type {
			case game.Rock:
				boardState[i] = 4
			case game.Paper:
				boardState[i] = 5
			case game.Scissors:
				boardState[i] = 6
			}
		}
	}

	// Create hand representations
	p1Hand := encodeHand(g.Player1Hand)
	p2Hand := encodeHand(g.Player2Hand)

	// Determine game phase
	phase := getGamePhase(g)

	// Current player (1 or 2)
	currentPlayer := 1
	if g.CurrentPlayer == game.Player2 {
		currentPlayer = 2
	}

	return TrainingExample{
		BoardState:    boardState,
		Player1Hand:   p1Hand,
		Player2Hand:   p2Hand,
		CurrentPlayer: currentPlayer,
		BestMove:      move.Position, // 0-8 position index
		Evaluation:    0.0,           // Fixed - we'll need to update the MinimaxAgent to expose this
		GamePhase:     phase,
		SearchDepth:   depth,
	}
}

// encodeHand converts a slice of cards to counts of each type
func encodeHand(hand []game.RPSCard) []int {
	counts := make([]int, 3) // Rock, Paper, Scissors

	for _, card := range hand {
		switch card.Type {
		case game.Rock:
			counts[0]++
		case game.Paper:
			counts[1]++
		case game.Scissors:
			counts[2]++
		}
	}

	return counts
}

// getGamePhase determines the current phase of the game
func getGamePhase(g *game.RPSGame) string {
	// Count cards on board
	cardsOnBoard := 0
	for _, card := range g.Board {
		if card.Owner != game.NoPlayer {
			cardsOnBoard++
		}
	}

	if cardsOnBoard <= 2 {
		return "opening"
	} else if cardsOnBoard >= 7 {
		return "endgame"
	} else {
		return "midgame"
	}
}
