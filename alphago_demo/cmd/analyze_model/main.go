package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/analysis"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

type benchmarkPosition struct {
	Name        string
	Description string
	Game        *game.RPSGame
}

// loadBenchmarkPositions loads a set of predefined benchmark positions
func loadBenchmarkPositions() []benchmarkPosition {
	positions := []benchmarkPosition{
		createEarlyGamePosition(),
		createMidGamePosition(),
		createEndGamePosition(),
	}

	return positions
}

// createEarlyGamePosition creates a benchmark position representative of early game
func createEarlyGamePosition() benchmarkPosition {
	g := game.NewRPSGame(21, 5, 10)

	// Set up an early game position with some cards played
	// Board:
	//   0 1 2
	// 0 R . .
	// 1 . P .
	// 2 . . S

	// Clear the board first
	for i := range g.Board {
		g.Board[i] = game.RPSCard{Type: game.Rock, Owner: game.NoPlayer}
	}

	// Set up the board with just a few moves played
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 0,0: R
	g.Board[4] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Position 1,1: p
	g.Board[8] = game.RPSCard{Type: game.Scissors, Owner: game.Player1} // Position 2,2: S

	// Set hands (3 cards each)
	g.Player1Hand = []game.RPSCard{
		{Type: game.Rock, Owner: game.NoPlayer},
		{Type: game.Paper, Owner: game.NoPlayer},
		{Type: game.Scissors, Owner: game.NoPlayer},
	}

	g.Player2Hand = []game.RPSCard{
		{Type: game.Rock, Owner: game.NoPlayer},
		{Type: game.Scissors, Owner: game.NoPlayer},
		{Type: game.Paper, Owner: game.NoPlayer},
	}

	// Set current player to Player1
	g.CurrentPlayer = game.Player1
	g.Round = 3

	return benchmarkPosition{
		Name:        "Early Game",
		Description: "Few cards played, many options remaining",
		Game:        g,
	}
}

// createMidGamePosition creates a benchmark position representative of midgame
func createMidGamePosition() benchmarkPosition {
	g := game.NewRPSGame(21, 5, 10)

	// Set up a midgame position with about half the cards played
	// Board:
	//   0 1 2
	// 0 R p .
	// 1 s P R
	// 2 . . S

	// Clear the board first
	for i := range g.Board {
		g.Board[i] = game.RPSCard{Type: game.Rock, Owner: game.NoPlayer}
	}

	// Set up the board
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 0,0: R
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Position 0,1: p
	g.Board[3] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 1,0: s
	g.Board[4] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 1,1: P
	g.Board[5] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 1,2: R
	g.Board[8] = game.RPSCard{Type: game.Scissors, Owner: game.Player1} // Position 2,2: S

	// Set hands (2 cards for Player1, 1 for Player2)
	g.Player1Hand = []game.RPSCard{
		{Type: game.Rock, Owner: game.NoPlayer},
		{Type: game.Paper, Owner: game.NoPlayer},
	}

	g.Player2Hand = []game.RPSCard{
		{Type: game.Scissors, Owner: game.NoPlayer},
	}

	// Set current player to Player2
	g.CurrentPlayer = game.Player2
	g.Round = 5

	return benchmarkPosition{
		Name:        "Midgame",
		Description: "About half the cards played, tactical decisions important",
		Game:        g,
	}
}

// createEndGamePosition creates a benchmark position representative of endgame
func createEndGamePosition() benchmarkPosition {
	g := game.NewRPSGame(21, 5, 10)

	// Set up an endgame position with most cards played
	// Board:
	//   0 1 2
	// 0 r P R
	// 1 s R p
	// 2 P r .

	// Clear the board first
	for i := range g.Board {
		g.Board[i] = game.RPSCard{Type: game.Rock, Owner: game.NoPlayer}
	}

	// Set up the board
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 0,0: r
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 0,1: P
	g.Board[2] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 0,2: R
	g.Board[3] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Position 1,0: s
	g.Board[4] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Position 1,1: R
	g.Board[5] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Position 1,2: p
	g.Board[6] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Position 2,0: P
	g.Board[7] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Position 2,1: r

	// One card left for each player
	g.Player1Hand = []game.RPSCard{
		{Type: game.Scissors, Owner: game.NoPlayer},
	}

	g.Player2Hand = []game.RPSCard{
		{Type: game.Scissors, Owner: game.NoPlayer},
	}

	// Set current player to Player1
	g.CurrentPlayer = game.Player1
	g.Round = 8

	return benchmarkPosition{
		Name:        "Endgame",
		Description: "Final moves, outcome nearly determined",
		Game:        g,
	}
}

func main() {
	// Parse command line arguments
	modelPath := flag.String("model", "", "Path to model file (policy network)")
	depth := flag.Int("depth", 5, "Minimax search depth")
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	outputPath := flag.String("output", "", "Output file for analysis report")

	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: Model path is required")
		flag.Usage()
		os.Exit(1)
	}

	// Load benchmark positions
	fmt.Println("Loading benchmark positions...")
	positions := loadBenchmarkPositions()

	// Load model
	fmt.Printf("Loading model from %s...\n", *modelPath)
	model, err := neural.LoadPolicyNetwork(*modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}

	// Initialize minimax engine
	minimaxEngine := analysis.NewMinimaxEngine(*depth, analysis.StandardEvaluator)

	// Analyze each position
	fmt.Printf("Analyzing positions with minimax at depth %d...\n", *depth)

	// Prepare results for output file
	analysisResults := make(map[string]interface{})
	positionResults := make([]map[string]interface{}, 0, len(positions))

	for i, position := range positions {
		fmt.Printf("\n[%d/%d] Analyzing position: %s\n", i+1, len(positions), position.Name)
		if *verbose {
			fmt.Println(position.Game.String())
		}

		// Use minimax to find the best move
		startTime := time.Now()
		bestMove, bestValue := minimaxEngine.FindBestMove(position.Game)
		minimaxTime := time.Since(startTime)

		// Get model's prediction
		modelMove, err := getModelMove(model, position.Game)
		if err != nil {
			fmt.Printf("Error getting model move: %v\n", err)
			continue
		}

		// Print results
		fmt.Printf("Minimax best move: %v (value: %.2f, time: %v, nodes: %d)\n",
			formatMove(bestMove), bestValue, minimaxTime, minimaxEngine.NodesEvaluated)
		fmt.Printf("Model's move: %v\n", formatMove(modelMove))

		// Check if model's move matches minimax
		matches := moveEquals(bestMove, modelMove)
		if matches {
			fmt.Println("✓ Model's move matches minimax!")
		} else {
			fmt.Println("✗ Model's move differs from minimax")
		}

		// Add to results for output file
		positionResult := map[string]interface{}{
			"position_name":   position.Name,
			"minimax_move":    formatMove(bestMove),
			"minimax_value":   bestValue,
			"minimax_nodes":   minimaxEngine.NodesEvaluated,
			"minimax_time_ms": minimaxTime.Milliseconds(),
			"model_move":      formatMove(modelMove),
			"matches_minimax": matches,
		}
		positionResults = append(positionResults, positionResult)

		// Show board after model's move if verbose
		if *verbose {
			gameCopy := position.Game.Copy()
			modelMoveCopy := modelMove
			modelMoveCopy.Player = gameCopy.CurrentPlayer

			err := gameCopy.MakeMove(modelMoveCopy)
			if err != nil {
				fmt.Printf("Error making model move: %v\n", err)
			} else {
				fmt.Println("\nBoard after model's move:")
				fmt.Println(gameCopy.String())
			}
		}
	}

	// Save results to output file if specified
	if *outputPath != "" {
		analysisResults["positions"] = positionResults
		analysisResults["model_path"] = *modelPath
		analysisResults["minimax_depth"] = *depth
		analysisResults["timestamp"] = time.Now().Format(time.RFC3339)

		err := saveResultsToFile(*outputPath, analysisResults)
		if err != nil {
			fmt.Printf("Error saving results to file: %v\n", err)
		} else {
			fmt.Printf("\nAnalysis results saved to %s\n", *outputPath)
		}
	}
}

// saveResultsToFile saves analysis results to a JSON file
func saveResultsToFile(filename string, results map[string]interface{}) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(results)
}

// getModelMove gets the move predicted by the policy network
func getModelMove(model *neural.RPSPolicyNetwork, gameState *game.RPSGame) (game.RPSMove, error) {
	// Get valid moves
	validMoves := gameState.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}

	// Get model's predictions directly using the RPSPolicyNetwork's Predict method
	predictions := model.Predict(gameState)

	// Find highest probability valid move
	bestScore := -1.0
	var bestMove game.RPSMove

	for _, move := range validMoves {
		// Convert move to index in prediction array
		moveIndex := move.Position

		if predictions[moveIndex] > bestScore {
			bestScore = predictions[moveIndex]
			bestMove = move
		}
	}

	return bestMove, nil
}

// formatMove formats a move for display
func formatMove(move game.RPSMove) string {
	row := move.Position / 3
	col := move.Position % 3
	return fmt.Sprintf("(%d,%d)", row, col)
}

// moveEquals checks if two moves are the same
func moveEquals(move1, move2 game.RPSMove) bool {
	return move1.Position == move2.Position
}
