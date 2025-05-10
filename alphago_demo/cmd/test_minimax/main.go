package main

import (
	"fmt"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/analysis"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
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
	// 1 . . .
	// 2 . . s

	// Place some cards on the board
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

	return benchmarkPosition{
		Name:        "Early Game",
		Description: "2 cards on board, 4 cards in each player's hand",
		Game:        g,
	}
}

// createMidGamePosition creates a benchmark position representative of mid-game
func createMidGamePosition() benchmarkPosition {
	g := game.NewRPSGame(21, 5, 10)

	// Set up a mid-game position with several cards played
	// Board:
	//   0 1 2
	// 0 R P s
	// 1 p S .
	// 2 r . P

	// Place cards on the board
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Player 1's Rock at (0,0)
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Player 1's Paper at (0,1)
	g.Board[2] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Player 2's Scissors at (0,2)
	g.Board[3] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Player 2's Paper at (1,0)
	g.Board[4] = game.RPSCard{Type: game.Scissors, Owner: game.Player1} // Player 1's Scissors at (1,1)
	g.Board[6] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Player 2's Rock at (2,0)
	g.Board[8] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Player 1's Paper at (2,2)

	// Set up player hands
	p1Hand := []game.RPSCard{
		{Type: game.Rock, Owner: game.Player1},
		{Type: game.Paper, Owner: game.Player1},
	}
	g.Player1Hand = p1Hand

	p2Hand := []game.RPSCard{
		{Type: game.Rock, Owner: game.Player2},
		{Type: game.Scissors, Owner: game.Player2},
	}
	g.Player2Hand = p2Hand

	// Set current player to Player 2
	g.CurrentPlayer = game.Player2

	return benchmarkPosition{
		Name:        "Mid Game",
		Description: "7 cards on board, 2 cards in each player's hand",
		Game:        g,
	}
}

// createEndGamePosition creates a benchmark position representative of end-game
func createEndGamePosition() benchmarkPosition {
	g := game.NewRPSGame(21, 5, 10)

	// Set up an end-game position with almost full board
	// Board:
	//   0 1 2
	// 0 R P s
	// 1 p S r
	// 2 r P .

	// Place cards on the board
	g.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}     // Player 1's Rock at (0,0)
	g.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Player 1's Paper at (0,1)
	g.Board[2] = game.RPSCard{Type: game.Scissors, Owner: game.Player2} // Player 2's Scissors at (0,2)
	g.Board[3] = game.RPSCard{Type: game.Paper, Owner: game.Player2}    // Player 2's Paper at (1,0)
	g.Board[4] = game.RPSCard{Type: game.Scissors, Owner: game.Player1} // Player 1's Scissors at (1,1)
	g.Board[5] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Player 2's Rock at (1,2)
	g.Board[6] = game.RPSCard{Type: game.Rock, Owner: game.Player2}     // Player 2's Rock at (2,0)
	g.Board[7] = game.RPSCard{Type: game.Paper, Owner: game.Player1}    // Player 1's Paper at (2,1)

	// Set up player hands - just one card each
	p1Hand := []game.RPSCard{
		{Type: game.Scissors, Owner: game.Player1},
	}
	g.Player1Hand = p1Hand

	p2Hand := []game.RPSCard{
		{Type: game.Paper, Owner: game.Player2},
	}
	g.Player2Hand = p2Hand

	// Set current player to Player 1
	g.CurrentPlayer = game.Player1

	return benchmarkPosition{
		Name:        "End Game",
		Description: "8 cards on board, 1 card in each player's hand",
		Game:        g,
	}
}

func main() {
	fmt.Println("=== Minimax Analyzer Test With Transposition Table ===")

	// Initialize positions
	positions := loadBenchmarkPositions()

	// Test with different depths
	depths := []int{2, 3, 4, 5, 6}

	for _, depth := range depths {
		fmt.Printf("\n=== Testing at depth %d ===\n", depth)

		// Create minimax engine with caching disabled
		minimaxWithoutCache := analysis.NewMinimaxEngine(depth, analysis.StandardEvaluator)

		// Create minimax engine with caching enabled
		minimaxWithCache := analysis.NewMinimaxEngine(depth, analysis.StandardEvaluator)
		minimaxWithCache.EnableTranspositionTable()

		// Test each position
		for i, position := range positions {
			fmt.Printf("\n[Position %d/%d] %s\n", i+1, len(positions), position.Name)
			fmt.Println(position.Game.String())

			// Test without cache
			fmt.Println("\nWithout caching:")
			startTime := time.Now()
			bestMove, bestValue := minimaxWithoutCache.FindBestMove(position.Game)
			elapsedWithoutCache := time.Since(startTime)
			fmt.Printf("Best move: %v (value: %.2f)\n", bestMove, bestValue)
			fmt.Printf("Time: %v\n", elapsedWithoutCache)
			fmt.Printf("Nodes evaluated: %d\n", minimaxWithoutCache.NodesEvaluated)
			fmt.Printf("Nodes per second: %.2f\n",
				float64(minimaxWithoutCache.NodesEvaluated)/elapsedWithoutCache.Seconds())

			// Test with cache
			fmt.Println("\nWith caching:")
			startTime = time.Now()
			bestMove, bestValue = minimaxWithCache.FindBestMove(position.Game)
			elapsedWithCache := time.Since(startTime)
			fmt.Printf("Best move: %v (value: %.2f)\n", bestMove, bestValue)
			fmt.Printf("Time: %v\n", elapsedWithCache)
			fmt.Printf("Nodes evaluated: %d\n", minimaxWithCache.NodesEvaluated)

			hits, misses, hitRate := minimaxWithCache.GetCacheStats()
			fmt.Printf("Cache stats - Hits: %d, Misses: %d, Hit rate: %.2f%%\n",
				hits, misses, hitRate)
			fmt.Printf("Nodes per second: %.2f\n",
				float64(minimaxWithCache.NodesEvaluated)/elapsedWithCache.Seconds())

			// Calculate speedup
			speedup := float64(elapsedWithoutCache) / float64(elapsedWithCache)
			fmt.Printf("\nSpeedup with cache: %.2fx\n", speedup)

			// Reset cache for next position
			minimaxWithCache.DisableTranspositionTable()
			minimaxWithCache.EnableTranspositionTable()
		}
	}

	// Now run a stress test to demonstrate cache effectiveness
	fmt.Println("\n\n=== Cache Efficiency Stress Test ===")
	stressTest()
}

// stressTest repeatedly searches the same position to show cache effectiveness
func stressTest() {
	// Create a complex mid-game position
	position := createMidGamePosition()
	fmt.Printf("Position: %s\n", position.Name)
	fmt.Println(position.Game.String())

	// Set up minimax with and without cache
	depth := 5
	minimaxWithoutCache := analysis.NewMinimaxEngine(depth, analysis.StandardEvaluator)
	minimaxWithCache := analysis.NewMinimaxEngine(depth, analysis.StandardEvaluator)
	minimaxWithCache.EnableTranspositionTable()

	iterations := 10
	fmt.Printf("\nRunning %d iterations at depth %d...\n\n", iterations, depth)

	// Test without cache
	startTotal := time.Now()
	for i := 0; i < iterations; i++ {
		fmt.Printf("Iteration %d without cache... ", i+1)
		start := time.Now()
		minimaxWithoutCache.FindBestMove(position.Game)
		elapsed := time.Since(start)
		fmt.Printf("Time: %v, Nodes: %d\n",
			elapsed, minimaxWithoutCache.NodesEvaluated)
	}
	totalWithoutCache := time.Since(startTotal)
	fmt.Printf("Total time without cache: %v\n", totalWithoutCache)

	// Test with cache
	startTotal = time.Now()
	for i := 0; i < iterations; i++ {
		fmt.Printf("Iteration %d with cache... ", i+1)
		start := time.Now()
		minimaxWithCache.FindBestMove(position.Game)
		elapsed := time.Since(start)
		hits, misses, hitRate := minimaxWithCache.GetCacheStats()
		fmt.Printf("Time: %v, Hits: %d, Misses: %d, Hit rate: %.2f%%\n",
			elapsed, hits, misses, hitRate)
	}
	totalWithCache := time.Since(startTotal)
	fmt.Printf("Total time with cache: %v\n", totalWithCache)

	// Overall speedup
	speedup := float64(totalWithoutCache) / float64(totalWithCache)
	fmt.Printf("\nOverall speedup with cache: %.2fx\n", speedup)
}
