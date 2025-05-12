package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/agents"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

func main() {
	// Parse command line arguments
	depth := flag.Int("depth", 5, "Minimax search depth")
	timeLimit := flag.Int("time", 3, "Time limit for minimax in seconds")
	games := flag.Int("games", 30, "Number of games to play")
	outFile := flag.String("out", "", "Output file for detailed results (optional)")
	useCache := flag.Bool("cache", true, "Enable transposition table")
	hiddenSize := flag.Int("hidden", 64, "Neural network hidden layer size")
	flag.Parse()

	// Create neural network agent with a newly initialized network
	neuralAgent := neural.NewNeuralAgent(
		fmt.Sprintf("Neural-H%d", *hiddenSize),
		neural.NewRPSPolicyNetwork(*hiddenSize),
	)

	// Create minimax agent
	minimaxName := fmt.Sprintf("Minimax-%d", *depth)
	minimaxAgent := agents.NewMinimaxAgent(
		minimaxName,
		*depth,
		time.Duration(*timeLimit)*time.Second,
		*useCache,
	)

	// Play games
	fmt.Printf("Starting tournament: %s vs %s (%d games)\n",
		neuralAgent.Name(), minimaxAgent.Name(), *games)

	// Configure game parameters
	deckSize := 21
	handSize := 5
	maxRounds := 10

	// Random seed for game generation
	rand.Seed(time.Now().UnixNano())

	// Track wins and game outcomes
	neuralWins := 0
	minimaxWins := 0
	draws := 0
	moveAgreements := 0
	totalMoves := 0

	for i := 0; i < *games; i++ {
		// Create a new game
		g := game.NewRPSGame(deckSize, handSize, maxRounds)

		// Determine first player randomly for fairness
		minimaxIsP1 := rand.Intn(2) == 0
		var minimaxPlayer, neuralPlayer game.RPSPlayer
		if minimaxIsP1 {
			minimaxPlayer = game.Player1
			neuralPlayer = game.Player2
		} else {
			minimaxPlayer = game.Player2
			neuralPlayer = game.Player1
		}

		fmt.Printf("\nGame %d of %d (Minimax plays as %s)\n",
			i+1, *games, playerToString(minimaxPlayer))
		fmt.Println(g.String())

		// Play the game
		moveNum := 0
		for !g.IsGameOver() {
			moveNum++
			totalMoves++

			// Determine whose turn it is and get that agent's move
			var move game.RPSMove
			var err error

			// Capture the board state for analysis
			phase := getGamePhase(g, moveNum)

			// Track neural and minimax choices for comparison
			var minimaxMove, neuralMove game.RPSMove

			if g.CurrentPlayer == minimaxPlayer {
				// Minimax agent's turn
				minimaxMove, err = minimaxAgent.GetMove(g)
				if err != nil {
					fmt.Printf("Error getting minimax move: %v\n", err)
					break
				}

				// Also get neural agent's move for comparison
				neuralMove, err = neuralAgent.GetMove(g)
				if err != nil {
					fmt.Printf("Error getting neural move: %v\n", err)
					break
				}

				move = minimaxMove
				fmt.Printf("[%s] %s plays: %v\n", phase, minimaxAgent.Name(), move)
			} else {
				// Neural agent's turn
				neuralMove, err = neuralAgent.GetMove(g)
				if err != nil {
					fmt.Printf("Error getting neural move: %v\n", err)
					break
				}

				// Also get minimax agent's move for comparison
				minimaxMove, err = minimaxAgent.GetMove(g)
				if err != nil {
					fmt.Printf("Error getting minimax move: %v\n", err)
					break
				}

				move = neuralMove
				fmt.Printf("[%s] %s plays: %v\n", phase, neuralAgent.Name(), move)
			}

			// Record move agreement
			agreement := minimaxMove.Position == neuralMove.Position
			if agreement {
				moveAgreements++
				fmt.Printf("  Move agreement! Both chose position %d\n", minimaxMove.Position)
			} else {
				fmt.Printf("  Moves differ: Minimax chose %d, Neural chose %d\n",
					minimaxMove.Position, neuralMove.Position)
			}

			// Apply the move
			err = g.MakeMove(move)
			if err != nil {
				fmt.Printf("Error applying move: %v\n", err)
				break
			}
		}

		// Determine winner
		winner := g.GetWinner()
		switch winner {
		case minimaxPlayer:
			minimaxWins++
			fmt.Printf("Game %d result: %s wins!\n", i+1, minimaxAgent.Name())
		case neuralPlayer:
			neuralWins++
			fmt.Printf("Game %d result: %s wins!\n", i+1, neuralAgent.Name())
		default:
			draws++
			fmt.Printf("Game %d result: Draw\n", i+1)
		}
	}

	// Print tournament results
	fmt.Printf("\n=== Tournament Results ===\n")
	fmt.Printf("Games played: %d\n", *games)
	fmt.Printf("%s wins: %d (%.1f%%)\n",
		minimaxAgent.Name(), minimaxWins, float64(minimaxWins)/float64(*games)*100)
	fmt.Printf("%s wins: %d (%.1f%%)\n",
		neuralAgent.Name(), neuralWins, float64(neuralWins)/float64(*games)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n",
		draws, float64(draws)/float64(*games)*100)

	// Print agreement stats
	fmt.Printf("\n=== Move Agreement Analysis ===\n")
	fmt.Printf("Total moves: %d\n", totalMoves)
	fmt.Printf("Moves where neural network agreed with minimax: %d (%.1f%%)\n",
		moveAgreements, float64(moveAgreements)/float64(totalMoves)*100)

	// Print minimax agent stats
	avgTime, totalPositions, avgPositionsPerMove := minimaxAgent.GetStats()
	fmt.Printf("\nMinimax agent stats:\n")
	fmt.Printf("  Average time per move: %v\n", avgTime)
	fmt.Printf("  Total positions evaluated: %d\n", totalPositions)
	fmt.Printf("  Average positions per move: %.1f\n", avgPositionsPerMove)

	// Output a summary if an output file is specified
	if *outFile != "" {
		f, err := os.Create(*outFile)
		if err != nil {
			fmt.Printf("Error creating output file: %v\n", err)
		} else {
			defer f.Close()
			fmt.Fprintf(f, "Tournament: %s vs %s\n", minimaxAgent.Name(), neuralAgent.Name())
			fmt.Fprintf(f, "Games: %d\n", *games)
			fmt.Fprintf(f, "Minimax wins: %d (%.1f%%)\n",
				minimaxWins, float64(minimaxWins)/float64(*games)*100)
			fmt.Fprintf(f, "Neural wins: %d (%.1f%%)\n",
				neuralWins, float64(neuralWins)/float64(*games)*100)
			fmt.Fprintf(f, "Draws: %d (%.1f%%)\n",
				draws, float64(draws)/float64(*games)*100)
			fmt.Fprintf(f, "Move agreement: %d/%d (%.1f%%)\n",
				moveAgreements, totalMoves, float64(moveAgreements)/float64(totalMoves)*100)
			fmt.Printf("Summary written to %s\n", *outFile)
		}
	}
}

// getGamePhase determines the current phase of the game
func getGamePhase(g *game.RPSGame, moveNum int) string {
	// Count cards on board
	cardsOnBoard := 0
	for _, card := range g.Board {
		if card.Owner != game.NoPlayer {
			cardsOnBoard++
		}
	}

	if cardsOnBoard <= 2 {
		return "Opening"
	} else if cardsOnBoard >= 7 {
		return "Endgame"
	} else {
		return "Midgame"
	}
}

// playerToString converts a player enum to a string
func playerToString(player game.RPSPlayer) string {
	switch player {
	case game.Player1:
		return "Player1"
	case game.Player2:
		return "Player2"
	default:
		return "NoPlayer"
	}
}
