package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/agents"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

type gameResult struct {
	minimaxMove   game.RPSMove
	neuralMove    game.RPSMove
	minimaxEval   float64
	agreement     bool
	evalDiff      float64
	phase         string
	boardState    string
	minimaxPlayer game.RPSPlayer
}

func main() {
	// Parse command line arguments
	modelPath := flag.String("model", "", "Path to neural network model file")
	depth := flag.Int("depth", 5, "Minimax search depth")
	timeLimit := flag.Int("time", 3, "Time limit for minimax in seconds")
	games := flag.Int("games", 30, "Number of games to play")
	outFile := flag.String("out", "", "Output file for detailed results (optional)")
	useCache := flag.Bool("cache", true, "Enable transposition table")
	flag.Parse()

	// Validate arguments
	if *modelPath == "" {
		fmt.Println("Error: Model path is required")
		flag.Usage()
		os.Exit(1)
	}

	// Load neural network model
	fmt.Printf("Loading neural network from %s\n", *modelPath)
	policyNetwork, err := neural.LoadPolicyNetwork(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load neural network: %v", err)
	}

	// Create neural network agent
	neuralAgent := neural.NewNeuralAgent("Neural", policyNetwork)

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

	results := runTournament(neuralAgent, minimaxAgent, *games)

	// Analyze results
	analyzeResults(results, *outFile)

	// Print minimax agent stats
	avgTime, totalPositions, avgPositionsPerMove := minimaxAgent.GetStats()
	fmt.Printf("\nMinimax agent stats:\n")
	fmt.Printf("  Average time per move: %v\n", avgTime)
	fmt.Printf("  Total positions evaluated: %d\n", totalPositions)
	fmt.Printf("  Average positions per move: %.1f\n", avgPositionsPerMove)
}

// runTournament plays a series of games between two agents
func runTournament(neuralAgent neural.Agent, minimaxAgent *agents.MinimaxAgent, numGames int) []gameResult {
	results := make([]gameResult, 0, numGames*9) // Each game has ~9 moves

	// Set up game parameters
	deckSize := 21
	handSize := 5
	maxRounds := 10

	// Random seed for game generation
	rand.Seed(time.Now().UnixNano())

	// Track wins and game outcomes
	neuralWins := 0
	minimaxWins := 0
	draws := 0

	for i := 0; i < numGames; i++ {
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
			i+1, numGames, playerToString(minimaxPlayer))
		fmt.Println(g.String())

		// Play the game
		moveNum := 0
		for !g.IsGameOver() {
			moveNum++

			// Determine whose turn it is and get that agent's move
			var move game.RPSMove
			var err error

			// Capture the board state for analysis
			boardState := g.String()
			phase := getGamePhase(g, moveNum)

			// Track neural and minimax choices for comparison
			var minimaxMove, neuralMove game.RPSMove
			var minimaxEval float64

			if g.CurrentPlayer == minimaxPlayer {
				// Minimax agent's turn
				minimaxMove, err = minimaxAgent.GetMove(g)
				if err != nil {
					log.Fatalf("Error getting minimax move: %v", err)
				}

				// Also get neural agent's move for comparison
				neuralMove, err = neuralAgent.GetMove(g)
				if err != nil {
					log.Fatalf("Error getting neural move: %v", err)
				}

				move = minimaxMove
			} else {
				// Neural agent's turn
				neuralMove, err = neuralAgent.GetMove(g)
				if err != nil {
					log.Fatalf("Error getting neural move: %v", err)
				}

				// Also get minimax agent's move for comparison
				minimaxMove, err = minimaxAgent.GetMove(g)
				if err != nil {
					log.Fatalf("Error getting minimax move: %v", err)
				}

				move = neuralMove
			}

			// For the evaluation difference, we need to obtain the actual evaluation
			// This is stored in the last field of the minimax move string
			minimaxEvalStr := fmt.Sprintf("%v", minimaxMove)
			lastCommaIndex := strings.LastIndex(minimaxEvalStr, ",")
			if lastCommaIndex != -1 {
				fmt.Sscanf(minimaxEvalStr[lastCommaIndex+1:], "%f", &minimaxEval)
			}

			// Record move agreement
			agreement := minimaxMove.Position == neuralMove.Position
			evalDiff := 0.0 // We'll need additional logic to compute this properly

			// Record the result
			result := gameResult{
				minimaxMove:   minimaxMove,
				neuralMove:    neuralMove,
				minimaxEval:   minimaxEval,
				agreement:     agreement,
				evalDiff:      evalDiff,
				phase:         phase,
				boardState:    boardState,
				minimaxPlayer: minimaxPlayer,
			}
			results = append(results, result)

			// Apply the move
			err = g.MakeMove(move)
			if err != nil {
				log.Fatalf("Error applying move: %v", err)
			}

			// Print move details
			fmt.Printf("Move %d by %s: %v\n",
				moveNum, playerToString(g.CurrentPlayer), move)
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
	fmt.Printf("Games played: %d\n", numGames)
	fmt.Printf("%s wins: %d (%.1f%%)\n",
		minimaxAgent.Name(), minimaxWins, float64(minimaxWins)/float64(numGames)*100)
	fmt.Printf("%s wins: %d (%.1f%%)\n",
		neuralAgent.Name(), neuralWins, float64(neuralWins)/float64(numGames)*100)
	fmt.Printf("Draws: %d (%.1f%%)\n",
		draws, float64(draws)/float64(numGames)*100)

	return results
}

// analyzeResults analyzes tournament results and prints metrics
func analyzeResults(results []gameResult, outFile string) {
	// Track agreement stats
	totalMoves := len(results)
	agreementCount := 0

	// Track game phase stats
	phaseAgreement := make(map[string]struct{ count, total int })

	// Analyze agreement rate
	for _, result := range results {
		if result.agreement {
			agreementCount++
		}

		// Update phase stats
		phaseStats := phaseAgreement[result.phase]
		phaseStats.total++
		if result.agreement {
			phaseStats.count++
		}
		phaseAgreement[result.phase] = phaseStats
	}

	// Calculate agreement percentage
	agreementPct := 0.0
	if totalMoves > 0 {
		agreementPct = float64(agreementCount) / float64(totalMoves) * 100
	}

	// Print agreement stats
	fmt.Printf("\n=== Move Agreement Analysis ===\n")
	fmt.Printf("Total moves analyzed: %d\n", totalMoves)
	fmt.Printf("Moves where neural network agreed with minimax: %d (%.1f%%)\n",
		agreementCount, agreementPct)

	// Print phase analysis
	fmt.Printf("\n=== Game Phase Analysis ===\n")
	for phase, stats := range phaseAgreement {
		phasePct := 0.0
		if stats.total > 0 {
			phasePct = float64(stats.count) / float64(stats.total) * 100
		}
		fmt.Printf("%s phase: %d/%d moves agree (%.1f%%)\n",
			phase, stats.count, stats.total, phasePct)
	}

	// Write detailed results to file if requested
	if outFile != "" {
		writeResultsToFile(results, outFile)
	}
}

// writeResultsToFile writes detailed results to a CSV file
func writeResultsToFile(results []gameResult, outFile string) {
	f, err := os.Create(outFile)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer f.Close()

	// Write header
	fmt.Fprintln(f, "Phase,MinimaxPlayer,MinimaxMove,NeuralMove,Agreement,MinimaxEval,EvalDiff,BoardState")

	// Write results
	for _, r := range results {
		fmt.Fprintf(f, "%s,%s,%v,%v,%v,%.2f,%.2f,%s\n",
			r.phase,
			playerToString(r.minimaxPlayer),
			r.minimaxMove,
			r.neuralMove,
			r.agreement,
			r.minimaxEval,
			r.evalDiff,
			strings.ReplaceAll(r.boardState, "\n", "|"))
	}

	fmt.Printf("Detailed results written to %s\n", outFile)
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
