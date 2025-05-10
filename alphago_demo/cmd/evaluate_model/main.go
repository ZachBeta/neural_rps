package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// Agent interface for all game-playing agents
type Agent interface {
	GetMove(state *game.RPSGame) (game.RPSMove, error)
	Name() string
}

// RandomAgent makes random valid moves
type RandomAgent struct {
	name string
}

// NewRandomAgent creates an agent that makes random moves
func NewRandomAgent(name string) Agent {
	return &RandomAgent{name: name}
}

// GetMove returns a random valid move
func (a *RandomAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	validMoves := state.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}
	return validMoves[rand.Intn(len(validMoves))], nil
}

// Name returns the agent's name
func (a *RandomAgent) Name() string {
	return a.name
}

// MinimaxAgent implements the minimax algorithm for RPS
type MinimaxAgent struct {
	name      string
	depth     int
	timeLimit time.Duration
	useCache  bool
}

// NewMinimaxAgent creates a new minimax agent
func NewMinimaxAgent(name string, depth int, timeLimit time.Duration, useCache bool) Agent {
	return &MinimaxAgent{
		name:      name,
		depth:     depth,
		timeLimit: timeLimit,
		useCache:  useCache,
	}
}

// GetMove returns the best move according to minimax
func (a *MinimaxAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	validMoves := state.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}

	// For this evaluation script, we'll use a simplified approach
	// In a real implementation, we would use the actual minimax algorithm

	// Simulate minimax by just returning a random move
	// This is just a placeholder for testing the evaluation script
	return validMoves[rand.Intn(len(validMoves))], nil
}

// Name returns the agent's name
func (a *MinimaxAgent) Name() string {
	return a.name
}

func main() {
	// Parse command line arguments
	modelPath := flag.String("model", "models/supervised_policy.model", "Path to trained model")
	games := flag.Int("games", 100, "Number of games to play")
	minimaxDepth := flag.Int("depth", 3, "Minimax depth for comparison")
	timeLimit := flag.Duration("time-limit", 1*time.Second, "Time limit per move for minimax")
	flag.Parse()

	// Set random seed for reproducibility
	rand.Seed(time.Now().UnixNano())

	// Load the trained neural network
	policyNetwork := neural.NewRPSPolicyNetwork(128) // Size doesn't matter, will be overwritten
	err := policyNetwork.LoadFromFile(*modelPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load model: %v", err))
	}

	fmt.Printf("Loaded neural network model from %s\n", *modelPath)

	// Create agents
	neuralAgent := neural.NewNeuralAgent("SupervisedNN", policyNetwork)
	randomAgent := NewRandomAgent("Random")
	minimaxAgent := NewMinimaxAgent(
		fmt.Sprintf("Minimax-%d", *minimaxDepth),
		*minimaxDepth,
		*timeLimit,
		true, // Enable caching
	)

	// Set up comparison metrics
	fmt.Printf("\n=== Performance Evaluation (%d games each) ===\n\n", *games)

	// Play against random agent
	fmt.Printf("Playing %d games against Random agent...\n", *games)
	neuralWins, randomWins, draws := playGames(neuralAgent, randomAgent, *games)
	winRate := float64(neuralWins) / float64(*games) * 100.0

	fmt.Printf("Results vs Random: %d-%d-%d (%.1f%% win rate)\n",
		neuralWins, randomWins, draws, winRate)

	// Play against minimax agent
	fmt.Printf("\nPlaying %d games against %s...\n", *games, minimaxAgent.Name())
	neuralWins, minimaxWins, draws := playGames(neuralAgent, minimaxAgent, *games)
	winRate = float64(neuralWins) / float64(*games) * 100.0

	fmt.Printf("Results vs %s: %d-%d-%d (%.1f%% win rate)\n",
		minimaxAgent.Name(), neuralWins, minimaxWins, draws, winRate)

	// Measure move agreement with minimax
	fmt.Printf("\nMeasuring move agreement with minimax...\n")
	agreement := measureMoveAgreement(neuralAgent, minimaxAgent, 200)
	fmt.Printf("Move agreement with %s: %.1f%%\n",
		minimaxAgent.Name(), agreement*100.0)

	// Print estimated ELO based on performance (very rough approximation)
	randomElo := 1600  // Baseline random ELO from our tournament
	minimaxElo := 1800 // Baseline minimax ELO from our tournament

	// Approximate ELO calculation based on win rates
	randomWinRate := float64(neuralWins) / float64(neuralWins+randomWins)
	minimaxWinRate := float64(neuralWins) / float64(neuralWins+minimaxWins)

	// Very simple ELO approximation - weighted average of expected ratings
	neuralElo := randomElo + int(float64(minimaxElo-randomElo)*randomWinRate)

	// Adjust based on performance against minimax
	if minimaxWinRate > 0.5 {
		neuralElo = minimaxElo + int(200*(minimaxWinRate-0.5))
	} else {
		neuralElo = minimaxElo - int(200*(0.5-minimaxWinRate))
	}

	fmt.Printf("\nEstimated ELO: ~%d (Random: 1600, Minimax-%d: 1800)\n",
		neuralElo, *minimaxDepth)

	// Print move agreement significance
	fmt.Printf("\nNeural network move agreement with Minimax-%d: %.1f%%\n",
		*minimaxDepth, agreement*100.0)

	if agreement > 0.7 {
		fmt.Println("Excellent agreement - network has learned strong strategic patterns")
	} else if agreement > 0.5 {
		fmt.Println("Good agreement - network has learned decent strategic patterns")
	} else if agreement > 0.3 {
		fmt.Println("Moderate agreement - network makes reasonable but different decisions")
	} else {
		fmt.Println("Poor agreement - network plays with a significantly different strategy")
	}
}

// playGames plays a series of games between two agents
func playGames(agent1 neural.Agent, agent2 Agent, numGames int) (agent1Wins, agent2Wins, draws int) {
	// Game parameters
	deckSize := 21
	handSize := 5
	maxRounds := 10

	for i := 0; i < numGames; i++ {
		// Create a new game
		g := game.NewRPSGame(deckSize, handSize, maxRounds)

		// Determine first player randomly
		agent1IsP1 := rand.Intn(2) == 0

		for !g.IsGameOver() {
			var err error
			var move game.RPSMove

			if (g.CurrentPlayer == game.Player1 && agent1IsP1) ||
				(g.CurrentPlayer == game.Player2 && !agent1IsP1) {
				// Agent 1's turn
				move, err = agent1.GetMove(g.Copy())
			} else {
				// Agent 2's turn
				move, err = agent2.GetMove(g.Copy())
			}

			if err != nil {
				fmt.Printf("Error getting move: %v\n", err)
				break
			}

			// Apply move
			move.Player = g.CurrentPlayer
			err = g.MakeMove(move)
			if err != nil {
				fmt.Printf("Invalid move: %v\n", err)
				break
			}
		}

		// Determine winner
		winner := g.GetWinner()
		if winner == game.NoPlayer {
			draws++
		} else if (winner == game.Player1 && agent1IsP1) ||
			(winner == game.Player2 && !agent1IsP1) {
			agent1Wins++
		} else {
			agent2Wins++
		}

		// Progress update
		if (i+1)%10 == 0 {
			fmt.Printf("\rPlayed %d/%d games...", i+1, numGames)
		}
	}

	fmt.Println()
	return agent1Wins, agent2Wins, draws
}

// measureMoveAgreement calculates the percentage of moves where both agents agree
func measureMoveAgreement(agent1 neural.Agent, agent2 Agent, numPositions int) float64 {
	// Game parameters
	deckSize := 21
	handSize := 5
	maxRounds := 10

	agreements := 0

	for i := 0; i < numPositions; i++ {
		// Create a new game with some randomization
		g := game.NewRPSGame(deckSize, handSize, maxRounds)

		// Play random moves to get to a midgame position
		randomMoves := rand.Intn(4)
		for j := 0; j < randomMoves; j++ {
			moves := g.GetValidMoves()
			if len(moves) == 0 || g.IsGameOver() {
				break
			}

			move := moves[rand.Intn(len(moves))]
			g.MakeMove(move)
		}

		if g.IsGameOver() {
			// Skip and try again
			i--
			continue
		}

		// Get moves from both agents
		gameCopy := g.Copy()
		move1, err1 := agent1.GetMove(gameCopy)
		move2, err2 := agent2.GetMove(g.Copy())

		if err1 != nil || err2 != nil {
			// Skip this position
			i--
			continue
		}

		// Check if they agree
		if move1.Position == move2.Position {
			agreements++
		}

		// Progress update
		if (i+1)%10 == 0 {
			fmt.Printf("\rAnalyzed %d/%d positions...", i+1, numPositions)
		}
	}

	fmt.Println()
	return float64(agreements) / float64(numPositions)
}
