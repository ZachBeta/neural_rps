package neat

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

// Match represents a single evaluation match for a genome.
type Match struct {
	GenomeIdx int     // index of genome being evaluated
	Opponent  *Genome // pointer to opponent (round robin or HOF)
	Games     int
	IsHOF     bool
}

// GenomeResult accumulates results for a genome in parallel evaluation.
type GenomeResult struct {
	Wins  int32
	Draws int32
	Games int32
}

// prepareMatches generates the list of evaluation matches for all genomes.
func prepareMatches(pop *Population, hof []*Genome) []Match {
	const (
		nRoundRobin = 5
		gamesPerRR  = 2
		gamesPerHOF = 5
	)
	matches := make([]Match, 0)
	for i := range pop.Genomes {
		// Round robin: pick nRoundRobin random opponents (excluding self)
		opponents := randomSubset(i, nRoundRobin, len(pop.Genomes))
		for _, oppIdx := range opponents {
			matches = append(matches, Match{
				GenomeIdx: i,
				Opponent:  pop.Genomes[oppIdx],
				Games:     gamesPerRR,
				IsHOF:     false,
			})
		}
		// Hall of Fame
		for _, hofGenome := range hof {
			matches = append(matches, Match{
				GenomeIdx: i,
				Opponent:  hofGenome,
				Games:     gamesPerHOF,
				IsHOF:     true,
			})
		}
	}
	return matches
}

// randomSubset returns n unique random indices in [0, max), excluding 'exclude'.
func randomSubset(exclude, n, max int) []int {
	indices := make([]int, 0, max-1)
	for i := 0; i < max; i++ {
		if i != exclude {
			indices = append(indices, i)
		}
	}
	if n > len(indices) {
		n = len(indices)
	}
	rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	return indices[:n]
}

// runGames runs 'games' matches between evalGenome and opponent, returns (wins, draws) for evalGenome.
func runGames(evalGenome, opponent *Genome, games int) (wins int, draws int) {
	params := training.DefaultRPSSelfPlayParams()
	deckSize, handSize, maxRounds := params.DeckSize, params.HandSize, params.MaxRounds
	mctsParams := params.MCTSParams

	// Build networks for each genome
	p1Pol, p1Val := evalGenome.ToNetworks()
	p2Pol, p2Val := opponent.ToNetworks()

	for i := 0; i < games; i++ {
		// Alternate who is player 1/2
		var (
			player1Pol  *neural.RPSPolicyNetwork
			player1Val  *neural.RPSValueNetwork
			player2Pol  *neural.RPSPolicyNetwork
			player2Val  *neural.RPSValueNetwork
			isEvalFirst bool
		)
		if i%2 == 0 {
			player1Pol, player1Val = p1Pol, p1Val
			player2Pol, player2Val = p2Pol, p2Val
			isEvalFirst = true
		} else {
			player1Pol, player1Val = p2Pol, p2Val
			player2Pol, player2Val = p1Pol, p1Val
			isEvalFirst = false
		}

		gme := game.NewRPSGame(deckSize, handSize, maxRounds)
		e1 := mcts.NewRPSMCTS(player1Pol, player1Val, mctsParams)
		e2 := mcts.NewRPSMCTS(player2Pol, player2Val, mctsParams)

		for !gme.IsGameOver() {
			if gme.CurrentPlayer == game.Player1 {
				e1.SetRootState(gme)
				if node := e1.Search(); node != nil && node.Move != nil {
					gme.MakeMove(*node.Move)
				}
			} else {
				e2.SetRootState(gme)
				if node := e2.Search(); node != nil && node.Move != nil {
					gme.MakeMove(*node.Move)
				}
			}
		}

		winner := gme.GetWinner()
		if winner == game.NoPlayer {
			draws++
		} else if (winner == game.Player1 && isEvalFirst) || (winner == game.Player2 && !isEvalFirst) {
			wins++
		}
	}
	return wins, draws
}

// parallelEvaluate evaluates all genomes in pop against round robin and HOF opponents in parallel.
func parallelEvaluate(pop *Population, hof []*Genome) []*GenomeResult {
	startTime := time.Now()
	matches := prepareMatches(pop, hof)
	matchCount := len(matches)
	fmt.Printf("Evaluating %d genomes with %d total matches (%d workers)...\n",
		len(pop.Genomes), matchCount, runtime.NumCPU()-1)

	results := make([]*GenomeResult, len(pop.Genomes))
	for i := range results {
		results[i] = &GenomeResult{}
	}

	numWorkers := runtime.NumCPU() - 1
	if numWorkers < 1 {
		numWorkers = 1
	}

	workCh := make(chan Match, len(matches))
	var wg sync.WaitGroup

	// Progress tracking
	completedMatches := int32(0)
	ticker := time.NewTicker(5 * time.Second)
	done := make(chan bool)

	// Start progress reporting goroutine
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				completed := atomic.LoadInt32(&completedMatches)
				percent := float64(completed) / float64(matchCount) * 100
				elapsed := time.Since(startTime)
				matchesPerSec := float64(completed) / elapsed.Seconds()

				// Simple progress bar
				width := 30
				progress := int(float64(width) * float64(completed) / float64(matchCount))
				bar := "["
				for i := 0; i < width; i++ {
					if i < progress {
						bar += "="
					} else if i == progress {
						bar += ">"
					} else {
						bar += " "
					}
				}
				bar += "]"

				fmt.Printf("\r%s %.1f%% (%d/%d) | %.1f matches/sec | %s ",
					bar, percent, completed, matchCount, matchesPerSec, elapsed.Round(time.Second))
			}
		}
	}()

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerId int) {
			defer wg.Done()
			for match := range workCh {
				wins, draws := runGames(pop.Genomes[match.GenomeIdx], match.Opponent, match.Games)
				atomic.AddInt32(&results[match.GenomeIdx].Wins, int32(wins))
				atomic.AddInt32(&results[match.GenomeIdx].Draws, int32(draws))
				atomic.AddInt32(&results[match.GenomeIdx].Games, int32(match.Games))

				// Update progress counter
				atomic.AddInt32(&completedMatches, 1)

				// We've removed the detailed worker logs here to reduce console noise
			}
		}(i)
	}

	for _, m := range matches {
		workCh <- m
	}
	close(workCh)
	wg.Wait()

	// Stop progress reporting
	done <- true
	fmt.Println() // Add a newline after the progress bar

	// Print summary
	duration := time.Since(startTime)
	fmt.Printf("Evaluation complete - took %s (%.1f matches/sec)\n",
		duration.Round(time.Second), float64(matchCount)/duration.Seconds())

	return results
}

// Evaluate runs self-play to compute a genome's fitness.
// It plays `games` matches using the genome as both players or against a baseline.
// Returns the win rate in [0.0,1.0].
func Evaluate(g *Genome, games, threads int) float64 {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	rand.Seed(time.Now().UnixNano())

	params := training.DefaultRPSSelfPlayParams()
	deckSize, handSize, maxRounds := params.DeckSize, params.HandSize, params.MaxRounds
	mctsParams := params.MCTSParams

	winsCh := make(chan int, threads)
	drawsCh := make(chan int, threads)
	var wg sync.WaitGroup
	gamesPerWorker := games / threads
	remainder := games % threads

	for w := 0; w < threads; w++ {
		wg.Add(1)
		count := gamesPerWorker
		if w == threads-1 {
			count += remainder
		}
		go func(offset, count int) {
			defer wg.Done()
			localWins, localDraws := 0, 0
			for i := 0; i < count; i++ {
				gme := game.NewRPSGame(deckSize, handSize, maxRounds)
				p1, v1 := g.ToNetworks()
				p2, v2 := g.ToNetworks()
				e1 := mcts.NewRPSMCTS(p1, v1, mctsParams)
				e2 := mcts.NewRPSMCTS(p2, v2, mctsParams)

				first := ((offset + i) % 2) == 0
				for !gme.IsGameOver() {
					currentIsP1 := gme.CurrentPlayer == game.Player1
					if currentIsP1 == first {
						e1.SetRootState(gme)
						if node := e1.Search(); node != nil && node.Move != nil {
							gme.MakeMove(*node.Move)
						}
					} else {
						e2.SetRootState(gme)
						if node := e2.Search(); node != nil && node.Move != nil {
							gme.MakeMove(*node.Move)
						}
					}
				}
				if gme.GetWinner() == game.NoPlayer {
					localDraws++
				} else {
					localWins++
				}
			}
			winsCh <- localWins
			drawsCh <- localDraws
		}(w*gamesPerWorker, count)
	}
	wg.Wait()
	close(winsCh)
	close(drawsCh)

	totalWins, totalDraws := 0, 0
	for w := range winsCh {
		totalWins += w
	}
	for d := range drawsCh {
		totalDraws += d
	}
	return (float64(totalWins) + 0.5*float64(totalDraws)) / float64(games)
}
