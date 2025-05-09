package neat

import (
	"math/rand"
	"runtime"
	"sync"
	"time"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
)

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
