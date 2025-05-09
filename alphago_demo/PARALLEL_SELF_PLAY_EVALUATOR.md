# Parallel MCTS Self-Play Evaluator for NEAT

This guide shows a mid-level SWE how to implement a parallel, head-to-head self-play evaluator for genomes in `neat/evaluator.go`.

## Goals
- Run `N` games per genome with two identical MCTS agents.
- Alternate first player to remove bias.
- Parallelize across CPU cores for speed.
- Return fitness = `(wins + 0.5*draws) / N`.

---

## 1. Update Imports

In `pkg/training/neat/evaluator.go`, add:

```go
import (
  "runtime"
  "sync"
  "math/rand"
  "time"
  "github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
  "github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
)
```

---

## 2. Parallel Evaluate Signature

Change `Evaluate` to accept `threads`:

```go
func Evaluate(g *Genome, games, threads int) float64 {
  if threads <= 0 {
    threads = runtime.NumCPU()
  }
  // ...
}
```

---

## 3. Worker Pool Setup

Inside `Evaluate`, create channels and a WaitGroup:

```go
winsCh := make(chan int, threads)
drawsCh := make(chan int, threads)
var wg sync.WaitGroup
gamesPerWorker := games / threads
``` 

Last worker handles the remainder:

```go
remainder := games % threads
``` 

---

## 4. Worker Function

Spawn `threads` goroutines:

```go
for w := 0; w < threads; w++ {
  wg.Add(1)
  start := w * gamesPerWorker
  count := gamesPerWorker
  if w == threads-1 {
    count += remainder
  }
  go func(offset, count int) {
    defer wg.Done()
    localWins, localDraws := 0, 0
    // two MCTS engines
    for i := 0; i < count; i++ {
      // instantiate game & engines
      gme := game.NewRPSGame(...)
      p1, v1 := g.ToNetworks()
      p2, v2 := g.ToNetworks()
      e1 := mcts.NewRPSMCTS(p1, v1, mcts.DefaultRPSMCTSParams())
      e2 := mcts.NewRPSMCTS(p2, v2, mcts.DefaultRPSMCTSParams())

      // choose who starts to alternate bias
      first := ((offset + i) % 2 == 0)
      for !gme.IsGameOver() {
        if (gme.GetCurrentPlayer() == game.Player1) == first {
          e1.SetRootState(gme)
          gme.MakeMove(*e1.Search().Move)
        } else {
          e2.SetRootState(gme)
          gme.MakeMove(*e2.Search().Move)
        }
      }
      // tally
      switch gme.GetWinner() {
      case game.Player1:
        if first { localWins++ } else { localDraws++ }
      case game.Player2:
        if !first { localWins++ } else { localDraws++ }
      default:
        localDraws++
      }
    }
    winsCh <- localWins
    drawsCh <- localDraws
  }(start, count)
}
```

---

## 5. Aggregate & Compute Fitness

After spawning, wait and collect:

```go
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
return float64(totalWins) + 0.5*float64(totalDraws)
  / float64(games)
```

---

## 6. Tuning Tips

- Expose `mctsParams.NumSimulations` as a flag.
- Balance `games` vs. simulations for performance.
- Seed RNG once (`rand.Seed(time.Now().UnixNano())`).

—and that’s it! Your evaluator now runs full self-play head-to-head NEAT fitness in parallel.
