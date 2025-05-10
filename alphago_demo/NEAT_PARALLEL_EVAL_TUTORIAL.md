# NEAT Parallel Evaluation Tutorial

This guide walks a mid-level SWE through implementing an efficient, parallelized evaluation system for NEAT training in Go, combining round robin and Hall-of-Fame (HOF) matches. The goal is to maximize CPU usage, ensure robust fitness evaluation, and provide clear CLI feedback during training.

---

## Overview
- **Worker Pool Model:** Use a work queue and goroutine pool for efficient parallel evaluation.
- **Evaluation Types:** Each genome is evaluated against a random subset of current-population genomes (round robin) and a Hall-of-Fame archive.
- **CLI Feedback:** Print concise per-generation summaries for real-time monitoring.

---

## Step 1: Define Match and Result Structures
```go
type Match struct {
    GenomeIdx int      // index of genome being evaluated
    Opponent  *Genome  // pointer to opponent (round robin or HOF)
    Games     int
    IsHOF     bool
}

type GenomeResult struct {
    Wins  int32
    Draws int32
    Games int32
}
```

---

## Step 2: Prepare the Match List
- For each genome:
    - Select N random round robin opponents from the current population (excluding self).
    - Add M matches against each HOF member.
- Example:
```go
matches := make([]Match, 0)
for i, g := range population.Genomes {
    // Round robin
    for _, oppIdx := range randomSubset(i, N, len(population.Genomes)) {
        matches = append(matches, Match{i, population.Genomes[oppIdx], 2, false})
    }
    // Hall of Fame
    for _, hof := range HOF {
        matches = append(matches, Match{i, hof, 5, true})
    }
}
```

---

## Step 3: Set Up the Worker Pool
```go
results := make([]*GenomeResult, len(population.Genomes))
for i := range results {
    results[i] = &GenomeResult{}
}

workCh := make(chan Match, len(matches))
wg := sync.WaitGroup{}
for i := 0; i < numWorkers; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        for match := range workCh {
            wins, draws := runGames(match.GenomeIdx, match.Opponent, match.Games)
            atomic.AddInt32(&results[match.GenomeIdx].Wins, int32(wins))
            atomic.AddInt32(&results[match.GenomeIdx].Draws, int32(draws))
            atomic.AddInt32(&results[match.GenomeIdx].Games, int32(match.Games))
        }
    }()
}
for _, m := range matches {
    workCh <- m
}
close(workCh)
wg.Wait()
```

---

## Step 4: Aggregate and Assign Fitness
```go
for i, res := range results {
    g := population.Genomes[i]
    g.Fitness = (float64(res.Wins) + 0.5*float64(res.Draws)) / float64(res.Games)
}
```

---

## Step 5: Print Per-Generation CLI Summary
Add this after fitness aggregation:
```go
best, sum := 0.0, 0.0
for _, g := range population.Genomes {
    if g.Fitness > best {
        best = g.Fitness
    }
    sum += g.Fitness
}
avg := sum / float64(len(population.Genomes))
fmt.Printf("Generation %d/%d | Best: %.3f | Avg: %.3f | Species: %d\n",
    gen, cfg.Generations, best, avg, len(population.Species))
```

---

## Tips & Best Practices
- Use `runtime.NumCPU()` for `numWorkers` by default.
- Avoid printing from inside workers; aggregate and print from the main goroutine.
- Tune number of round robin and HOF matches for speed vs. robustness.
- Optionally, add a progress bar or log HOF updates for advanced feedback.

---

## Conclusion
This pattern ensures robust, scalable NEAT evaluation and clear CLI progress, making it easy for any mid-level SWE to implement and maintain.
