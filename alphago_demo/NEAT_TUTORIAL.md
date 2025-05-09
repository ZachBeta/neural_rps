# NEAT Integration Tutorial

This step-by-step guide walks a mid-level SWE through adding a home-grown NEAT trainer to the existing AlphaGo-style CLI in Go.

## Prerequisites
- Familiarity with Go, `flag` package, and basic project layout
- Understanding of AlphaGo self-play flow (`train_models/main.go`)
- Basic knowledge of evolutionary algorithms (genomes, crossover, mutation)

## Overview
We’ll extend `cmd/train_models/main.go` with a `--method neat` branch and scaffold a new `pkg/training/neat` module. At the end, you can:

```bash
go build -o bin/train_models cmd/train_models/main.go
./bin/train_models --method neat --pop-size 100 --generations 30 --eval-games 10
``` 

This produces `output/rps_neat_*.model` files that you can pit against AlphaGo models.

---

## Step 1: CLI Dispatch & Flags

1. Open `cmd/train_models/main.go`.
2. Add at top:
   ```go
   import "github.com/zachbeta/neural_rps/alphago_demo/pkg/training/neat"
   ```
3. Before `flag.Parse()`, define:
   ```go
   method         := flag.String("method", "alphago", "alphago | neat")
   popSize        := flag.Int("pop-size", 150, "Population size for NEAT")
   generations    := flag.Int("generations", 30, "Number of NEAT generations")
   mutRate        := flag.Float64("mut-rate", 0.05, "Mutation rate per gene")
   cxRate         := flag.Float64("cx-rate", 0.8, "Crossover rate between parents")
   compatThr      := flag.Float64("compat-threshold", 3.0, "Speciation distance threshold")
   evalGames      := flag.Int("eval-games", 10, "Self-play matches per genome")
   weightStd      := flag.Float64("weight-std", 0.1, "Std-dev for weight mutations")
   ```
4. After parsing, insert:
   ```go
   if *method == "neat" {
     cfg := neat.Config{*popSize, *generations, *mutRate, *cxRate, *compatThr, *evalGames, *weightStd}
     policyNet, valueNet := neat.Train(cfg, *parallel, *threads)
     // save networks (as-is)
     return
   }
   ```

---

## Step 2: Scaffold `pkg/training/neat`

In `alphago_demo/pkg/training/neat/`, create files:

- **config.go**: defines `type Config struct { PopSize, Generations, EvalGames int; MutRate, CxRate, CompatThreshold, WeightStd float64 }`
- **genome.go**: stub `type Genome { PolicyWeights, ValueWeights []float64; Fitness float64 }` with `NewGenome`, `Mutate`, `Crossover`, `CompatibilityDistance`, `ToNetworks()`
- **population.go**: `type Population` with `NewPopulation(cfg)`, `Evolve(cfg)`—skeleton of speciation & breeding.
- **evaluator.go**: `func Evaluate(g *Genome, games int) float64` stub.
- **trainer.go**: `func Train(cfg Config, parallel bool, threads int) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork)` calls the above.

---

## Step 3: Genome ↔ Network Conversion

Implement `NewGenome(cfg)`:
```go
wInH := cfg.HiddenSize * InputSize
wHO := OutputSize * cfg.HiddenSize
g := &Genome{PolicyWeights: randN(wInH), ValueWeights: randN(wHO)}
```
And `ToNetworks()`:
```go
p := neural.NewRPSPolicyNetwork(cfg.HiddenSize)
s := neural.NewRPSValueNetwork(cfg.HiddenSize)
p.SetWeights(g.PolicyWeights)
s.SetWeights(g.ValueWeights)
return p,s
```

---

## Step 4: Fitness Evaluation

In `evaluator.go`:
```go
func Evaluate(g *Genome, games int) float64 {
  p, v := g.ToNetworks()
  sp := training.NewRPSSelfPlay(p, v, params)
  examples := sp.GenerateGames(false)
  wins := countWins(examples)
  return float64(wins)/float64(games)
}
```
Write a smoke test to ensure return ∈ [0,1].

---

## Step 5: Mutation & Crossover

- **Mutate**: loop weights, if `rand.Float64() < cfg.MutRate` add `rand.NormFloat64()*cfg.WeightStd`.
- **Crossover**: create child weights by choosing each gene from parent1 or parent2 based on fitness or mix.

Add unit tests verifying genetic operations.

---

## Step 6: Speciation & Evolution Loop

In `Population.Evolve`:
1. Compute distances via `CompatibilityDistance` and assign species.
2. For each species, select top genomes (elitism).
3. Fill new generation: pair parents, crossover, mutate.
4. Evaluate fitness with `Evaluate`.
5. Repeat for cfg.Generations.

---

## Step 7: Integration & Testing

1. Run: `make train && bin/train_models --method neat`.
2. Check `output/` for new model files.
3. Pit NEAT vs. AlphaGo: `--tournament-games 50`.

---

## Next Steps
- Experiment with speciation thresholds, population size vs. generations.
- Add advanced structural mutations (add/remove nodes/links).
- Port this module to Rust using the same package structure.

Happy evolving!
