# NEAT Integration Plan for AlphaGo-Style CLI  
A phase-by-phase roadmap for adding a home-grown NEAT trainer alongside the existing AlphaGo flow. Tailored for a mid-level SWE.

---

## Phase 1: CLI Dispatch & Config

1. **Add method flag** in `cmd/train_models/main.go`:
   ```go
   method := flag.String("method", "alphago", "Training method: alphago | neat")
   ```
2. **Expose NEAT flags** before `flag.Parse()`:
   - `--pop-size` (default: 150)
   - `--generations` (default: 30)
   - `--mut-rate` (default: 0.05)
   - `--cx-rate` (default: 0.8)
   - `--compat-threshold` (default: 3.0)
   - `--eval-games` (default: 10)
   - `--weight-std` (default: 0.1)
3. **Dispatch by method** after parsing:
   ```go
   switch *method {
   case "alphago":
     trainWithAlphaGo(...)
   case "neat":
     cfg := neat.Config{*popSize, *generations, *mutRate, *cxRate, *compatThr, *evalGames, *weightStd}
     policy, value := trainWithNEAT(cfg, *parallel, *threads)
   default:
     log.Fatalf("Unknown method: %s", *method)
   }
   ```
4. **Keep `runTournament(...)`** identical for both branches.

---

## Phase 2: Scaffold `pkg/training/neat`

- **config.go**: Define `Config` with core knobs.
- **genome.go**: Implement a fixed-topology `Genome`:
  - Weight slices for each layer
  - `Mutate(cfg Config)` and `Crossover(p1, p2 Genome, cfg Config)`
- **population.go**:
  - `NewPopulation(cfg Config)`
  - `Evolve(cfg Config) Genome` (runs all generations)
- **evaluator.go**: Evaluate a genome’s fitness via `cfg.EvalGames` self-play
- **trainer.go**: Entry point:
  ```go
  func Train(cfg Config, parallel bool, threads int) (policyNet, valueNet) {
    pop := NewPopulation(cfg)
    champ := pop.Evolve(cfg)
    return champ.ToNetworks()
  }
  ```

---

## Phase 3: Core NEAT Mechanics

1. **Initialization**: Random weight genomes
2. **Speciation**: Use a compatibility distance vs. `cfg.CompatThreshold`
3. **Selection & Crossover**:
   - Tournament or fitness‐proportionate
   - Mix weights with `cfg.CxRate`, else clone
4. **Mutation**: Gaussian perturbations on weights (σ=`cfg.WeightStd`) with chance `cfg.MutRate`
5. **Replacement**: Elitism + children fill

---

## Phase 4: Integration & Saving

- In `trainWithNEAT()`, call `neat.Train(...)` and get back two networks.
- Save models to disk as:
  - `output/rps_neat_policy.model`
  - `output/rps_neat_value.model`
- Pass them into `runTournament(...)` just like AlphaGo nets.

---

## Phase 5: Documentation & Testing

- **README update**: Add NEAT usage under CLI Flags section.
- **Unit tests** in `pkg/training/neat/` for:
  - `compatibilityDistance`
  - `Mutate`, `Crossover`, speciation boundaries
- (Optional) **Benchmark harness** to sweep `popSize` vs. `generations`.

---

This keeps the existing CLI and tournament workflow unchanged while isolating NEAT logic for easy porting to Rust later.
