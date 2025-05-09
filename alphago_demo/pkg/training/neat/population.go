package neat

import (
	"fmt"
	"math/rand"
)

// Population manages a NEAT population over generations.
type Population struct {
	Genomes      []*Genome        // all genomes in current generation
	Species      map[int][]int    // species ID -> indices of genomes
	innovCounter int              // global innovation counter for new genes (if using dynamic topology)
}

// NewPopulation creates an initial population of random genomes.
func NewPopulation(cfg Config) *Population {
	pop := &Population{
		Genomes:      make([]*Genome, cfg.PopSize),
		Species:      make(map[int][]int),
		innovCounter: 0,
	}
	for i := range pop.Genomes {
		pop.Genomes[i] = NewGenome(cfg)
	}
	return pop
}

// Evolve runs the NEAT algorithm for the configured number of generations
// and returns the best genome found.
func (p *Population) Evolve(cfg Config, threads int) *Genome {
	for gen := 1; gen <= cfg.Generations; gen++ {
		// Evaluate all genomes
		for _, g := range p.Genomes {
			g.Fitness = Evaluate(g, cfg.EvalGames, threads)
		}
		// Speciation
		p.Species = make(map[int][]int)
		for i, g := range p.Genomes {
			assigned := false
			for repIdx := range p.Species {
				rep := p.Genomes[repIdx]
				if g.CompatibilityDistance(rep) < cfg.CompatThreshold {
					p.Species[repIdx] = append(p.Species[repIdx], i)
					assigned = true
					break
				}
			}
			if !assigned {
				p.Species[i] = []int{i}
			}
		}
		// Log generation stats
		var sumFit float64
		bestFit := p.Genomes[0].Fitness
		for _, g := range p.Genomes {
			sumFit += g.Fitness
			if g.Fitness > bestFit {
				bestFit = g.Fitness
			}
		}
		avgFit := sumFit / float64(len(p.Genomes))
		fmt.Printf("NEAT Generation %d/%d — best=%.4f, avg=%.4f, species=%d\n", gen, cfg.Generations, bestFit, avgFit, len(p.Species))
		// Reproduction
		newGen := make([]*Genome, len(p.Genomes))
		// Preserve best
		bestIdx, bestFit := 0, p.Genomes[0].Fitness
		for i := 1; i < len(p.Genomes); i++ {
			if p.Genomes[i].Fitness > bestFit {
				bestFit = p.Genomes[i].Fitness
				bestIdx = i
			}
		}
		// Place champion at index 0
		champion := p.Genomes[bestIdx]
		newGen[0] = champion
		// Checkpoint champion networks
		polNet, valNet := champion.ToNetworks()
		polPath := fmt.Sprintf("output/neat_gen%02d_policy.model", gen)
		valPath := fmt.Sprintf("output/neat_gen%02d_value.model", gen)
		if err := polNet.SaveToFile(polPath); err != nil {
			panic(fmt.Sprintf("neat checkpoint policy save error: %v", err))
		}
		if err := valNet.SaveToFile(valPath); err != nil {
			panic(fmt.Sprintf("neat checkpoint value save error: %v", err))
		}
		// Collect species reps
		reps := make([]int, 0, len(p.Species))
		for repIdx := range p.Species {
			reps = append(reps, repIdx)
		}
		// Fill rest
		for j := 1; j < len(newGen); j++ {
			rep := reps[rand.Intn(len(reps))]
			members := p.Species[rep]
			p1 := p.Genomes[members[rand.Intn(len(members))]]
			p2 := p.Genomes[members[rand.Intn(len(members))]]
			child := Crossover(p1, p2, cfg)
			child.Mutate(cfg)
			newGen[j] = child
		}
		p.Genomes = newGen
	}
	// Return best from final generation
	return p.Genomes[0]
}
