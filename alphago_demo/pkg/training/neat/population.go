package neat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// Population manages a NEAT population over generations.
type Population struct {
	Genomes      []*Genome     // all genomes in current generation
	Species      map[int][]int // species ID -> indices of genomes
	innovCounter int           // global innovation counter for new genes (if using dynamic topology)
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
	fmt.Printf("\n=== Starting NEAT evolution with %d genomes, %d generations ===\n",
		len(p.Genomes), cfg.Generations)

	// Display network architecture information
	fmt.Println("\n=== Network Architecture ===")
	// Create example networks to analyze structure
	exampleGenome := p.Genomes[0]
	policyNet, valueNet := exampleGenome.ToNetworks()

	// Get network structure details using the network stats functions
	policyStats := neural.CalculatePolicyNetworkStats(policyNet)
	valueStats := neural.CalculateValueNetworkStats(valueNet)

	// Display structure
	fmt.Printf("Policy Network: %d inputs, %d hidden neurons, %d outputs\n",
		policyStats.InputSize, policyStats.HiddenSize, policyStats.OutputSize)
	fmt.Printf("Value Network: %d inputs, %d hidden neurons, %d outputs\n",
		valueStats.InputSize, valueStats.HiddenSize, valueStats.OutputSize)

	// Calculate parameters
	totalParams := policyStats.TotalParameters + valueStats.TotalParameters

	fmt.Printf("Total parameters: %d (%d policy, %d value)\n",
		totalParams, policyStats.TotalParameters, valueStats.TotalParameters)

	// Display weight initialization statistics
	policyWeightStats := analyzeWeights(exampleGenome.PolicyWeights)
	valueWeightStats := analyzeWeights(exampleGenome.ValueWeights)

	fmt.Printf("Policy weights: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n",
		policyWeightStats.min, policyWeightStats.max,
		policyWeightStats.mean, policyWeightStats.std)
	fmt.Printf("Value weights: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n",
		valueWeightStats.min, valueWeightStats.max,
		valueWeightStats.mean, valueWeightStats.std)

	startTime := time.Now()
	var bestGenome *Genome
	var bestFitness float64

	for gen := 1; gen <= cfg.Generations; gen++ {
		genStartTime := time.Now()
		fmt.Printf("\n--- Generation %d/%d ---\n", gen, cfg.Generations)

		// Parallel evaluation: assign fitness to all genomes
		var hof []*Genome // Hall-of-Fame (empty for now)
		results := parallelEvaluate(p, hof)

		// Update fitness values
		for i, res := range results {
			g := p.Genomes[i]
			if res.Games > 0 {
				g.Fitness = (float64(res.Wins) + 0.5*float64(res.Draws)) / float64(res.Games)
			} else {
				g.Fitness = 0
			}
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

		// Calculate statistics for this generation
		best, sum, fitnessValues := 0.0, 0.0, make([]float64, len(p.Genomes))
		bestIdx := 0
		for i, g := range p.Genomes {
			fitnessValues[i] = g.Fitness
			sum += g.Fitness
			if g.Fitness > best {
				best = g.Fitness
				bestIdx = i
			}
		}
		avg := sum / float64(len(p.Genomes))

		// Sort fitness values for distribution analysis
		sort.Float64s(fitnessValues)
		median := fitnessValues[len(fitnessValues)/2]
		q1 := fitnessValues[len(fitnessValues)/4]
		q3 := fitnessValues[3*len(fitnessValues)/4]

		// Print generation summary
		genTime := time.Since(genStartTime)
		fmt.Printf("NEAT Generation %d/%d — best=%.4f, avg=%.4f, median=%.4f\n",
			gen, cfg.Generations, best, avg, median)
		fmt.Printf("Fitness distribution: min=%.4f, q1=%.4f, median=%.4f, q3=%.4f, max=%.4f\n",
			fitnessValues[0], q1, median, q3, fitnessValues[len(fitnessValues)-1])
		fmt.Printf("Species: %d | Generation time: %s\n", len(p.Species), genTime)

		// Print species information
		fmt.Printf("Species distribution:\n")
		speciesFitness := make(map[int]float64)
		for speciesID, members := range p.Species {
			speciesSum := 0.0
			for _, memberIdx := range members {
				speciesSum += p.Genomes[memberIdx].Fitness
			}
			speciesAvg := speciesSum / float64(len(members))
			speciesFitness[speciesID] = speciesAvg
			fmt.Printf("  Species %d: %d members, avg fitness=%.4f\n",
				speciesID, len(members), speciesAvg)
		}

		// Track best genome over all generations
		if gen == 1 || best > bestFitness {
			bestFitness = best
			bestGenome = p.Genomes[bestIdx].Copy()
			fmt.Printf("New best genome found: fitness=%.4f\n", bestFitness)
		}

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

	totalTime := time.Since(startTime)
	fmt.Printf("\n=== Evolution complete ===\n")
	fmt.Printf("Total time: %s, generations: %d\n", totalTime, cfg.Generations)
	fmt.Printf("Best fitness achieved: %.4f\n", bestFitness)

	return bestGenome
}

// weightStats holds basic statistics about a weight array
type weightStats struct {
	min, max, mean, std float64
}

// analyzeWeights computes min, max, mean, and std of weights
func analyzeWeights(weights []float64) weightStats {
	if len(weights) == 0 {
		return weightStats{}
	}

	min, max := weights[0], weights[0]
	sum := 0.0

	for _, w := range weights {
		if w < min {
			min = w
		}
		if w > max {
			max = w
		}
		sum += w
	}

	mean := sum / float64(len(weights))

	// Calculate standard deviation
	variance := 0.0
	for _, w := range weights {
		variance += (w - mean) * (w - mean)
	}
	variance /= float64(len(weights))
	std := math.Sqrt(variance)

	return weightStats{
		min:  min,
		max:  max,
		mean: mean,
		std:  std,
	}
}
