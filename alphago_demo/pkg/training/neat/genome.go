package neat

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// Genome represents a candidate solution with policy and value network weights.
type Genome struct {
	PolicyWeights []float64
	ValueWeights  []float64
	HiddenSize    int
	Fitness       float64
}

// NewGenome initializes a new random genome based on config.
func NewGenome(cfg Config) *Genome {
	// Initialize random networks
	pNet := neural.NewRPSPolicyNetwork(cfg.HiddenSize)
	vNet := neural.NewRPSValueNetwork(cfg.HiddenSize)
	// Package their weights into the genome
	return &Genome{
		PolicyWeights: pNet.GetWeights(),
		ValueWeights:  vNet.GetWeights(),
		HiddenSize:    cfg.HiddenSize,
		Fitness:       0.0,
	}
}

// Mutate applies genetic mutations to the genome's weights.
func (g *Genome) Mutate(cfg Config) {
	for i := range g.PolicyWeights {
		if rand.Float64() < cfg.MutRate {
			g.PolicyWeights[i] += rand.NormFloat64() * cfg.WeightStd
		}
	}
	for i := range g.ValueWeights {
		if rand.Float64() < cfg.MutRate {
			g.ValueWeights[i] += rand.NormFloat64() * cfg.WeightStd
		}
	}
}

// Crossover combines two parent genomes into a new child genome.
func Crossover(parent1, parent2 *Genome, cfg Config) *Genome {
	child := &Genome{HiddenSize: parent1.HiddenSize}
	if rand.Float64() > cfg.CxRate {
		fitter := parent1
		if parent2.Fitness > parent1.Fitness {
			fitter = parent2
		}
		child.PolicyWeights = append([]float64(nil), fitter.PolicyWeights...)
		child.ValueWeights = append([]float64(nil), fitter.ValueWeights...)
	} else {
		child.PolicyWeights = make([]float64, len(parent1.PolicyWeights))
		for i := range child.PolicyWeights {
			if rand.Float64() < 0.5 {
				child.PolicyWeights[i] = parent1.PolicyWeights[i]
			} else {
				child.PolicyWeights[i] = parent2.PolicyWeights[i]
			}
		}
		child.ValueWeights = make([]float64, len(parent1.ValueWeights))
		for i := range child.ValueWeights {
			if rand.Float64() < 0.5 {
				child.ValueWeights[i] = parent1.ValueWeights[i]
			} else {
				child.ValueWeights[i] = parent2.ValueWeights[i]
			}
		}
	}
	child.Fitness = 0.0
	return child
}

// CompatibilityDistance computes how different two genomes are for speciation.
func (g *Genome) CompatibilityDistance(other *Genome) float64 {
	totalDiff := 0.0
	count := 0
	for i, v := range g.PolicyWeights {
		totalDiff += math.Abs(v - other.PolicyWeights[i])
		count++
	}
	for i, v := range g.ValueWeights {
		totalDiff += math.Abs(v - other.ValueWeights[i])
		count++
	}
	if count == 0 {
		return 0.0
	}
	return totalDiff / float64(count)
}

// ToNetworks converts the genome's weights into policy and value neural networks.
func (g *Genome) ToNetworks() (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork) {
	pNet := neural.NewRPSPolicyNetwork(g.HiddenSize)
	if err := pNet.SetWeights(g.PolicyWeights); err != nil {
		panic(fmt.Sprintf("failed to set policy weights: %v", err))
	}
	vNet := neural.NewRPSValueNetwork(g.HiddenSize)
	if err := vNet.SetWeights(g.ValueWeights); err != nil {
		panic(fmt.Sprintf("failed to set value weights: %v", err))
	}
	return pNet, vNet
}

// Copy creates a deep copy of the genome
func (g *Genome) Copy() *Genome {
	// Create new genome with same weights
	newGenome := &Genome{
		HiddenSize:    g.HiddenSize,
		Fitness:       g.Fitness,
		PolicyWeights: make([]float64, len(g.PolicyWeights)),
		ValueWeights:  make([]float64, len(g.ValueWeights)),
	}

	// Copy weights
	copy(newGenome.PolicyWeights, g.PolicyWeights)
	copy(newGenome.ValueWeights, g.ValueWeights)

	return newGenome
}
