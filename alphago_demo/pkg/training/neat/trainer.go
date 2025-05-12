package neat

import neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"

// Train runs the NEAT algorithm and produces a policy and value network.
func Train(cfg Config, parallel bool, threads int) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork) {
	// Initialize population
	pop := NewPopulation(cfg)

	// Evolve population over generations (parallel vs sequential)
	if !parallel {
		threads = 1
	}
	champion := pop.Evolve(cfg, threads)

	// Convert champion genome into neural networks
	policyNet, valueNet := champion.ToNetworks()
	return policyNet, valueNet
}
