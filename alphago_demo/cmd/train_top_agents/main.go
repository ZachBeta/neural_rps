package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/training/neat"
)

// Training configuration
const (
	// Game parameters
	deckSize  = 21
	handSize  = 5
	maxRounds = 10

	// Training parameters
	defaultSelfPlayGames      = 2000
	defaultMCTSSimulations    = 400
	defaultCheckpointInterval = 500

	// NEAT parameters
	defaultPopulationSize = 150
	defaultGenerations    = 20
	defaultHiddenSize     = 64
)

// Agent represents a model to be trained
type Agent struct {
	Name              string
	Type              string // "AlphaGo" or "NEAT"
	PolicyPath        string
	ValuePath         string
	TrainedPolicyPath string
	TrainedValuePath  string
}

func main() {
	// Parse command line flags
	selfPlayGames := flag.Int("games", defaultSelfPlayGames, "Self-play games for training")
	mctsSimulations := flag.Int("sims", defaultMCTSSimulations, "MCTS simulations per move")
	tournamentOnly := flag.Bool("tournament-only", false, "Skip training and run tournament only")
	trainingOnly := flag.Bool("training-only", false, "Skip tournament and do training only")
	outputDir := flag.String("output", "output/extended_training", "Directory for output files")
	tournamentGames := flag.Int("tournament-games", 100, "Games per matchup in final tournament")

	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create output directory if needed
	os.MkdirAll(*outputDir, 0755)

	// Define top agents based on previous tournament
	topAgents := []Agent{
		{
			Name:       "AlphaGo-128_g20_e6_20250325-192621",
			Type:       "AlphaGo",
			PolicyPath: "output/rps_h128_g20_e6_20250325-192621_policy.model",
			ValuePath:  "output/rps_h128_g20_e6_20250325-192621_value.model",
		},
		{
			Name: "Random",
			Type: "Random",
			// Random agent has no model files
		},
		{
			Name:       "NEAT-_gen09",
			Type:       "NEAT",
			PolicyPath: "output/neat_gen09_policy.model",
			ValuePath:  "output/neat_gen09_value.model",
		},
		{
			Name:       "AlphaGo-64_g100_e5_20250509-072802",
			Type:       "AlphaGo",
			PolicyPath: "output/rps_h64_g100_e5_20250509-072802_policy.model",
			ValuePath:  "output/rps_h64_g100_e5_20250509-072802_value.model",
		},
		{
			Name:       "NEAT-_gen15",
			Type:       "NEAT",
			PolicyPath: "output/neat_gen15_policy.model",
			ValuePath:  "output/neat_gen15_value.model",
		},
		{
			Name:       "NEAT-_gen01",
			Type:       "NEAT",
			PolicyPath: "output/neat_gen01_policy.model",
			ValuePath:  "output/neat_gen01_value.model",
		},
		{
			Name:       "NEAT-_gen03",
			Type:       "NEAT",
			PolicyPath: "output/neat_gen03_policy.model",
			ValuePath:  "output/neat_gen03_value.model",
		},
		{
			Name:       "NEAT-_gen24",
			Type:       "NEAT",
			PolicyPath: "output/neat_gen24_policy.model",
			ValuePath:  "output/neat_gen24_value.model",
		},
	}

	// Generate output paths for trained models
	timestamp := time.Now().Format("20060102-150405")
	for i := range topAgents {
		if topAgents[i].Type == "Random" {
			continue // Skip Random agent
		}
		// Extract base name without extension
		baseName := filepath.Base(topAgents[i].PolicyPath)
		baseName = strings.TrimSuffix(baseName, "_policy.model")

		topAgents[i].TrainedPolicyPath = fmt.Sprintf("%s/%s_extended_%s_policy.model",
			*outputDir, baseName, timestamp)
		topAgents[i].TrainedValuePath = fmt.Sprintf("%s/%s_extended_%s_value.model",
			*outputDir, baseName, timestamp)
	}

	if !*tournamentOnly {
		// Train all non-random agents
		for i, agent := range topAgents {
			if agent.Type == "Random" {
				fmt.Printf("Skipping training for Random agent\n\n")
				continue
			}

			fmt.Printf("\n=== Training Agent %d/%d: %s ===\n",
				i+1, len(topAgents), agent.Name)

			if agent.Type == "AlphaGo" {
				trainAlphaGoAgent(agent, *selfPlayGames, *mctsSimulations, *outputDir)
			} else if agent.Type == "NEAT" {
				trainNEATAgent(agent, *selfPlayGames, *mctsSimulations, *outputDir)
			}
		}
	}

	if !*trainingOnly {
		// Run tournament with trained agents
		runTournament(topAgents, *tournamentGames, *outputDir)
	}
}

// trainAlphaGoAgent extends training of an AlphaGo agent
func trainAlphaGoAgent(agent Agent, selfPlayGames, mctsSimulations int, outputDir string) {
	fmt.Printf("Loading AlphaGo model from %s and %s\n",
		agent.PolicyPath, agent.ValuePath)

	// Determine hidden size by loading the policy network
	policyNet := neural.NewRPSPolicyNetwork(64) // Default size, will be adjusted on load
	err := policyNet.LoadFromFile(agent.PolicyPath)
	if err != nil {
		fmt.Printf("Error loading policy network: %v\n", err)
		return
	}

	valueNet := neural.NewRPSValueNetwork(policyNet.GetHiddenSize())
	err = valueNet.LoadFromFile(agent.ValuePath)
	if err != nil {
		fmt.Printf("Error loading value network: %v\n", err)
		return
	}

	// Get network stats for logging
	policyStats := neural.CalculatePolicyNetworkStats(policyNet)
	valueStats := neural.CalculateValueNetworkStats(valueNet)
	hiddenSize := policyStats.HiddenSize

	fmt.Printf("Network architecture: %d inputs, %d hidden neurons\n",
		policyStats.InputSize, hiddenSize)
	fmt.Printf("Total parameters: %d (%d policy, %d value)\n",
		policyStats.TotalParameters+valueStats.TotalParameters,
		policyStats.TotalParameters, valueStats.TotalParameters)

	// Create self-play parameters
	selfPlayParams := training.DefaultRPSSelfPlayParams()
	selfPlayParams.NumGames = selfPlayGames
	selfPlayParams.DeckSize = deckSize
	selfPlayParams.HandSize = handSize
	selfPlayParams.MaxRounds = maxRounds
	selfPlayParams.MCTSParams.NumSimulations = mctsSimulations
	selfPlayParams.ForceParallel = true

	// Create self-play instance
	selfPlay := training.NewRPSSelfPlay(policyNet, valueNet, selfPlayParams)

	// Run self-play
	fmt.Printf("Starting self-play with %d games, %d simulations per move...\n",
		selfPlayGames, mctsSimulations)
	startTime := time.Now()
	examples := selfPlay.GenerateGames(true) // Use verbose mode
	genTime := time.Since(startTime)

	examplesPerGame := float64(len(examples)) / float64(selfPlayGames)
	gamesPerSecond := float64(selfPlayGames) / genTime.Seconds()

	fmt.Printf("Generated %d training examples in %s (%.1f examples/game, %.2f games/sec)\n",
		len(examples), genTime, examplesPerGame, gamesPerSecond)

	// Train networks
	fmt.Printf("Training networks for %d epochs...\n", 10) // Fixed 10 epochs
	startTime = time.Now()
	policyLosses, valueLosses := selfPlay.TrainNetworks(10, 32, 0.001, true)
	trainTime := time.Since(startTime)

	// Print final losses
	if len(policyLosses) > 0 && len(valueLosses) > 0 {
		fmt.Printf("Final losses - Policy: %.4f, Value: %.4f\n",
			policyLosses[len(policyLosses)-1], valueLosses[len(valueLosses)-1])
	}

	fmt.Printf("Training completed in %s\n", trainTime)

	// Save trained models
	fmt.Printf("Saving trained models to %s and %s\n",
		agent.TrainedPolicyPath, agent.TrainedValuePath)

	if err := policyNet.SaveToFile(agent.TrainedPolicyPath); err != nil {
		fmt.Printf("Error saving policy network: %v\n", err)
	}

	if err := valueNet.SaveToFile(agent.TrainedValuePath); err != nil {
		fmt.Printf("Error saving value network: %v\n", err)
	}
}

// trainNEATAgent extends training of a NEAT agent
func trainNEATAgent(agent Agent, selfPlayGames, mctsSimulations int, outputDir string) {
	fmt.Printf("Starting NEAT extended training from %s and %s\n",
		agent.PolicyPath, agent.ValuePath)

	// Load the existing networks
	policyNet := neural.NewRPSPolicyNetwork(defaultHiddenSize)
	err := policyNet.LoadFromFile(agent.PolicyPath)
	if err != nil {
		fmt.Printf("Error loading policy network: %v\n", err)
		return
	}

	valueNet := neural.NewRPSValueNetwork(policyNet.GetHiddenSize())
	err = valueNet.LoadFromFile(agent.ValuePath)
	if err != nil {
		fmt.Printf("Error loading value network: %v\n", err)
		return
	}

	// Extract weights to create initial genome
	policyWeights := policyNet.GetWeights()
	valueWeights := valueNet.GetWeights()
	hiddenSize := policyNet.GetHiddenSize()

	// Create NEAT configuration
	cfg := neat.Config{
		PopSize:         defaultPopulationSize,
		Generations:     defaultGenerations,
		MutRate:         0.05,
		CxRate:          0.8,
		CompatThreshold: 3.0,
		EvalGames:       10,
		WeightStd:       0.1,
		HiddenSize:      hiddenSize,
	}

	// Create initial population with best genome as template
	fmt.Printf("Creating NEAT population based on existing model (hidden size: %d)\n", hiddenSize)
	fmt.Printf("Population size: %d, Generations: %d\n", cfg.PopSize, cfg.Generations)

	pop := newPopulationFromTemplate(cfg, policyWeights, valueWeights)

	// Run evolution
	fmt.Printf("Starting NEAT evolution...\n")
	startTime := time.Now()
	bestGenome := pop.Evolve(cfg, 0) // Use auto thread selection
	evolveTime := time.Since(startTime)

	fmt.Printf("NEAT evolution completed in %s\n", evolveTime)
	fmt.Printf("Best fitness achieved: %.4f\n", bestGenome.Fitness)

	// Convert best genome to networks and save
	bestPolicy, bestValue := bestGenome.ToNetworks()

	fmt.Printf("Saving best networks to %s and %s\n",
		agent.TrainedPolicyPath, agent.TrainedValuePath)

	if err := bestPolicy.SaveToFile(agent.TrainedPolicyPath); err != nil {
		fmt.Printf("Error saving policy network: %v\n", err)
	}

	if err := bestValue.SaveToFile(agent.TrainedValuePath); err != nil {
		fmt.Printf("Error saving value network: %v\n", err)
	}
}

// newPopulationFromTemplate creates a NEAT population with the first genome initialized from template weights
func newPopulationFromTemplate(cfg neat.Config, policyWeights, valueWeights []float64) *neat.Population {
	// Create a new population using the NEAT API instead of directly accessing fields
	pop := neat.NewPopulation(cfg)

	// Create first genome from template
	if len(pop.Genomes) > 0 {
		pop.Genomes[0] = &neat.Genome{
			PolicyWeights: make([]float64, len(policyWeights)),
			ValueWeights:  make([]float64, len(valueWeights)),
			HiddenSize:    cfg.HiddenSize,
			Fitness:       0.0,
		}

		// Copy weights
		copy(pop.Genomes[0].PolicyWeights, policyWeights)
		copy(pop.Genomes[0].ValueWeights, valueWeights)

		// Create rest of population with mutations of the template
		for i := 1; i < cfg.PopSize && i < len(pop.Genomes); i++ {
			// Clone and mutate
			pop.Genomes[i] = pop.Genomes[0].Copy()
			pop.Genomes[i].Mutate(cfg)
		}
	}

	return pop
}

// runTournament runs the final tournament with all trained agents
func runTournament(agents []Agent, gamesPerPair int, outputDir string) {
	fmt.Printf("\n=== Running Final Tournament with Trained Agents ===\n")

	// Create tournament results file
	timestamp := time.Now().Format("20060102-150405")
	tournamentOutput := fmt.Sprintf("%s/extended_tournament_%s.csv", outputDir, timestamp)

	// Prepare agent list for the tournament command
	agentArgs := []string{
		"run", "cmd/elo_tournament/main.go",
		"--games", fmt.Sprintf("%d", gamesPerPair),
		"--output", tournamentOutput,
		"--cutoff", "0", // Don't eliminate any agents
	}

	// Add explicit agents instead of auto-discovery
	agentArgs = append(agentArgs, "--agents")

	// Build agent list string
	var agentList []string
	for _, agent := range agents {
		if agent.Type == "Random" {
			agentList = append(agentList, "Random")
		} else {
			// Use trained model paths if available, otherwise fall back to original
			policyPath := agent.PolicyPath
			valuePath := agent.ValuePath
			if agent.TrainedPolicyPath != "" && agent.TrainedValuePath != "" {
				if _, err := os.Stat(agent.TrainedPolicyPath); err == nil {
					policyPath = agent.TrainedPolicyPath
				}
				if _, err := os.Stat(agent.TrainedValuePath); err == nil {
					valuePath = agent.TrainedValuePath
				}
			}
			agentList = append(agentList,
				fmt.Sprintf("%s:%s:%s", agent.Name, policyPath, valuePath))
		}
	}
	agentArgs = append(agentArgs, strings.Join(agentList, ","))

	// Run the tournament command
	fmt.Printf("Starting tournament with %d agents...\n", len(agents))
	fmt.Printf("Running command: go %s\n", strings.Join(agentArgs, " "))

	cmd := exec.Command("go", agentArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		fmt.Printf("Error running tournament: %v\n", err)
		return
	}

	fmt.Printf("Tournament completed. Results saved to %s\n", tournamentOutput)
}
