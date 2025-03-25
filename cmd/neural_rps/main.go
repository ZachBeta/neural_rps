package main

import (
	"fmt"
	"log"
	"time"

	"github.com/zachbeta/neural_rps/pkg/agent"
	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/visualizer"
)

func main() {
	// Initialize visualizer
	viz, err := visualizer.NewVisualizer("output")
	if err != nil {
		log.Fatalf("Failed to initialize visualizer: %v", err)
	}
	defer viz.Close()

	// Initialize environment and agent
	env := game.NewEnvironment()
	ppoAgent := agent.NewPPOAgent(9, 3) // 9 state dimensions, 3 actions

	// Visualize network architecture
	layerSizes := []int{9, 64, 3}
	layerNames := []string{"Input", "Hidden", "Output"}
	if err := viz.VisualizeArchitecture(layerSizes, layerNames); err != nil {
		log.Printf("Warning: Failed to visualize architecture: %v", err)
	}

	// Training parameters
	numEpisodes := 1000
	episodesPerUpdate := 10

	// Training buffers
	states := make([][]float64, 0)
	actions := make([]int, 0)
	rewards := make([]float64, 0)
	values := make([]float64, 0)
	episodeRewards := make([]float64, 0)

	var totalReward float64

	fmt.Println("Starting training...")
	if err := viz.WriteToFile("Starting training...\n"); err != nil {
		log.Printf("Warning: Failed to write to file: %v", err)
	}

	// Define labels for visualization
	inputLabels := []string{
		"LastW", "LastM", "LastA",
		"HandW", "HandM", "HandA",
		"OppW", "OppM", "OppA",
	}
	outputLabels := []string{"Warrior", "Mage", "Archer"}

	// Training loop
	for episode := 0; episode < numEpisodes; episode++ {
		env.Reset()
		var episodeReward float64

		for {
			state := env.GetState()
			validActions := make([]int, 0)
			for i := 0; i < 3; i++ {
				if env.IsValidAction(game.CardType(i)) {
					validActions = append(validActions, i)
				}
			}

			// Get action from policy
			action := ppoAgent.SampleAction(state, validActions)
			value := ppoAgent.GetValue(state)

			// Take action in environment
			reward, done := env.Step(game.CardType(action))
			episodeReward += reward

			// Store transition
			states = append(states, state)
			actions = append(actions, action)
			rewards = append(rewards, reward)
			values = append(values, value)

			// Visualize every 100 episodes
			if episode%100 == 0 {
				probs := ppoAgent.GetPolicyProbs(state)
				if err := viz.VisualizeActionProbs(probs, outputLabels); err != nil {
					log.Printf("Warning: Failed to visualize action probs: %v", err)
				}
				time.Sleep(500 * time.Millisecond)
			}

			if done {
				break
			}
		}

		totalReward += episodeReward
		episodeRewards = append(episodeRewards, episodeReward)

		// Update policy every episodesPerUpdate episodes
		if (episode+1)%episodesPerUpdate == 0 {
			ppoAgent.Update(states, actions, rewards, values)
			states = states[:0]
			actions = actions[:0]
			rewards = rewards[:0]
			values = values[:0]

			avgReward := totalReward / float64(episodesPerUpdate)
			fmt.Printf("Episode %d, Average Reward: %.3f\n", episode+1, avgReward)
			if err := viz.WriteToFile(fmt.Sprintf("Episode %d, Average Reward: %.3f\n", episode+1, avgReward)); err != nil {
				log.Printf("Warning: Failed to write to file: %v", err)
			}

			// Visualize weights and training progress
			if (episode+1)%100 == 0 {
				if err := viz.VisualizeWeights([][]float64{}, inputLabels, outputLabels); err != nil {
					log.Printf("Warning: Failed to visualize weights: %v", err)
				}
				if err := viz.VisualizeTrainingProgress(episodeRewards, 100); err != nil {
					log.Printf("Warning: Failed to visualize training progress: %v", err)
				}
			}

			totalReward = 0
		}
	}

	fmt.Println("\nTraining completed!")
	if err := viz.WriteToFile("\nTraining completed!\n"); err != nil {
		log.Printf("Warning: Failed to write to file: %v", err)
	}

	// Play demonstration games
	fmt.Println("\nPlaying demonstration games...")
	if err := viz.WriteToFile("\nPlaying demonstration games...\n"); err != nil {
		log.Printf("Warning: Failed to write to file: %v", err)
	}

	for i := 0; i < 3; i++ {
		env.Reset()
		fmt.Printf("\nGame %d:\n", i+1)
		if err := viz.WriteToFile(fmt.Sprintf("\nGame %d:\n", i+1)); err != nil {
			log.Printf("Warning: Failed to write to file: %v", err)
		}

		for {
			state := env.GetState()
			validActions := make([]int, 0)
			for i := 0; i < 3; i++ {
				if env.IsValidAction(game.CardType(i)) {
					validActions = append(validActions, i)
				}
			}

			probs := ppoAgent.GetPolicyProbs(state)
			if err := viz.VisualizeActionProbs(probs, outputLabels); err != nil {
				log.Printf("Warning: Failed to visualize action probs: %v", err)
			}

			action := ppoAgent.SampleAction(state, validActions)
			reward, done := env.Step(game.CardType(action))

			move := fmt.Sprintf("Agent played: %s, Reward: %.2f\n",
				game.NewCard(game.CardType(action)).Name(), reward)
			fmt.Print(move)
			if err := viz.WriteToFile(move); err != nil {
				log.Printf("Warning: Failed to write to file: %v", err)
			}

			time.Sleep(1000 * time.Millisecond)

			if done {
				break
			}
		}
	}
}
