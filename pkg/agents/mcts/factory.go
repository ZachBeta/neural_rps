package mcts

import (
	"fmt"

	"github.com/zachbeta/neural_rps/pkg/game"
)

// CreateGPUAgent creates a new GPU-accelerated MCTS agent
func CreateGPUAgent(serviceAddr string, numSimulations int, batchSize int) (MCTSAgent, GameState, error) {
	// Create default parameters
	params := DefaultMCTSParams()
	params.NumSimulations = numSimulations

	// Create the adapter for the MCTS agent
	agent, err := NewGPUMCTSAdapter(serviceAddr, params)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create GPU MCTS agent: %v", err)
	}

	// Set batch size if specified
	if batchSize > 0 {
		agent.SetBatchSize(batchSize)
	}

	// Create a new game state
	gameState := game.NewRPSCardGame(9, 3, 9)

	// Wrap game state in adapter
	adaptedState := NewRPSGameStateAdapter(gameState)

	return agent, adaptedState, nil
}

// CreateCPUAgent creates a new CPU MCTS agent (TODO: implement this when needed)
func CreateCPUAgent(numSimulations int, batchSize int) (MCTSAgent, GameState, error) {
	// TODO: Add implementation when needed
	return nil, nil, fmt.Errorf("CPU agent not implemented yet")
}
