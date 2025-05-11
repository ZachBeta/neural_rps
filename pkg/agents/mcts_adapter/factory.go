package mcts_adapter

import (
	"fmt"

	"github.com/zachbeta/neural_rps/pkg/agents/mcts"
	"github.com/zachbeta/neural_rps/pkg/game"
)

// CreateGPUAgent creates a new GPU-accelerated MCTS agent
func CreateGPUAgent(serviceAddr string, numSimulations int, batchSize int) (mcts.MCTSAgent, mcts.GameState, error) {
	// Create a new adapter for the GPU MCTS agent
	agent, err := NewGPUMCTSAdapter(serviceAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create GPU MCTS agent: %v", err)
	}

	// Set the number of simulations by adjusting the batch size
	if numSimulations > 0 {
		// We'll use the batch size as configured
		agent.SetBatchSize(batchSize)
	}

	// Create a new game state
	gameState := game.NewRPSCardGame(9, 3, 9)

	// Wrap game state in adapter
	adaptedState := NewRPSGameStateAdapter(gameState)

	return agent, adaptedState, nil
}

// CreateCPUAgent creates a new CPU-based MCTS agent (placeholder for now)
func CreateCPUAgent(numSimulations int, batchSize int) (mcts.MCTSAgent, mcts.GameState, error) {
	// TODO: Implement a CPU-based MCTS agent adapter when needed
	return nil, nil, fmt.Errorf("CPU agent not implemented yet")
}
