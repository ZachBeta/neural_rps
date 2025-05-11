package mcts_adapter

import (
	"context"
	"fmt"

	"github.com/zachbeta/neural_rps/pkg/agents/mcts"
	"github.com/zachbeta/neural_rps/pkg/game"
)

// RPSGameStateAdapter adapts RPSCardGame to the mcts.GameState interface
type RPSGameStateAdapter struct {
	*game.RPSCardGame
}

// NewRPSGameStateAdapter creates a new adapter for RPSCardGame
func NewRPSGameStateAdapter(g *game.RPSCardGame) *RPSGameStateAdapter {
	return &RPSGameStateAdapter{g}
}

// Clone implements the mcts.GameState interface
func (g *RPSGameStateAdapter) Clone() mcts.GameState {
	return NewRPSGameStateAdapter(g.RPSCardGame.Copy())
}

// GetLegalMoves implements the mcts.GameState interface
func (g *RPSGameStateAdapter) GetLegalMoves() []game.RPSCardMove {
	return g.RPSCardGame.GetValidMoves()
}

// ApplyMove implements the mcts.GameState interface
func (g *RPSGameStateAdapter) ApplyMove(move game.RPSCardMove) {
	err := g.RPSCardGame.MakeMove(move)
	if err != nil {
		panic(fmt.Sprintf("Invalid move: %v", err))
	}
}

// IsGameOver implements the mcts.GameState interface
func (g *RPSGameStateAdapter) IsGameOver() bool {
	return g.RPSCardGame.IsGameOver()
}

// GetWinner implements the mcts.GameState interface
func (g *RPSGameStateAdapter) GetWinner() game.Player {
	return g.RPSCardGame.GetWinner()
}

// GetCurrentPlayer implements the mcts.GameState interface
func (g *RPSGameStateAdapter) GetCurrentPlayer() game.Player {
	return g.RPSCardGame.CurrentPlayer
}

// ToTensor implements the mcts.GameState interface
func (g *RPSGameStateAdapter) ToTensor() []float32 {
	return g.RPSCardGame.ToTensor()
}

// GetLastMove implements the mcts.GameState interface
func (g *RPSGameStateAdapter) GetLastMove() game.RPSCardMove {
	return g.RPSCardGame.LastMove
}

// Ensure RPSGameStateAdapter implements mcts.GameState
var _ mcts.GameState = (*RPSGameStateAdapter)(nil)

// GPUMCTSAdapter adapts the existing GPU MCTS implementation to an agent interface
// This agent interface would likely be mcts_adapter.MCTSAgent (if kept)
// For now, let's assume it takes an mcts.GameState for SetRootState directly.
type GPUMCTSAdapter struct {
	mctsAgent *mcts.GPUBatchedMCTS
}

// NewGPUMCTSAdapter creates a new GPU MCTS adapter
func NewGPUMCTSAdapter(serviceAddr string) (*GPUMCTSAdapter, error) {
	params := mcts.DefaultMCTSParams()
	agent, err := mcts.NewGPUBatchedMCTS(serviceAddr, params)
	if err != nil {
		return nil, err
	}
	return &GPUMCTSAdapter{mctsAgent: agent}, nil
}

// SetRootState sets the root state for the search.
// The input 'state' should be an mcts.GameState (e.g., *RPSGameStateAdapter from this package).
func (a *GPUMCTSAdapter) SetRootState(state mcts.GameState) {
	// The underlying MCTS engine also expects mcts.GameState
	a.mctsAgent.SetRootState(state)
}

// Search performs the MCTS search and returns the best move
func (a *GPUMCTSAdapter) Search(ctx context.Context) game.RPSCardMove {
	return a.mctsAgent.Search(ctx)
}

// SetBatchSize sets the batch size for neural network operations
func (a *GPUMCTSAdapter) SetBatchSize(size int) {
	a.mctsAgent.SetBatchSize(size)
}

// GetStats returns performance statistics
func (a *GPUMCTSAdapter) GetStats() map[string]interface{} {
	return a.mctsAgent.GetStats()
}

// Close releases resources
func (a *GPUMCTSAdapter) Close() {
	a.mctsAgent.Close()
}

// This adapter could implement mcts_adapter.MCTSAgent if that interface is still used.
// var _ MCTSAgent = (*GPUMCTSAdapter)(nil)
