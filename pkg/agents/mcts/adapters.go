package mcts

import (
	"context"
	"fmt"

	"github.com/zachbeta/neural_rps/pkg/game"
)

// RPSGameStateAdapter adapts RPSCardGame to the GameState interface
type RPSGameStateAdapter struct {
	*game.RPSCardGame
}

// NewRPSGameStateAdapter creates a new adapter for RPSCardGame
func NewRPSGameStateAdapter(game *game.RPSCardGame) *RPSGameStateAdapter {
	return &RPSGameStateAdapter{game}
}

// Clone implements the GameState interface
func (g *RPSGameStateAdapter) Clone() GameState {
	return &RPSGameStateAdapter{g.RPSCardGame.Copy()}
}

// GetCurrentPlayer implements the GameState interface
func (g *RPSGameStateAdapter) GetCurrentPlayer() game.Player {
	return g.RPSCardGame.CurrentPlayer
}

// GetLastMove returns the last move made
func (g *RPSGameStateAdapter) GetLastMove() game.RPSCardMove {
	return g.RPSCardGame.LastMove
}

// ToTensor converts the game state to tensor representation
func (g *RPSGameStateAdapter) ToTensor() []float32 {
	// Convert the board to a flat representation for the neural network
	features := make([]float32, 0, 64) // Using 64 as a common size for neural input

	// Board state (27 features: 9 positions x 3 states per position)
	for pos := 0; pos < 9; pos++ {
		// One-hot encoding for each position
		isRock := 0.0
		isPaper := 0.0
		isScissors := 0.0
		isEmpty := 1.0

		if g.BoardOwner[pos] != game.NoPlayer {
			isEmpty = 0.0
			if g.Board[pos] == game.Rock {
				isRock = 1.0
			} else if g.Board[pos] == game.Paper {
				isPaper = 1.0
			} else if g.Board[pos] == game.Scissors {
				isScissors = 1.0
			}
		}

		// Add ownership information
		isPlayer1 := 0.0
		isPlayer2 := 0.0
		if g.BoardOwner[pos] == game.Player1 {
			isPlayer1 = 1.0
		} else if g.BoardOwner[pos] == game.Player2 {
			isPlayer2 = 1.0
		}

		features = append(features, float32(isRock), float32(isPaper), float32(isScissors))
		features = append(features, float32(isEmpty), float32(isPlayer1), float32(isPlayer2))
	}

	// Current player
	if g.CurrentPlayer == game.Player1 {
		features = append(features, 1.0, 0.0)
	} else {
		features = append(features, 0.0, 1.0)
	}

	// Player 1 hand (one-hot encoding for each card type)
	rockCount := 0
	paperCount := 0
	scissorsCount := 0
	for _, card := range g.Player1Hand {
		if card == game.Rock {
			rockCount++
		} else if card == game.Paper {
			paperCount++
		} else if card == game.Scissors {
			scissorsCount++
		}
	}
	features = append(features, float32(rockCount)/float32(g.HandSize))
	features = append(features, float32(paperCount)/float32(g.HandSize))
	features = append(features, float32(scissorsCount)/float32(g.HandSize))

	// Player 2 hand (one-hot encoding for each card type)
	rockCount = 0
	paperCount = 0
	scissorsCount = 0
	for _, card := range g.Player2Hand {
		if card == game.Rock {
			rockCount++
		} else if card == game.Paper {
			paperCount++
		} else if card == game.Scissors {
			scissorsCount++
		}
	}
	features = append(features, float32(rockCount)/float32(g.HandSize))
	features = append(features, float32(paperCount)/float32(g.HandSize))
	features = append(features, float32(scissorsCount)/float32(g.HandSize))

	// Round information
	features = append(features, float32(g.Round)/float32(g.MaxRounds))

	// Pad to 64 features if needed
	for len(features) < 64 {
		features = append(features, 0.0)
	}

	return features
}

// GetLegalMoves returns all valid moves for the current state
func (g *RPSGameStateAdapter) GetLegalMoves() []game.RPSCardMove {
	return g.RPSCardGame.GetValidMoves()
}

// ApplyMove applies a move to the current state
func (g *RPSGameStateAdapter) ApplyMove(move game.RPSCardMove) {
	err := g.RPSCardGame.MakeMove(move)
	if err != nil {
		panic(fmt.Sprintf("Invalid move: %v", err))
	}
}

// GPUMCTSAdapter adapts GPUBatchedMCTS to the MCTSAgent interface
type GPUMCTSAdapter struct {
	*GPUBatchedMCTS
}

// NewGPUMCTSAdapter creates a new adapter for GPUBatchedMCTS
func NewGPUMCTSAdapter(serviceAddr string, params MCTSParams) (*GPUMCTSAdapter, error) {
	mcts, err := NewGPUBatchedMCTS(serviceAddr, params)
	if err != nil {
		return nil, err
	}
	return &GPUMCTSAdapter{mcts}, nil
}

// SetRootState sets the root state for search
func (a *GPUMCTSAdapter) SetRootState(state GameState) {
	// The GPUBatchedMCTS.SetRootState now correctly takes a GameState.
	// So, we can pass the 'state' directly.
	a.GPUBatchedMCTS.SetRootState(state)
}

// Search implements the MCTSAgent interface
func (a *GPUMCTSAdapter) Search(ctx context.Context) game.RPSCardMove {
	return a.GPUBatchedMCTS.Search(ctx)
}

// SetBatchSize sets the batch size for the GPU MCTS agent
func (a *GPUMCTSAdapter) SetBatchSize(size int) {
	a.GPUBatchedMCTS.SetBatchSize(size)
}

// GetStats returns statistics about the GPU MCTS agent
func (a *GPUMCTSAdapter) GetStats() map[string]interface{} {
	return a.GPUBatchedMCTS.GetStats()
}

// Close releases resources used by the GPU MCTS agent
func (a *GPUMCTSAdapter) Close() {
	a.GPUBatchedMCTS.Close()
}
