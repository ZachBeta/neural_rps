package analysis

import (
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// StandardEvaluator provides a comprehensive evaluation function
func StandardEvaluator(state *game.RPSGame) float64 {
	if state.IsGameOver() {
		winner := state.GetWinner()
		if winner == game.Player1 {
			return 1000.0 // Large positive value for Player1 win
		} else if winner == game.Player2 {
			return -1000.0 // Large negative value for Player2 win
		}
		return 0.0 // Draw
	}

	// Combine multiple evaluation factors with appropriate weights
	return materialScore(state)*1.0 + positionalScore(state)*0.5 + relationshipScore(state)*0.8
}

// materialScore evaluates the material advantage (difference in number of cards)
func materialScore(state *game.RPSGame) float64 {
	p1Cards := state.CountPlayerCards(game.Player1)
	p2Cards := state.CountPlayerCards(game.Player2)
	return float64(p1Cards-p2Cards) * 10.0
}

// positionalScore evaluates board control and positioning
func positionalScore(state *game.RPSGame) float64 {
	score := 0.0

	// Define position values (center is most valuable, corners next)
	positionValues := [][]float64{
		{0.7, 0.5, 0.7}, // Top row
		{0.5, 1.0, 0.5}, // Middle row (center = 1.0)
		{0.7, 0.5, 0.7}, // Bottom row
	}

	// Calculate positional score based on occupied positions
	board := state.GetBoard()
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			pos := row*3 + col
			card := board[pos]
			if card.Owner == game.Player1 {
				score += positionValues[row][col]
			} else if card.Owner == game.Player2 {
				score -= positionValues[row][col]
			}
		}
	}

	return score * 5.0 // Weight position appropriately
}

// relationshipScore evaluates the RPS relationships between adjacent cards
func relationshipScore(state *game.RPSGame) float64 {
	score := 0.0
	board := state.GetBoard()

	// Define directional offsets to check adjacency (horizontal, vertical, diagonal)
	directions := []struct{ dRow, dCol int }{
		{0, 1}, {1, 0}, {1, 1}, {1, -1}, // Right, Down, Diagonal down-right, Diagonal down-left
	}

	// Check each cell on the board
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			pos := row*3 + col
			cell := board[pos]
			if cell.Owner == game.NoPlayer {
				continue // Skip empty cells
			}

			// Check adjacent cells in all directions
			for _, dir := range directions {
				newRow, newCol := row+dir.dRow, col+dir.dCol

				// Check if position is within bounds
				if newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3 {
					newPos := newRow*3 + newCol
					adjCell := board[newPos]
					if adjCell.Owner != game.NoPlayer && adjCell.Owner != cell.Owner {
						// Calculate advantage based on RPS relationships
						advantage := getCardAdvantage(cell.Type, adjCell.Type)

						if cell.Owner == game.Player1 {
							score += advantage
						} else {
							score -= advantage
						}
					}
				}
			}
		}
	}

	return score * 3.0 // Weight relationships appropriately
}

// getCardAdvantage returns the advantage of card1 over card2
// 1.0 if card1 beats card2, -1.0 if card2 beats card1, 0.0 if tie
func getCardAdvantage(card1, card2 game.RPSCardType) float64 {
	if card1 == card2 {
		return 0.0 // Same card type
	}

	// Rock beats Scissors, Scissors beats Paper, Paper beats Rock
	if (card1 == game.Rock && card2 == game.Scissors) ||
		(card1 == game.Scissors && card2 == game.Paper) ||
		(card1 == game.Paper && card2 == game.Rock) {
		return 1.0
	}

	return -1.0
}
