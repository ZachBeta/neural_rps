package game

import (
	"fmt"
	"math/rand"
)

// Player represents a player in the game
type Player int

const (
	Player1 Player = iota
	Player2
	NoPlayer
)

// RPSCardType represents card types
type RPSCardType int

const (
	Rock RPSCardType = iota
	Paper
	Scissors
)

// RPSCardMove represents a move in the RPS card game
type RPSCardMove struct {
	CardIndex int    // Index of the card in hand
	Position  int    // Position on the board (0-8)
	Player    Player // Player making the move
}

// RPSCardGame represents the RPS card game environment
type RPSCardGame struct {
	Board         [9]RPSCardType // 3x3 board
	BoardOwner    [9]Player      // Who owns each position
	Player1Hand   []RPSCardType  // Player 1's hand
	Player2Hand   []RPSCardType  // Player 2's hand
	CurrentPlayer Player         // Current player
	Round         int            // Current round
	MaxRounds     int            // Maximum rounds
	DeckSize      int            // Initial deck size
	HandSize      int            // Initial hand size
}

// NewRPSCardGame creates a new RPS card game
func NewRPSCardGame(deckSize, handSize, maxRounds int) *RPSCardGame {
	game := &RPSCardGame{
		CurrentPlayer: Player1,
		Round:         1,
		MaxRounds:     maxRounds,
		DeckSize:      deckSize,
		HandSize:      handSize,
	}

	// Initialize board
	for i := range game.Board {
		game.Board[i] = -1 // Empty
		game.BoardOwner[i] = NoPlayer
	}

	// Initialize hands
	game.Player1Hand = make([]RPSCardType, handSize)
	game.Player2Hand = make([]RPSCardType, handSize)

	// Fill hands with random cards
	for i := 0; i < handSize; i++ {
		game.Player1Hand[i] = RPSCardType(rand.Intn(3))
		game.Player2Hand[i] = RPSCardType(rand.Intn(3))
	}

	return game
}

// Copy returns a deep copy of the game
func (g *RPSCardGame) Copy() *RPSCardGame {
	gameCopy := &RPSCardGame{
		CurrentPlayer: g.CurrentPlayer,
		Round:         g.Round,
		MaxRounds:     g.MaxRounds,
		DeckSize:      g.DeckSize,
		HandSize:      g.HandSize,
	}

	// Copy board
	copy(gameCopy.Board[:], g.Board[:])
	copy(gameCopy.BoardOwner[:], g.BoardOwner[:])

	// Copy hands
	gameCopy.Player1Hand = make([]RPSCardType, len(g.Player1Hand))
	copy(gameCopy.Player1Hand, g.Player1Hand)

	gameCopy.Player2Hand = make([]RPSCardType, len(g.Player2Hand))
	copy(gameCopy.Player2Hand, g.Player2Hand)

	return gameCopy
}

// GetValidMoves returns all valid moves for the current player
func (g *RPSCardGame) GetValidMoves() []RPSCardMove {
	validMoves := make([]RPSCardMove, 0)

	// Determine current hand
	var hand []RPSCardType
	if g.CurrentPlayer == Player1 {
		hand = g.Player1Hand
	} else {
		hand = g.Player2Hand
	}

	// For each card in hand, check all possible placements
	for cardIdx, _ := range hand {
		for pos := 0; pos < 9; pos++ {
			// Check if position is empty
			if g.BoardOwner[pos] == NoPlayer {
				move := RPSCardMove{
					CardIndex: cardIdx,
					Position:  pos,
					Player:    g.CurrentPlayer,
				}
				validMoves = append(validMoves, move)
			}
		}
	}

	return validMoves
}

// GetRandomMove returns a random valid move
func (g *RPSCardGame) GetRandomMove() (RPSCardMove, error) {
	validMoves := g.GetValidMoves()
	if len(validMoves) == 0 {
		return RPSCardMove{}, fmt.Errorf("no valid moves")
	}
	return validMoves[rand.Intn(len(validMoves))], nil
}

// MakeMove makes a move in the game
func (g *RPSCardGame) MakeMove(move RPSCardMove) error {
	// Validate player
	if move.Player != g.CurrentPlayer {
		return fmt.Errorf("not your turn")
	}

	// Validate position
	if g.BoardOwner[move.Position] != NoPlayer {
		return fmt.Errorf("position already occupied")
	}

	// Validate card index
	var hand []RPSCardType
	if g.CurrentPlayer == Player1 {
		hand = g.Player1Hand
	} else {
		hand = g.Player2Hand
	}

	if move.CardIndex < 0 || move.CardIndex >= len(hand) {
		return fmt.Errorf("invalid card index")
	}

	// Place the card
	cardType := hand[move.CardIndex]
	g.Board[move.Position] = cardType
	g.BoardOwner[move.Position] = g.CurrentPlayer

	// Remove card from hand
	if g.CurrentPlayer == Player1 {
		g.Player1Hand = append(g.Player1Hand[:move.CardIndex], g.Player1Hand[move.CardIndex+1:]...)
	} else {
		g.Player2Hand = append(g.Player2Hand[:move.CardIndex], g.Player2Hand[move.CardIndex+1:]...)
	}

	// Switch player
	if g.CurrentPlayer == Player1 {
		g.CurrentPlayer = Player2
	} else {
		g.CurrentPlayer = Player1
		g.Round++
	}

	return nil
}

// IsGameOver checks if the game is over
func (g *RPSCardGame) IsGameOver() bool {
	// Game is over if any player has no cards left
	if len(g.Player1Hand) == 0 || len(g.Player2Hand) == 0 {
		return true
	}

	// Game is over if we reached max rounds
	if g.Round > g.MaxRounds {
		return true
	}

	// Game is over if the board is full
	for _, owner := range g.BoardOwner {
		if owner == NoPlayer {
			return false
		}
	}

	return true
}

// GetWinner returns the winner of the game
func (g *RPSCardGame) GetWinner() Player {
	// Count cards on board
	player1Count := 0
	player2Count := 0

	for _, owner := range g.BoardOwner {
		if owner == Player1 {
			player1Count++
		} else if owner == Player2 {
			player2Count++
		}
	}

	if player1Count > player2Count {
		return Player1
	} else if player2Count > player1Count {
		return Player2
	}

	return NoPlayer // Draw
}

// GetBoardAsFeatures returns the board state as a feature vector for neural network input
func (g *RPSCardGame) GetBoardAsFeatures() []float64 {
	// Use the same format as the alphago_demo implementation
	// 9 board positions * 9 possible states (3 card types * 3 ownership states) = 81 inputs
	features := make([]float64, 81)

	for pos := 0; pos < 9; pos++ {
		// Skip empty positions
		if g.BoardOwner[pos] == NoPlayer {
			continue
		}

		// Calculate feature index:
		// Base index for this position + card type + ownership offset
		var indexOffset int
		if g.BoardOwner[pos] == Player1 {
			indexOffset = 0 // Player 1's cards use first 3 indices
		} else {
			indexOffset = 3 // Player 2's cards use next 3 indices
		}

		// Set the feature
		index := pos*9 + int(g.Board[pos]) + indexOffset
		features[index] = 1.0
	}

	return features
}

// String returns a string representation of the game
func (g *RPSCardGame) String() string {
	// Create an ASCII representation
	result := "  0 1 2\n"

	for row := 0; row < 3; row++ {
		result += fmt.Sprintf("%d ", row)
		for col := 0; col < 3; col++ {
			pos := row*3 + col
			if g.BoardOwner[pos] == NoPlayer {
				result += ". "
			} else {
				var symbol string
				switch g.Board[pos] {
				case Rock:
					symbol = "R"
				case Paper:
					symbol = "P"
				case Scissors:
					symbol = "S"
				}

				if g.BoardOwner[pos] == Player2 {
					symbol = symbol + " "
				} else {
					symbol = symbol + " "
				}

				result += symbol
			}
		}
		result += "\n"
	}

	result += "\nPlayer 1 Hand: "
	for _, card := range g.Player1Hand {
		var symbol string
		switch card {
		case Rock:
			symbol = "R "
		case Paper:
			symbol = "P "
		case Scissors:
			symbol = "S "
		}
		result += symbol
	}

	result += "\nPlayer 2 Hand: "
	for _, card := range g.Player2Hand {
		var symbol string
		switch card {
		case Rock:
			symbol = "r "
		case Paper:
			symbol = "p "
		case Scissors:
			symbol = "s "
		}
		result += symbol
	}

	result += fmt.Sprintf("\n\nCard Counts: Player 1 (UPPERCASE): %d, Player 2 (lowercase): %d\n", len(g.Player1Hand), len(g.Player2Hand))
	result += fmt.Sprintf("Current player: %s (Round %d/%d)\n", g.CurrentPlayer, g.Round, g.MaxRounds)

	return result
}
