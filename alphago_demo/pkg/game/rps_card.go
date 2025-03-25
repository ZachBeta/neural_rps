package game

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
)

// RPSCardType represents a card type in RPS
type RPSCardType int

const (
	Rock     RPSCardType = 0
	Paper    RPSCardType = 1
	Scissors RPSCardType = 2
)

// RPSPlayer represents a player in the game
type RPSPlayer int

const (
	NoPlayer RPSPlayer = 0
	Player1  RPSPlayer = 1
	Player2  RPSPlayer = 2
)

// RPSCard represents a card in the game
type RPSCard struct {
	Type  RPSCardType
	Owner RPSPlayer // Initially NoPlayer, changes when card is played
}

// RPSMove represents a move in the game
type RPSMove struct {
	CardIndex int // Index in hand
	Position  int // Position on the board (0-8 for a 3x3 board)
	Player    RPSPlayer
}

// RPSGame represents the game state
type RPSGame struct {
	Board         [9]RPSCard // 3x3 board
	Player1Hand   []RPSCard
	Player2Hand   []RPSCard
	CurrentPlayer RPSPlayer
	MoveHistory   []RPSMove
	Round         int
	MaxRounds     int
}

// NewRPSGame creates a new RPS card game
func NewRPSGame(deckSize int, handSize int, maxRounds int) *RPSGame {
	game := &RPSGame{
		Board:         [9]RPSCard{},
		Player1Hand:   make([]RPSCard, 0, handSize),
		Player2Hand:   make([]RPSCard, 0, handSize),
		CurrentPlayer: Player1, // Player1 goes first
		MoveHistory:   []RPSMove{},
		Round:         1,
		MaxRounds:     maxRounds,
	}

	// Generate deck
	deck := generateDeck(deckSize)

	// Deal cards
	game.dealCards(deck, handSize)

	return game
}

// generateDeck creates a deck of cards with roughly equal distribution of types
func generateDeck(size int) []RPSCard {
	deck := make([]RPSCard, size)
	for i := 0; i < size; i++ {
		cardType := RPSCardType(i % 3) // Cycle through Rock, Paper, Scissors
		deck[i] = RPSCard{Type: cardType, Owner: NoPlayer}
	}

	// Shuffle deck
	rand.Shuffle(len(deck), func(i, j int) {
		deck[i], deck[j] = deck[j], deck[i]
	})

	return deck
}

// dealCards deals cards to both players
func (g *RPSGame) dealCards(deck []RPSCard, handSize int) {
	for i := 0; i < handSize*2 && i < len(deck); i++ {
		if i < handSize {
			g.Player1Hand = append(g.Player1Hand, deck[i])
		} else {
			g.Player2Hand = append(g.Player2Hand, deck[i])
		}
	}
}

// GetValidMoves returns all valid moves for the current player
func (g *RPSGame) GetValidMoves() []RPSMove {
	var moves []RPSMove
	var hand []RPSCard

	if g.CurrentPlayer == Player1 {
		hand = g.Player1Hand
	} else {
		hand = g.Player2Hand
	}

	// Find empty positions on the board
	for pos := 0; pos < 9; pos++ {
		if g.Board[pos].Owner == NoPlayer {
			// For each card in hand
			for i := range hand {
				moves = append(moves, RPSMove{
					CardIndex: i,
					Position:  pos,
					Player:    g.CurrentPlayer,
				})
			}
		}
	}

	return moves
}

// MakeMove applies a move to the game state
func (g *RPSGame) MakeMove(move RPSMove) error {
	// Check if the move is valid
	if move.Position < 0 || move.Position >= 9 {
		return errors.New("position is out of bounds")
	}
	if g.Board[move.Position].Owner != NoPlayer {
		return errors.New("position is already occupied")
	}
	if move.Player != g.CurrentPlayer {
		return errors.New("not the player's turn")
	}

	var hand *[]RPSCard
	if move.Player == Player1 {
		hand = &g.Player1Hand
	} else {
		hand = &g.Player2Hand
	}

	if move.CardIndex < 0 || move.CardIndex >= len(*hand) {
		return errors.New("invalid card index")
	}

	// Apply the move
	card := (*hand)[move.CardIndex]
	card.Owner = move.Player
	g.Board[move.Position] = card

	// Remove card from hand
	*hand = append((*hand)[:move.CardIndex], (*hand)[move.CardIndex+1:]...)

	// Add to move history
	g.MoveHistory = append(g.MoveHistory, move)

	// Switch player
	if g.CurrentPlayer == Player1 {
		g.CurrentPlayer = Player2
	} else {
		g.CurrentPlayer = Player1
		g.Round++
	}

	// Check for captures
	g.processCapturesAt(move.Position)

	return nil
}

// processCapturesAt checks and processes potential captures around the given position
func (g *RPSGame) processCapturesAt(position int) {
	row := position / 3
	col := position % 3

	// Positions to check (adjacent positions: up, right, down, left)
	directions := []struct{ dr, dc int }{
		{-1, 0}, {0, 1}, {1, 0}, {0, -1},
	}

	for _, dir := range directions {
		newRow := row + dir.dr
		newCol := col + dir.dc

		// Check if position is within bounds
		if newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3 {
			newPos := newRow*3 + newCol

			// If there's a card and it belongs to the opponent
			if g.Board[newPos].Owner != NoPlayer && g.Board[newPos].Owner != g.Board[position].Owner {
				// Check if our card beats theirs
				if g.cardBeats(g.Board[position], g.Board[newPos]) {
					// Capture the card
					captured := g.Board[newPos]
					captured.Owner = g.Board[position].Owner
					g.Board[newPos] = captured
				}
			}
		}
	}
}

// cardBeats checks if card1 beats card2 in RPS
func (g *RPSGame) cardBeats(card1, card2 RPSCard) bool {
	switch card1.Type {
	case Rock:
		return card2.Type == Scissors
	case Paper:
		return card2.Type == Rock
	case Scissors:
		return card2.Type == Paper
	}
	return false
}

// IsGameOver checks if the game is over
func (g *RPSGame) IsGameOver() bool {
	// Game ends when both players are out of cards, maximum rounds reached, or no valid moves for current player
	if len(g.Player1Hand) == 0 && len(g.Player2Hand) == 0 {
		return true
	}

	if g.Round > g.MaxRounds {
		return true
	}

	// Check if current player has valid moves
	if len(g.GetValidMoves()) == 0 {
		return true
	}

	return false
}

// GetWinner returns the winner of the game
func (g *RPSGame) GetWinner() RPSPlayer {
	// Count cards owned by each player
	player1Count := 0
	player2Count := 0

	// Count only cards on the board
	for _, card := range g.Board {
		if card.Owner == Player1 {
			player1Count++
		} else if card.Owner == Player2 {
			player2Count++
		}
	}

	// Hand cards don't count towards victory - only cards on the board matter

	if player1Count > player2Count {
		return Player1
	} else if player2Count > player1Count {
		return Player2
	}

	return NoPlayer // Draw
}

// GetRandomMove returns a random valid move
func (g *RPSGame) GetRandomMove() (RPSMove, error) {
	moves := g.GetValidMoves()
	if len(moves) == 0 {
		return RPSMove{}, errors.New("no valid moves")
	}
	return moves[rand.Intn(len(moves))], nil
}

// Copy creates a deep copy of the game
func (g *RPSGame) Copy() *RPSGame {
	newGame := &RPSGame{
		CurrentPlayer: g.CurrentPlayer,
		MoveHistory:   make([]RPSMove, len(g.MoveHistory)),
		Round:         g.Round,
		MaxRounds:     g.MaxRounds,
	}
	copy(newGame.MoveHistory, g.MoveHistory)

	// Copy the board
	for i := range g.Board {
		newGame.Board[i] = g.Board[i]
	}

	// Copy hands
	newGame.Player1Hand = make([]RPSCard, len(g.Player1Hand))
	copy(newGame.Player1Hand, g.Player1Hand)

	newGame.Player2Hand = make([]RPSCard, len(g.Player2Hand))
	copy(newGame.Player2Hand, g.Player2Hand)

	return newGame
}

// GetBoardAsFeatures returns the board as a flattened feature vector
// For each position: 3 features for card type (one-hot) * 3 features for ownership (one-hot)
// So 9 features per position * 9 positions = 81 features
func (g *RPSGame) GetBoardAsFeatures() []float64 {
	features := make([]float64, 81)

	for pos := 0; pos < 9; pos++ {
		card := g.Board[pos]
		baseIdx := pos * 9

		// Card type
		if card.Owner != NoPlayer {
			typeIdx := int(card.Type)
			features[baseIdx+typeIdx] = 1.0
		}

		// Card ownership
		ownerIdx := int(card.Owner) + 3
		features[baseIdx+ownerIdx] = 1.0

		// Current player
		if g.CurrentPlayer == Player1 {
			features[baseIdx+6] = 1.0
		} else {
			features[baseIdx+7] = 1.0
		}
	}

	return features
}

// String returns a string representation of the game
func (g *RPSGame) String() string {
	var sb strings.Builder

	// Display board
	sb.WriteString("  0 1 2\n")
	for row := 0; row < 3; row++ {
		sb.WriteString(fmt.Sprintf("%d ", row))
		for col := 0; col < 3; col++ {
			pos := row*3 + col
			card := g.Board[pos]

			if card.Owner == NoPlayer {
				sb.WriteString(".")
			} else {
				var symbol string
				switch card.Type {
				case Rock:
					symbol = "R"
				case Paper:
					symbol = "P"
				case Scissors:
					symbol = "S"
				}

				if card.Owner == Player1 {
					symbol = strings.ToUpper(symbol)
				} else {
					symbol = strings.ToLower(symbol)
				}

				sb.WriteString(symbol)
			}

			if col < 2 {
				sb.WriteString(" ")
			}
		}
		sb.WriteString("\n")
	}

	// Display hands
	sb.WriteString("\nPlayer 1 Hand: ")
	for _, card := range g.Player1Hand {
		switch card.Type {
		case Rock:
			sb.WriteString("R ")
		case Paper:
			sb.WriteString("P ")
		case Scissors:
			sb.WriteString("S ")
		}
	}

	sb.WriteString("\nPlayer 2 Hand: ")
	for _, card := range g.Player2Hand {
		switch card.Type {
		case Rock:
			sb.WriteString("r ")
		case Paper:
			sb.WriteString("p ")
		case Scissors:
			sb.WriteString("s ")
		}
	}

	// Add card counts (DEBUG INFO)
	var player1Count, player2Count int
	for _, card := range g.Board {
		if card.Owner == Player1 {
			player1Count++
		} else if card.Owner == Player2 {
			player2Count++
		}
	}
	sb.WriteString(fmt.Sprintf("\n\nCard Counts: Player 1 (UPPERCASE): %d, Player 2 (lowercase): %d", player1Count, player2Count))

	// Add current player and game status
	sb.WriteString("\n")
	if g.IsGameOver() {
		winner := g.GetWinner()
		if winner == NoPlayer {
			sb.WriteString("Game over: Draw")
		} else if winner == Player1 {
			sb.WriteString("Game over: Player 1 wins")
		} else {
			sb.WriteString("Game over: Player 2 wins")
		}
	} else {
		if g.CurrentPlayer == Player1 {
			sb.WriteString("Current player: Player 1")
		} else {
			sb.WriteString("Current player: Player 2")
		}
		sb.WriteString(fmt.Sprintf(" (Round %d/%d)", g.Round, g.MaxRounds))
	}

	return sb.String()
}
