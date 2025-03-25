package game

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
)

// AGPlayer represents a player in the game
type AGPlayer int

const (
	Empty   AGPlayer = 0
	PlayerX AGPlayer = 1
	PlayerO AGPlayer = 2
)

// AGBoard represents a 3x3 Tic-Tac-Toe board
type AGBoard [3][3]AGPlayer

// AGGame represents the game state
type AGGame struct {
	Board         AGBoard
	CurrentPlayer AGPlayer
	MoveHistory   []AGMove
}

// AGMove represents a move in the game
type AGMove struct {
	Row    int
	Col    int
	Player AGPlayer
}

// NewAGGame creates a new Tic-Tac-Toe game
func NewAGGame() *AGGame {
	return &AGGame{
		Board:         AGBoard{},
		CurrentPlayer: PlayerX, // X goes first
		MoveHistory:   []AGMove{},
	}
}

// GetValidMoves returns a list of valid moves for the current player
func (g *AGGame) GetValidMoves() []AGMove {
	var moves []AGMove
	if g.IsGameOver() {
		return moves
	}

	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			if g.Board[row][col] == Empty {
				moves = append(moves, AGMove{Row: row, Col: col})
			}
		}
	}
	return moves
}

// IsValidMove checks if a move is valid for the current game state
func (g *AGGame) IsValidMove(move AGMove) bool {
	// Check if the game is over
	if g.IsGameOver() {
		return false
	}

	// Check if the move is within bounds
	if move.Row < 0 || move.Row >= 3 || move.Col < 0 || move.Col >= 3 {
		return false
	}

	// Check if the position is empty
	return g.Board[move.Row][move.Col] == Empty
}

// MakeMove applies a move to the game state
func (g *AGGame) MakeMove(move AGMove) error {
	// Set the player for this move
	move.Player = g.CurrentPlayer

	// Check if the move is valid
	if move.Row < 0 || move.Row >= 3 || move.Col < 0 || move.Col >= 3 {
		return errors.New("move is out of bounds")
	}
	if g.Board[move.Row][move.Col] != Empty {
		return errors.New("cell is already occupied")
	}

	// Apply the move
	g.Board[move.Row][move.Col] = move.Player
	g.MoveHistory = append(g.MoveHistory, move)

	// Switch player
	if g.CurrentPlayer == PlayerX {
		g.CurrentPlayer = PlayerO
	} else {
		g.CurrentPlayer = PlayerX
	}

	return nil
}

// IsGameOver checks if the game is over
func (g *AGGame) IsGameOver() bool {
	// Check if there's a winner
	if g.GetWinner() != Empty {
		return true
	}

	// Check if the board is full (no empty spaces)
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			if g.Board[row][col] == Empty {
				return false
			}
		}
	}
	return true
}

// GetWinner returns the winner of the game, or Empty if there is no winner yet
func (g *AGGame) GetWinner() AGPlayer {
	// Check rows
	for row := 0; row < 3; row++ {
		if g.Board[row][0] != Empty && g.Board[row][0] == g.Board[row][1] && g.Board[row][1] == g.Board[row][2] {
			return g.Board[row][0]
		}
	}

	// Check columns
	for col := 0; col < 3; col++ {
		if g.Board[0][col] != Empty && g.Board[0][col] == g.Board[1][col] && g.Board[1][col] == g.Board[2][col] {
			return g.Board[0][col]
		}
	}

	// Check diagonals
	if g.Board[0][0] != Empty && g.Board[0][0] == g.Board[1][1] && g.Board[1][1] == g.Board[2][2] {
		return g.Board[0][0]
	}
	if g.Board[0][2] != Empty && g.Board[0][2] == g.Board[1][1] && g.Board[1][1] == g.Board[2][0] {
		return g.Board[0][2]
	}

	return Empty
}

// GetRandomMove returns a random valid move
func (g *AGGame) GetRandomMove() (AGMove, error) {
	moves := g.GetValidMoves()
	if len(moves) == 0 {
		return AGMove{}, errors.New("no valid moves")
	}
	return moves[rand.Intn(len(moves))], nil
}

// Copy creates a deep copy of the game
func (g *AGGame) Copy() *AGGame {
	newGame := &AGGame{
		CurrentPlayer: g.CurrentPlayer,
		MoveHistory:   make([]AGMove, len(g.MoveHistory)),
	}
	copy(newGame.MoveHistory, g.MoveHistory)

	// Copy the board
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			newGame.Board[row][col] = g.Board[row][col]
		}
	}

	return newGame
}

// GetBoardAsFeatures returns the board as a flattened feature vector
// X = 1, O = -1, Empty = 0
func (g *AGGame) GetBoardAsFeatures() []float64 {
	features := make([]float64, 9)
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			idx := row*3 + col
			switch g.Board[row][col] {
			case PlayerX:
				features[idx] = 1.0
			case PlayerO:
				features[idx] = -1.0
			case Empty:
				features[idx] = 0.0
			}
		}
	}
	return features
}

// String returns a string representation of the game
func (g *AGGame) String() string {
	var sb strings.Builder
	sb.WriteString("  0 1 2\n")
	for row := 0; row < 3; row++ {
		sb.WriteString(fmt.Sprintf("%d ", row))
		for col := 0; col < 3; col++ {
			switch g.Board[row][col] {
			case Empty:
				sb.WriteString(".")
			case PlayerX:
				sb.WriteString("X")
			case PlayerO:
				sb.WriteString("O")
			}
			if col < 2 {
				sb.WriteString(" ")
			}
		}
		sb.WriteString("\n")
	}

	// Add current player and game status
	if g.IsGameOver() {
		winner := g.GetWinner()
		if winner == Empty {
			sb.WriteString("Game over: Draw")
		} else if winner == PlayerX {
			sb.WriteString("Game over: X wins")
		} else {
			sb.WriteString("Game over: O wins")
		}
	} else {
		if g.CurrentPlayer == PlayerX {
			sb.WriteString("Current player: X")
		} else {
			sb.WriteString("Current player: O")
		}
	}

	return sb.String()
}
