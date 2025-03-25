package game

import (
	"testing"
)

func TestNewAGGame(t *testing.T) {
	game := NewAGGame()

	// Check initial state
	if game.CurrentPlayer != PlayerX {
		t.Errorf("Expected initial player to be X, got %v", game.CurrentPlayer)
	}

	// Check board is empty
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			if game.Board[row][col] != Empty {
				t.Errorf("Expected board position [%d][%d] to be empty, got %v", row, col, game.Board[row][col])
			}
		}
	}

	// Check valid moves
	moves := game.GetValidMoves()
	if len(moves) != 9 {
		t.Errorf("Expected 9 valid moves in new game, got %d", len(moves))
	}
}

func TestMakeMove(t *testing.T) {
	game := NewAGGame()

	// Make a valid move
	err := game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	if err != nil {
		t.Errorf("Unexpected error making move: %v", err)
	}

	// Check board state
	if game.Board[0][0] != PlayerX {
		t.Errorf("Expected board position [0][0] to be X, got %v", game.Board[0][0])
	}

	// Check player switched
	if game.CurrentPlayer != PlayerO {
		t.Errorf("Expected current player to be O after move, got %v", game.CurrentPlayer)
	}

	// Check invalid moves
	err = game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerO})
	if err == nil {
		t.Error("Expected error when making move on occupied cell")
	}

	err = game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerX})
	if err == nil {
		t.Error("Expected error when wrong player makes move")
	}

	err = game.MakeMove(AGMove{Row: 3, Col: 0, Player: PlayerO})
	if err == nil {
		t.Error("Expected error when making move out of bounds")
	}
}

func TestGameOver(t *testing.T) {
	// Test horizontal win
	game := NewAGGame()
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 0, Player: PlayerO})
	game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerO})
	game.MakeMove(AGMove{Row: 0, Col: 2, Player: PlayerX})

	if !game.IsGameOver() {
		t.Error("Expected game to be over with horizontal win")
	}

	if game.GetWinner() != PlayerX {
		t.Errorf("Expected winner to be X, got %v", game.GetWinner())
	}

	// Test vertical win
	game = NewAGGame()
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerO})
	game.MakeMove(AGMove{Row: 1, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerO})
	game.MakeMove(AGMove{Row: 2, Col: 0, Player: PlayerX})

	if !game.IsGameOver() {
		t.Error("Expected game to be over with vertical win")
	}

	if game.GetWinner() != PlayerX {
		t.Errorf("Expected winner to be X, got %v", game.GetWinner())
	}

	// Test diagonal win
	game = NewAGGame()
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerO})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerX})
	game.MakeMove(AGMove{Row: 0, Col: 2, Player: PlayerO})
	game.MakeMove(AGMove{Row: 2, Col: 2, Player: PlayerX})

	if !game.IsGameOver() {
		t.Error("Expected game to be over with diagonal win")
	}

	if game.GetWinner() != PlayerX {
		t.Errorf("Expected winner to be X, got %v", game.GetWinner())
	}

	// Test draw
	game = NewAGGame()
	// X O X
	// X X O
	// O X O
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerO})
	game.MakeMove(AGMove{Row: 0, Col: 2, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 2, Player: PlayerO})
	game.MakeMove(AGMove{Row: 1, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 2, Col: 0, Player: PlayerO})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerX})
	game.MakeMove(AGMove{Row: 2, Col: 2, Player: PlayerO})
	game.MakeMove(AGMove{Row: 2, Col: 1, Player: PlayerX})

	if !game.IsGameOver() {
		t.Error("Expected game to be over with draw")
	}

	if game.GetWinner() != Empty {
		t.Errorf("Expected no winner in draw, got %v", game.GetWinner())
	}
}

func TestGetBoardAsFeatures(t *testing.T) {
	game := NewAGGame()
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerO})

	features := game.GetBoardAsFeatures()

	// Check specific cells
	if features[0] != 1.0 { // X at (0,0) should be 1.0
		t.Errorf("Expected feature 0 to be 1.0, got %f", features[0])
	}

	if features[4] != -1.0 { // O at (1,1) should be -1.0
		t.Errorf("Expected feature 4 to be -1.0, got %f", features[4])
	}

	if features[1] != 0.0 { // Empty at (0,1) should be 0.0
		t.Errorf("Expected feature 1 to be 0.0, got %f", features[1])
	}
}

func TestGameCopy(t *testing.T) {
	game := NewAGGame()
	game.MakeMove(AGMove{Row: 0, Col: 0, Player: PlayerX})
	game.MakeMove(AGMove{Row: 1, Col: 1, Player: PlayerO})

	copy := game.Copy()

	// Check board state is the same
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			if game.Board[row][col] != copy.Board[row][col] {
				t.Errorf("Board mismatch at [%d][%d]: original=%v copy=%v",
					row, col, game.Board[row][col], copy.Board[row][col])
			}
		}
	}

	// Check current player is the same
	if game.CurrentPlayer != copy.CurrentPlayer {
		t.Errorf("Current player mismatch: original=%v copy=%v",
			game.CurrentPlayer, copy.CurrentPlayer)
	}

	// Modify original, check copy is unchanged
	game.MakeMove(AGMove{Row: 0, Col: 1, Player: PlayerX})
	if copy.Board[0][1] != Empty {
		t.Error("Copy was modified when original was changed")
	}
}
