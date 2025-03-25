package game

import (
	"testing"
)

func TestNewRPSGame(t *testing.T) {
	game := NewRPSGame(15, 5, 10)

	// Check initial state
	if game.CurrentPlayer != Player1 {
		t.Errorf("Expected Player1 to start, got %v", game.CurrentPlayer)
	}

	if len(game.Player1Hand) != 5 {
		t.Errorf("Expected 5 cards in Player1 hand, got %d", len(game.Player1Hand))
	}

	if len(game.Player2Hand) != 5 {
		t.Errorf("Expected 5 cards in Player2 hand, got %d", len(game.Player2Hand))
	}

	if game.Round != 1 {
		t.Errorf("Expected round 1, got %d", game.Round)
	}

	if game.MaxRounds != 10 {
		t.Errorf("Expected max rounds 10, got %d", game.MaxRounds)
	}

	// Check that board is empty
	for i, card := range game.Board {
		if card.Owner != NoPlayer {
			t.Errorf("Expected empty board at position %d, but found a card owned by %v", i, card.Owner)
		}
	}
}

func TestGetValidMoves(t *testing.T) {
	game := NewRPSGame(15, 3, 10)

	// Initial state: 9 empty positions * 3 cards = 27 possible moves
	moves := game.GetValidMoves()
	if len(moves) != 27 {
		t.Errorf("Expected 27 valid moves, got %d", len(moves))
	}

	// After playing a move, there should be fewer valid moves
	move := moves[0]
	err := game.MakeMove(move)
	if err != nil {
		t.Errorf("Unexpected error making move: %v", err)
	}

	// Now Player2's turn: 8 empty positions * 3 cards = 24 possible moves
	moves = game.GetValidMoves()
	if len(moves) != 24 {
		t.Errorf("Expected 24 valid moves, got %d", len(moves))
	}
}

func TestRPSMakeMove(t *testing.T) {
	game := NewRPSGame(15, 3, 10)
	initialHand1Size := len(game.Player1Hand)

	// Get a valid move
	moves := game.GetValidMoves()
	move := moves[0]

	// Make the move
	err := game.MakeMove(move)
	if err != nil {
		t.Errorf("Unexpected error making move: %v", err)
	}

	// Check that the move was applied
	if game.Board[move.Position].Owner != Player1 {
		t.Errorf("Expected board position %d to be owned by Player1, got %v",
			move.Position, game.Board[move.Position].Owner)
	}

	// Check that the card was removed from hand
	if len(game.Player1Hand) != initialHand1Size-1 {
		t.Errorf("Expected Player1 hand to have %d cards, got %d",
			initialHand1Size-1, len(game.Player1Hand))
	}

	// Check that it's now Player2's turn
	if game.CurrentPlayer != Player2 {
		t.Errorf("Expected current player to be Player2, got %v", game.CurrentPlayer)
	}

	// Try an invalid move
	invalidMove := RPSMove{
		CardIndex: 0,
		Position:  move.Position, // Already occupied
		Player:    Player2,
	}

	err = game.MakeMove(invalidMove)
	if err == nil {
		t.Errorf("Expected error for invalid move, got nil")
	}
}

func TestCaptureLogic(t *testing.T) {
	game := NewRPSGame(15, 5, 10)

	// Create a controlled setup by directly modifying the game state
	// Place a Rock card for Player1 at position 0,0
	game.Board[0] = RPSCard{Type: Rock, Owner: Player1}

	// Place a Scissors card for Player2 at position 0,1 (adjacent)
	game.Board[1] = RPSCard{Type: Scissors, Owner: Player2}

	// Place a Paper card for Player2 at position 1,0 (adjacent)
	game.Board[3] = RPSCard{Type: Paper, Owner: Player2}

	// Rock should beat Scissors but lose to Paper
	game.processCapturesAt(0)

	// Check that the Scissors card was captured
	if game.Board[1].Owner != Player1 {
		t.Errorf("Expected Scissors card to be captured by Player1, got owner %v", game.Board[1].Owner)
	}

	// Check that the Paper card was not captured
	if game.Board[3].Owner != Player2 {
		t.Errorf("Expected Paper card to remain owned by Player2, got owner %v", game.Board[3].Owner)
	}
}

func TestGameEnd(t *testing.T) {
	// Create a game with small hands so it's easier to test
	game := NewRPSGame(6, 3, 5)

	// Game shouldn't be over initially
	if game.IsGameOver() {
		t.Errorf("Expected game not to be over, but it was")
	}

	// Play all cards from both hands
	for !game.IsGameOver() {
		move, err := game.GetRandomMove()
		if err != nil {
			t.Fatalf("Unexpected error getting random move: %v", err)
		}

		err = game.MakeMove(move)
		if err != nil {
			t.Fatalf("Unexpected error making move: %v", err)
		}
	}

	// After playing all cards, game should be over
	if !game.IsGameOver() {
		t.Errorf("Expected game to be over, but it wasn't")
	}

	// Test winner determination
	// This will vary based on how the random game played out
	winner := game.GetWinner()
	if winner != Player1 && winner != Player2 && winner != NoPlayer {
		t.Errorf("Invalid winner: %v", winner)
	}
}

func TestCopy(t *testing.T) {
	original := NewRPSGame(15, 4, 10)

	// Make a move
	move, _ := original.GetRandomMove()
	original.MakeMove(move)

	// Create a copy
	copy := original.Copy()

	// Check that the copy is a deep copy
	if &original.Board == &copy.Board {
		t.Errorf("Board was not deep copied")
	}

	if &original.Player1Hand == &copy.Player1Hand {
		t.Errorf("Player1Hand was not deep copied")
	}

	if &original.Player2Hand == &copy.Player2Hand {
		t.Errorf("Player2Hand was not deep copied")
	}

	// Make sure the state was copied correctly
	if original.CurrentPlayer != copy.CurrentPlayer {
		t.Errorf("CurrentPlayer not copied correctly: expected %v, got %v",
			original.CurrentPlayer, copy.CurrentPlayer)
	}

	// Check that modifications to one don't affect the other
	originalBoard := original.Board
	copy.Board[0] = RPSCard{Type: Rock, Owner: Player1}

	if original.Board[0].Owner == Player1 && original.Board[0].Type == Rock {
		t.Errorf("Modifying copy affected the original")
	}

	// Restore the original
	original.Board = originalBoard
}

func TestRPSGetBoardAsFeatures(t *testing.T) {
	game := NewRPSGame(15, 5, 10)

	// Initially, all features should be set to the default values
	features := game.GetBoardAsFeatures()

	if len(features) != 81 {
		t.Errorf("Expected 81 features, got %d", len(features))
	}

	// Place some cards and check that features are updated correctly
	game.Board[0] = RPSCard{Type: Rock, Owner: Player1}
	game.Board[4] = RPSCard{Type: Paper, Owner: Player2}

	features = game.GetBoardAsFeatures()

	// Check Rock at position 0
	if features[0] != 1.0 { // Rock type
		t.Errorf("Expected Rock feature at position 0, got %f", features[0])
	}
	if features[0+3+1] != 1.0 { // Player1 ownership
		t.Errorf("Expected Player1 ownership feature, got %f", features[0+3+1])
	}

	// Check Paper at position 4
	if features[4*9+1] != 1.0 { // Paper type
		t.Errorf("Expected Paper feature at position 4, got %f", features[4*9+1])
	}
	if features[4*9+3+2] != 1.0 { // Player2 ownership
		t.Errorf("Expected Player2 ownership feature, got %f", features[4*9+3+2])
	}
}
