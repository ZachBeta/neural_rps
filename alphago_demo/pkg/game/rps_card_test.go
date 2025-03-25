package game

import (
	"strings"
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

func TestWinnerDeterminationWithMoreCards(t *testing.T) {
	// Create a new game
	game := NewRPSGame(15, 5, 10)

	// Create a specific board configuration where Player1 has more cards
	// Board:
	//   0 1 2
	// 0 s P R
	// 1 s R R
	// 2 p r P
	// Player1 (uppercase): 5 cards
	// Player2 (lowercase): 4 cards

	// Clear the board first (in case there are default values)
	for i := range game.Board {
		game.Board[i] = RPSCard{Type: Rock, Owner: NoPlayer}
	}

	// Set up the board
	game.Board[0] = RPSCard{Type: Scissors, Owner: Player2} // Position 0,0: s
	game.Board[1] = RPSCard{Type: Paper, Owner: Player1}    // Position 0,1: P
	game.Board[2] = RPSCard{Type: Rock, Owner: Player1}     // Position 0,2: R
	game.Board[3] = RPSCard{Type: Scissors, Owner: Player2} // Position 1,0: s
	game.Board[4] = RPSCard{Type: Rock, Owner: Player1}     // Position 1,1: R
	game.Board[5] = RPSCard{Type: Rock, Owner: Player1}     // Position 1,2: R
	game.Board[6] = RPSCard{Type: Paper, Owner: Player2}    // Position 2,0: p
	game.Board[7] = RPSCard{Type: Rock, Owner: Player2}     // Position 2,1: r
	game.Board[8] = RPSCard{Type: Paper, Owner: Player1}    // Position 2,2: P

	// Empty the hands to simulate end of game
	game.Player1Hand = []RPSCard{}
	game.Player2Hand = []RPSCard{}

	// Manually set Round to be greater than MaxRounds to ensure game is over
	game.Round = game.MaxRounds + 1

	// Verify game is over
	if !game.IsGameOver() {
		t.Fatal("Game should be over but IsGameOver() returned false")
	}

	// Test winner determination
	winner := game.GetWinner()

	// Debug print
	var player1Count, player2Count int
	for _, card := range game.Board {
		if card.Owner == Player1 {
			player1Count++
		} else if card.Owner == Player2 {
			player2Count++
		}
	}
	t.Logf("Debug: Player1 has %d cards, Player2 has %d cards", player1Count, player2Count)

	// Player1 should win (5 cards vs 4 cards)
	if winner != Player1 {
		t.Errorf("Expected Player1 to win (with 5 cards vs 4), but got %v", winner)
	}
}

func TestBoardStringRepresentation(t *testing.T) {
	// Create a new game
	game := NewRPSGame(15, 5, 10)

	// Set up the same board configuration as in the winner determination test
	// Board:
	//   0 1 2
	// 0 s P R
	// 1 s R R
	// 2 p r P
	// Player1 (uppercase): 5 cards
	// Player2 (lowercase): 4 cards

	// Clear the board first (in case there are default values)
	for i := range game.Board {
		game.Board[i] = RPSCard{Type: Rock, Owner: NoPlayer}
	}

	// Set up the board
	game.Board[0] = RPSCard{Type: Scissors, Owner: Player2} // Position 0,0: s
	game.Board[1] = RPSCard{Type: Paper, Owner: Player1}    // Position 0,1: P
	game.Board[2] = RPSCard{Type: Rock, Owner: Player1}     // Position 0,2: R
	game.Board[3] = RPSCard{Type: Scissors, Owner: Player2} // Position 1,0: s
	game.Board[4] = RPSCard{Type: Rock, Owner: Player1}     // Position 1,1: R
	game.Board[5] = RPSCard{Type: Rock, Owner: Player1}     // Position 1,2: R
	game.Board[6] = RPSCard{Type: Paper, Owner: Player2}    // Position 2,0: p
	game.Board[7] = RPSCard{Type: Rock, Owner: Player2}     // Position 2,1: r
	game.Board[8] = RPSCard{Type: Paper, Owner: Player1}    // Position 2,2: P

	// Empty the hands to simulate end of game
	game.Player1Hand = []RPSCard{}
	game.Player2Hand = []RPSCard{}

	// Manually set Round to be greater than MaxRounds to ensure game is over
	game.Round = game.MaxRounds + 1

	// Get the string representation
	boardStr := game.String()

	// Print the board for debugging
	t.Logf("Board string representation:\n%s", boardStr)

	// Verify that the board is displayed correctly
	// Parse the board representation (first 3 lines after the header)
	lines := strings.Split(boardStr, "\n")

	// Check the board header
	if !strings.Contains(lines[0], "0 1 2") {
		t.Errorf("Expected board header to contain column numbers, got: %s", lines[0])
	}

	// Check row 0
	row0 := lines[1]
	if !strings.Contains(row0, "s P R") {
		t.Errorf("Row 0 does not match expected output. Got: %s", row0)
	}

	// Check row 1
	row1 := lines[2]
	if !strings.Contains(row1, "s R R") {
		t.Errorf("Row 1 does not match expected output. Got: %s", row1)
	}

	// Check row 2
	row2 := lines[3]
	if !strings.Contains(row2, "p r P") {
		t.Errorf("Row 2 does not match expected output. Got: %s", row2)
	}

	// Count uppercase and lowercase letters in the board representation
	uppercase := 0
	lowercase := 0

	// Only check the actual board part (first 3 rows after header)
	for i := 1; i <= 3; i++ {
		for _, ch := range lines[i] {
			if ch >= 'A' && ch <= 'Z' {
				uppercase++
			} else if ch >= 'a' && ch <= 'z' {
				lowercase++
			}
		}
	}

	t.Logf("Uppercase (Player1) count: %d, Lowercase (Player2) count: %d", uppercase, lowercase)

	// Verify the counts match our expectation
	if uppercase != 5 {
		t.Errorf("Expected 5 uppercase letters (Player1 cards), but found %d", uppercase)
	}

	if lowercase != 4 {
		t.Errorf("Expected 4 lowercase letters (Player2 cards), but found %d", lowercase)
	}

	// Verify game winner information is displayed correctly
	if !strings.Contains(boardStr, "Game over: Player 1 wins") {
		t.Errorf("Expected 'Game over: Player 1 wins' in output, but found: %s",
			strings.Join(lines[len(lines)-1:], " "))
	}
}
