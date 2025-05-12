package mcts

import (
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

func TestDefaultRPSMCTSParams(t *testing.T) {
	params := DefaultRPSMCTSParams()

	// Check that default parameters are reasonable
	if params.NumSimulations <= 0 {
		t.Errorf("Expected positive number of simulations, got %d", params.NumSimulations)
	}

	if params.ExplorationConst <= 0 {
		t.Errorf("Expected positive exploration constant, got %f", params.ExplorationConst)
	}
}

func TestNewRPSMCTS(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS with default parameters
	params := DefaultRPSMCTSParams()
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// Check that MCTS was created correctly
	if mctsEngine.PolicyNetwork != policyNetwork {
		t.Errorf("Expected PolicyNetwork to be the same as input")
	}

	if mctsEngine.ValueNetwork != valueNetwork {
		t.Errorf("Expected ValueNetwork to be the same as input")
	}

	if mctsEngine.Params != params {
		t.Errorf("Expected Params to be the same as input")
	}

	if mctsEngine.Root != nil {
		t.Errorf("Expected Root to be nil initially")
	}
}

func TestRPSMCTSSetRootState(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS
	params := DefaultRPSMCTSParams()
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Set the root state
	mctsEngine.SetRootState(gameState)

	// Check that root was created correctly
	if mctsEngine.Root == nil {
		t.Errorf("Expected Root to be non-nil after SetRootState")
		return
	}

	if mctsEngine.Root.GameState == nil {
		t.Errorf("Expected Root.GameState to be non-nil")
		return
	}

	// The root node should have a deep copy of the game state
	if mctsEngine.Root.GameState == gameState {
		t.Errorf("Expected Root.GameState to be a copy, not the same reference")
	}

	// The root node should have no parent
	if mctsEngine.Root.Parent != nil {
		t.Errorf("Expected Root.Parent to be nil")
	}

	// The root node should have no move
	if mctsEngine.Root.Move != nil {
		t.Errorf("Expected Root.Move to be nil")
	}

	// The root node should have policy priors
	if mctsEngine.Root.Priors == nil {
		t.Errorf("Expected Root.Priors to be non-nil")
	}

	if len(mctsEngine.Root.Priors) != 9 {
		t.Errorf("Expected Root.Priors to have length 9, got %d", len(mctsEngine.Root.Priors))
	}
}

func TestRPSMCTSSearch(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS with a small number of simulations for testing
	params := DefaultRPSMCTSParams()
	params.NumSimulations = 10 // Set small number for testing
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Set the root state
	mctsEngine.SetRootState(gameState)

	// Run the search
	bestNode := mctsEngine.Search()

	// Check that search returned a valid node
	if bestNode == nil {
		t.Errorf("Expected Search to return a non-nil node")
		return
	}

	// The best node should have a move
	if bestNode.Move == nil {
		t.Errorf("Expected best node to have a move")
		return
	}

	// The move should be valid
	if bestNode.Move.Player != gameState.CurrentPlayer {
		t.Errorf("Expected move to be for the current player %v, got %v",
			gameState.CurrentPlayer, bestNode.Move.Player)
	}

	if bestNode.Move.Position < 0 || bestNode.Move.Position >= 9 {
		t.Errorf("Expected move position to be in range [0, 8], got %d",
			bestNode.Move.Position)
	}

	// The node should have been visited at least once
	if bestNode.Visits == 0 {
		t.Errorf("Expected best node to have been visited at least once")
	}

	// After search, the root should have children
	if len(mctsEngine.Root.Children) == 0 {
		t.Errorf("Expected root to have children after search")
	}

	// The root should have been visited at least numSimulations times
	if mctsEngine.Root.Visits < params.NumSimulations {
		t.Errorf("Expected root to have at least %d visits, got %d",
			params.NumSimulations, mctsEngine.Root.Visits)
	}
}

func TestRPSMCTSSelection(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS
	params := DefaultRPSMCTSParams()
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Create a node with no children (should be returned by selection)
	node := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Test selection on a leaf node
	selected := mctsEngine.selection(node)
	if selected != node {
		t.Errorf("Expected selection on a leaf node to return the node itself")
	}

	// Create a root with children, but keep one unvisited
	priors := make([]float64, 9)
	for i := range priors {
		priors[i] = 1.0 / 9.0
	}

	root := NewRPSMCTSNode(gameState, nil, nil, priors)
	root.Visits = 10

	// Create children
	for i := 0; i < 3; i++ {
		move := game.RPSMove{CardIndex: 0, Position: i, Player: game.Player1}
		childState := gameState.Copy()
		childState.MakeMove(move)

		childNode := NewRPSMCTSNode(childState, &move, root, nil)

		// Make all but one node visited
		if i < 2 {
			childNode.Visits = 5
		}

		root.Children = append(root.Children, childNode)
	}

	// Test selection on a non-leaf node with an unvisited child
	selected = mctsEngine.selection(root)
	if selected.Visits != 0 {
		t.Errorf("Expected selection to return the unvisited node, got node with %d visits",
			selected.Visits)
	}
}

func TestRPSMCTSEvaluate(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS
	params := DefaultRPSMCTSParams()
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// Test evaluation on non-terminal state (should use the value network)
	gameState := game.NewRPSGame(15, 5, 10)
	node := NewRPSMCTSNode(gameState, nil, nil, nil)

	value := mctsEngine.evaluate(node)

	// Value should be in range [0, 1]
	if value < 0.0 || value > 1.0 {
		t.Errorf("Expected value to be in range [0, 1], got %f", value)
	}

	// Create a game ending in a draw (all positions filled)
	drawGame := game.NewRPSGame(0, 0, 0) // No cards in hand and no more rounds
	drawNode := NewRPSMCTSNode(drawGame, nil, nil, nil)

	drawValue := mctsEngine.evaluate(drawNode)
	if drawValue != 0.5 {
		t.Errorf("Expected value for draw to be 0.5, got %f", drawValue)
	}

	// Create a game where Player1 wins
	player1WinGame := game.NewRPSGame(15, 0, 0) // Game is over
	// Set up a winning position for Player1
	player1WinGame.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}
	player1WinGame.Board[1] = game.RPSCard{Type: game.Paper, Owner: game.Player1}
	player1WinGame.Board[2] = game.RPSCard{Type: game.Scissors, Owner: game.Player1}

	// PlayerX wins
	player1WinNode := NewRPSMCTSNode(player1WinGame, nil, nil, nil)
	player1WinValue := mctsEngine.evaluate(player1WinNode)

	// If current player is Player1, value should be 1.0
	player1WinGame.CurrentPlayer = game.Player1
	player1WinNode = NewRPSMCTSNode(player1WinGame, nil, nil, nil)
	player1WinValue = mctsEngine.evaluate(player1WinNode)
	if player1WinValue != 1.0 {
		t.Errorf("Expected value for Player1 win from Player1's perspective to be 1.0, got %f",
			player1WinValue)
	}

	// If current player is Player2, value should be 0.0
	player1WinGame.CurrentPlayer = game.Player2
	player1WinNode = NewRPSMCTSNode(player1WinGame, nil, nil, nil)
	player1WinValue = mctsEngine.evaluate(player1WinNode)
	if player1WinValue != 0.0 {
		t.Errorf("Expected value for Player1 win from Player2's perspective to be 0.0, got %f",
			player1WinValue)
	}
}

func TestRPSMCTSGetBestMove(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create MCTS with a small number of simulations for testing
	params := DefaultRPSMCTSParams()
	params.NumSimulations = 10 // Set small number for testing
	mctsEngine := NewRPSMCTS(policyNetwork, valueNetwork, params)

	// With no root set, should return nil
	bestMove := mctsEngine.GetBestMove()
	if bestMove != nil {
		t.Errorf("Expected GetBestMove with no root to return nil")
	}

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Set the root state
	mctsEngine.SetRootState(gameState)

	// Get the best move
	bestMove = mctsEngine.GetBestMove()

	// Check that a move was returned
	if bestMove == nil {
		t.Errorf("Expected GetBestMove to return a non-nil move")
		return
	}

	// The move should be valid
	if bestMove.Player != gameState.CurrentPlayer {
		t.Errorf("Expected move to be for the current player %v, got %v",
			gameState.CurrentPlayer, bestMove.Player)
	}

	if bestMove.Position < 0 || bestMove.Position >= 9 {
		t.Errorf("Expected move position to be in range [0, 8], got %d",
			bestMove.Position)
	}

	if bestMove.CardIndex < 0 || bestMove.CardIndex >= len(gameState.Player1Hand) {
		t.Errorf("Expected card index to be in range [0, %d], got %d",
			len(gameState.Player1Hand)-1, bestMove.CardIndex)
	}
}
