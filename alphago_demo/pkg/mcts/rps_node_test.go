package mcts

import (
	"math"
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func TestNewRPSMCTSNode(t *testing.T) {
	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Create a move
	move := game.RPSMove{
		CardIndex: 0,
		Position:  4,
		Player:    game.Player1,
	}

	// Create priors
	priors := make([]float64, 9)
	for i := range priors {
		priors[i] = 1.0 / 9.0
	}

	// Create a node
	node := NewRPSMCTSNode(gameState, &move, nil, priors)

	// Check that the node was created correctly
	if node.GameState != gameState {
		t.Errorf("Expected GameState to be the same as input")
	}

	if node.Move == nil || *node.Move != move {
		t.Errorf("Expected Move to be the same as input")
	}

	if node.Parent != nil {
		t.Errorf("Expected Parent to be nil")
	}

	if len(node.Children) != 0 {
		t.Errorf("Expected Children to be empty, got %d children", len(node.Children))
	}

	if node.Visits != 0 {
		t.Errorf("Expected Visits to be 0, got %d", node.Visits)
	}

	if node.TotalValue != 0.0 {
		t.Errorf("Expected TotalValue to be 0.0, got %f", node.TotalValue)
	}

	if node.Priors == nil || len(node.Priors) != len(priors) {
		t.Errorf("Expected Priors to have length %d, got %d", len(priors), len(node.Priors))
	}

	for i, p := range node.Priors {
		if p != priors[i] {
			t.Errorf("Expected Priors[%d] to be %f, got %f", i, priors[i], p)
		}
	}
}

func TestRPSMCTSNodeUCB(t *testing.T) {
	// Create a root node
	gameState := game.NewRPSGame(15, 5, 10)
	rootNode := NewRPSMCTSNode(gameState, nil, nil, nil)
	rootNode.Visits = 10

	// Create a child node
	move := game.RPSMove{CardIndex: 0, Position: 4, Player: game.Player1}
	childState := gameState.Copy()
	childState.MakeMove(move)

	// Create priors for the root
	priors := make([]float64, 9)
	for i := range priors {
		priors[i] = 0.1 // Uniform prior
	}
	priors[4] = 0.2 // Higher prior for position 4

	rootNode.Priors = priors

	// Create a child with the move to position 4
	childNode := NewRPSMCTSNode(childState, &move, rootNode, nil)
	childNode.Visits = 5
	childNode.TotalValue = 3.0 // 60% win rate

	// Add as child to root
	rootNode.Children = append(rootNode.Children, childNode)

	// Test UCB calculation
	explorationConstant := 1.0
	ucbValue := childNode.UCB(explorationConstant)

	// Manually calculate expected UCB
	// UCB = Q + U
	// Q = 3.0 / 5 = 0.6
	// U = c * P * sqrt(N_parent) / (1 + N_child)
	// U = 1.0 * 0.2 * sqrt(10) / (1 + 5) = 0.2 * sqrt(10) / 6
	expectedQ := 0.6
	expectedU := 0.2 * math.Sqrt(10) / 6.0
	expectedUCB := expectedQ + expectedU

	// Allow for small floating point differences
	if math.Abs(ucbValue-expectedUCB) > 0.0001 {
		t.Errorf("Expected UCB to be %f, got %f", expectedUCB, ucbValue)
	}

	// Test with an unvisited node (should return infinity)
	unvisitedNode := NewRPSMCTSNode(childState, &move, rootNode, nil)
	ucbValue = unvisitedNode.UCB(explorationConstant)

	if !math.IsInf(ucbValue, 1) {
		t.Errorf("Expected UCB for unvisited node to be +Inf, got %f", ucbValue)
	}
}

func TestRPSMCTSNodeSelectChild(t *testing.T) {
	// Create a root node
	gameState := game.NewRPSGame(15, 5, 10)
	rootNode := NewRPSMCTSNode(gameState, nil, nil, nil)
	rootNode.Visits = 30

	// Create uniform priors
	priors := make([]float64, 9)
	for i := range priors {
		priors[i] = 1.0 / 9.0
	}
	rootNode.Priors = priors

	// Create three children with different statistics
	for i := 0; i < 3; i++ {
		move := game.RPSMove{CardIndex: 0, Position: i, Player: game.Player1}
		childState := gameState.Copy()
		childState.MakeMove(move)

		childNode := NewRPSMCTSNode(childState, &move, rootNode, nil)

		// Set different values for each child
		switch i {
		case 0:
			// Low value, high visits
			childNode.Visits = 15
			childNode.TotalValue = 5.0 // 33% win rate
		case 1:
			// High value, medium visits
			childNode.Visits = 10
			childNode.TotalValue = 8.0 // 80% win rate
		case 2:
			// Medium value, low visits
			childNode.Visits = 5
			childNode.TotalValue = 3.0 // 60% win rate
		}

		rootNode.Children = append(rootNode.Children, childNode)
	}

	// Test selection with different exploration constants
	// Low exploration, should select high value node (case 1)
	bestChild := rootNode.SelectChild(0.1)
	if bestChild.Move.Position != 1 {
		t.Errorf("With low exploration, expected to select child with position 1 (high value), got %d",
			bestChild.Move.Position)
	}

	// High exploration, should select low visits node (case 2)
	bestChild = rootNode.SelectChild(10.0)
	if bestChild.Move.Position != 2 {
		t.Errorf("With high exploration, expected to select child with position 2 (low visits), got %d",
			bestChild.Move.Position)
	}
}

func TestRPSMCTSNodeExpandAll(t *testing.T) {
	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Create a node
	node := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Create priors
	priors := make([]float64, 9)
	for i := range priors {
		priors[i] = 1.0 / 9.0
	}

	// Expand all children
	node.ExpandAll(priors)

	// Check that children were created correctly
	validMoves := gameState.GetValidMoves()
	if len(node.Children) != len(validMoves) {
		t.Errorf("Expected %d children, got %d", len(validMoves), len(node.Children))
	}

	// Verify children have correct properties
	for _, child := range node.Children {
		// Each child should point to the parent
		if child.Parent != node {
			t.Errorf("Child does not point to the correct parent")
		}

		// Each child should have a valid move
		if child.Move == nil {
			t.Errorf("Child has nil move")
		} else {
			// Move should be for the current player
			if child.Move.Player != gameState.CurrentPlayer {
				t.Errorf("Child move has wrong player: %v vs %v",
					child.Move.Player, gameState.CurrentPlayer)
			}

			// Position should be between 0-8
			if child.Move.Position < 0 || child.Move.Position >= 9 {
				t.Errorf("Child move has invalid position: %d", child.Move.Position)
			}

			// Card index should be valid
			if gameState.CurrentPlayer == game.Player1 {
				if child.Move.CardIndex < 0 || child.Move.CardIndex >= len(gameState.Player1Hand) {
					t.Errorf("Child move has invalid card index: %d", child.Move.CardIndex)
				}
			} else {
				if child.Move.CardIndex < 0 || child.Move.CardIndex >= len(gameState.Player2Hand) {
					t.Errorf("Child move has invalid card index: %d", child.Move.CardIndex)
				}
			}
		}

		// Each child should have 0 visits
		if child.Visits != 0 {
			t.Errorf("Expected child to have 0 visits, got %d", child.Visits)
		}

		// Each child should have the priors we provided
		if child.Priors == nil || len(child.Priors) != len(priors) {
			t.Errorf("Child has wrong priors length: %d vs %d",
				len(child.Priors), len(priors))
		}
	}
}

func TestRPSMCTSNodeUpdate(t *testing.T) {
	// Create a node
	gameState := game.NewRPSGame(15, 5, 10)
	node := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Initial state
	if node.Visits != 0 {
		t.Errorf("Expected initial visits to be 0, got %d", node.Visits)
	}
	if node.TotalValue != 0.0 {
		t.Errorf("Expected initial total value to be 0.0, got %f", node.TotalValue)
	}

	// Update once
	node.Update(0.5)
	if node.Visits != 1 {
		t.Errorf("After one update, expected visits to be 1, got %d", node.Visits)
	}
	if node.TotalValue != 0.5 {
		t.Errorf("After one update, expected total value to be 0.5, got %f", node.TotalValue)
	}

	// Update again
	node.Update(0.8)
	if node.Visits != 2 {
		t.Errorf("After two updates, expected visits to be 2, got %d", node.Visits)
	}
	if node.TotalValue != 1.3 {
		t.Errorf("After two updates, expected total value to be 1.3, got %f", node.TotalValue)
	}
}

func TestRPSMCTSNodeUpdateRecursive(t *testing.T) {
	// Create a tree: root -> child1 -> child2
	gameState := game.NewRPSGame(15, 5, 10)
	root := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Create child1
	move1 := game.RPSMove{CardIndex: 0, Position: 0, Player: game.Player1}
	childState1 := gameState.Copy()
	childState1.MakeMove(move1)
	child1 := NewRPSMCTSNode(childState1, &move1, root, nil)

	// Create child2
	move2 := game.RPSMove{CardIndex: 0, Position: 1, Player: game.Player2}
	childState2 := childState1.Copy()
	childState2.MakeMove(move2)
	child2 := NewRPSMCTSNode(childState2, &move2, child1, nil)

	// Link nodes
	root.Children = []*RPSMCTSNode{child1}
	child1.Children = []*RPSMCTSNode{child2}

	// Update from child2
	child2.UpdateRecursive(1.0)

	// Check that child2 was updated
	if child2.Visits != 1 {
		t.Errorf("Expected child2 visits to be 1, got %d", child2.Visits)
	}
	if child2.TotalValue != 1.0 {
		t.Errorf("Expected child2 total value to be 1.0, got %f", child2.TotalValue)
	}

	// Check that child1 was updated with the opposite value
	if child1.Visits != 1 {
		t.Errorf("Expected child1 visits to be 1, got %d", child1.Visits)
	}
	if child1.TotalValue != 0.0 { // 1.0 - 1.0 = 0.0
		t.Errorf("Expected child1 total value to be 0.0, got %f", child1.TotalValue)
	}

	// Check that root was updated with the original value
	if root.Visits != 1 {
		t.Errorf("Expected root visits to be 1, got %d", root.Visits)
	}
	if root.TotalValue != 1.0 { // 1.0 - 0.0 = 1.0
		t.Errorf("Expected root total value to be 1.0, got %f", root.TotalValue)
	}
}

func TestRPSMCTSNodeMostVisitedChild(t *testing.T) {
	// Create a root node
	gameState := game.NewRPSGame(15, 5, 10)
	root := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Create three children with different visit counts
	for i := 0; i < 3; i++ {
		move := game.RPSMove{CardIndex: 0, Position: i, Player: game.Player1}
		childState := gameState.Copy()
		childState.MakeMove(move)

		childNode := NewRPSMCTSNode(childState, &move, root, nil)

		// Set different visits for each child
		childNode.Visits = i * 5

		root.Children = append(root.Children, childNode)
	}

	// Test most visited child (should be the last one with 10 visits)
	bestChild := root.MostVisitedChild()
	if bestChild.Move.Position != 2 {
		t.Errorf("Expected most visited child to have position 2, got %d",
			bestChild.Move.Position)
	}
	if bestChild.Visits != 10 {
		t.Errorf("Expected most visited child to have 10 visits, got %d",
			bestChild.Visits)
	}
}

func TestRPSMCTSNodeBestChild(t *testing.T) {
	// Create a root node
	gameState := game.NewRPSGame(15, 5, 10)
	root := NewRPSMCTSNode(gameState, nil, nil, nil)

	// Create three children with different value/visit ratios
	for i := 0; i < 3; i++ {
		move := game.RPSMove{CardIndex: 0, Position: i, Player: game.Player1}
		childState := gameState.Copy()
		childState.MakeMove(move)

		childNode := NewRPSMCTSNode(childState, &move, root, nil)

		// Set different values and visits for each child
		childNode.Visits = 10

		switch i {
		case 0:
			childNode.TotalValue = 5.0 // 50% win rate
		case 1:
			childNode.TotalValue = 8.0 // 80% win rate
		case 2:
			childNode.TotalValue = 6.0 // 60% win rate
		}

		root.Children = append(root.Children, childNode)
	}

	// Test best child (should be the one with position 1 and 80% win rate)
	bestChild := root.BestChild()
	if bestChild.Move.Position != 1 {
		t.Errorf("Expected best child to have position 1, got %d",
			bestChild.Move.Position)
	}

	value := bestChild.TotalValue / float64(bestChild.Visits)
	if value != 0.8 {
		t.Errorf("Expected best child to have value 0.8, got %f", value)
	}
}
