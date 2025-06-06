package mcts

import (
	"math"
	"sync/atomic"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// RPSMCTSNode represents a node in the MCTS tree for RPS
type RPSMCTSNode struct {
	GameState  *game.RPSGame
	Move       *game.RPSMove
	Parent     *RPSMCTSNode
	Children   []*RPSMCTSNode
	Visits     atomic.Int64
	TotalValue float64
	Priors     []float64 // Policy priors from neural network
}

// NewRPSMCTSNode creates a new MCTS node
func NewRPSMCTSNode(state *game.RPSGame, move *game.RPSMove, parent *RPSMCTSNode, priors []float64) *RPSMCTSNode {
	n := &RPSMCTSNode{
		GameState:  state,
		Move:       move,
		Parent:     parent,
		Children:   make([]*RPSMCTSNode, 0),
		TotalValue: 0,
		Priors:     priors,
	}
	return n
}

// UCB calculates the Upper Confidence Bound value for this node
// Used for node selection during MCTS
func (n *RPSMCTSNode) UCB(explorationConstant float64) float64 {
	visits := n.Visits.Load()
	if visits == 0 {
		return math.Inf(1) // Infinity for unvisited nodes
	}

	// Get prior based on move position if available
	prior := 1.0 / float64(9) // Uniform default
	if n.Move != nil && n.Parent != nil && n.Parent.Priors != nil {
		// Position-based prior (simplified for RPS card game)
		prior = n.Parent.Priors[n.Move.Position]
	}

	// UCB formula with prior: Q + U
	// Q = average value
	// U = exploration bonus with prior
	exploitation := n.TotalValue / float64(visits)
	parentVisits := int64(0)
	if n.Parent != nil {
		parentVisits = n.Parent.Visits.Load()
	}
	exploration := explorationConstant * prior * math.Sqrt(float64(parentVisits)) / (1.0 + float64(visits))

	return exploitation + exploration
}

// SelectChild selects the child with the highest UCB value
func (n *RPSMCTSNode) SelectChild(explorationConstant float64) *RPSMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}

	bestChild := n.Children[0]
	bestUCB := bestChild.UCB(explorationConstant)

	for _, child := range n.Children[1:] {
		ucb := child.UCB(explorationConstant)
		if ucb > bestUCB {
			bestChild = child
			bestUCB = ucb
		}
	}

	return bestChild
}

// ExpandAll expands all possible child nodes
func (n *RPSMCTSNode) ExpandAll(priors []float64) {
	// Clear any existing children
	n.Children = make([]*RPSMCTSNode, 0)

	// Get valid moves
	validMoves := n.GameState.GetValidMoves()

	// Create children for each valid move
	for _, move := range validMoves {
		// Create a copy of the game state
		childState := n.GameState.Copy()

		// Apply the move
		moveCopy := move // Copy to avoid issues with references inside the loop
		err := childState.MakeMove(moveCopy)
		if err != nil {
			continue // Skip invalid moves
		}

		// Create and add the child node
		child := NewRPSMCTSNode(childState, &moveCopy, n, priors)
		n.Children = append(n.Children, child)
	}
}

// Update updates the node statistics based on simulation results
func (n *RPSMCTSNode) Update(value float64) {
	n.Visits.Add(1)
	n.TotalValue += value
}

// UpdateRecursive updates this node and all its ancestors
func (n *RPSMCTSNode) UpdateRecursive(value float64) {
	// Update this node
	n.Update(value)

	// Update parent recursively
	if n.Parent != nil {
		// Flip value perspective for parent (from opponent's point of view)
		n.Parent.UpdateRecursive(1.0 - value)
	}
}

// MostVisitedChild returns the child with the most visits
func (n *RPSMCTSNode) MostVisitedChild() *RPSMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}

	bestChild := n.Children[0]
	mostVisits := bestChild.Visits.Load()

	for _, child := range n.Children[1:] {
		currentVisits := child.Visits.Load()
		if currentVisits > mostVisits {
			bestChild = child
			mostVisits = currentVisits
		}
	}

	return bestChild
}

// BestChild returns the child with the highest average value
func (n *RPSMCTSNode) BestChild() *RPSMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}

	bestChild := n.Children[0]
	bestChildVisits := bestChild.Visits.Load()
	bestValue := 0.0
	if bestChildVisits > 0 {
		bestValue = bestChild.TotalValue / float64(bestChildVisits)
	} else {
		// Handle case where the first child might have 0 visits if it's the only one
		// For safety, though UCB should prevent selecting unvisited if others exist
		bestValue = math.Inf(-1) // Or some other appropriate default for unvisited
	}

	for _, child := range n.Children[1:] {
		childVisits := child.Visits.Load()
		if childVisits == 0 {
			continue
		}

		value := child.TotalValue / float64(childVisits)
		if value > bestValue {
			bestChild = child
			bestValue = value
		}
	}

	return bestChild
}
