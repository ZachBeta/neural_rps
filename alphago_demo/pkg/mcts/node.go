package mcts

import (
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// AGMCTSNode represents a node in the Monte Carlo Tree Search
type AGMCTSNode struct {
	// Game state at this node
	GameState *game.AGGame

	// Move that led to this state (nil for root)
	Move *game.AGMove

	// Node statistics
	Visits     int
	TotalValue float64

	// Tree structure
	Parent   *AGMCTSNode
	Children []*AGMCTSNode

	// Policy prior probabilities from policy network
	Priors []float64

	// Whether the node is fully expanded
	fullyExpanded bool
}

// NewAGMCTSNode creates a new MCTS node
func NewAGMCTSNode(gameState *game.AGGame, parent *AGMCTSNode, move *game.AGMove, priors []float64) *AGMCTSNode {
	return &AGMCTSNode{
		GameState:     gameState.Copy(),
		Move:          move,
		Visits:        0,
		TotalValue:    0,
		Parent:        parent,
		Children:      make([]*AGMCTSNode, 0),
		Priors:        priors,
		fullyExpanded: false,
	}
}

// GetUCB returns the Upper Confidence Bound value for this node
// This balances exploration and exploitation in MCTS
func (n *AGMCTSNode) GetUCB(c float64) float64 {
	// If node has not been visited, return infinity
	if n.Visits == 0 {
		return math.Inf(1)
	}

	// Calculate exploitation term (average value)
	exploitation := n.TotalValue / float64(n.Visits)

	// Calculate exploration term (UCB formula)
	parentVisits := 1.0
	if n.Parent != nil {
		parentVisits = float64(n.Parent.Visits)
	}

	// Find the move index
	priorProb := 1.0 / float64(len(n.Priors))
	if n.Move != nil {
		moveIdx := n.Move.Row*3 + n.Move.Col
		if moveIdx >= 0 && moveIdx < len(n.Priors) {
			priorProb = n.Priors[moveIdx]
		}
	}

	exploration := c * priorProb * math.Sqrt(parentVisits) / (1 + float64(n.Visits))

	return exploitation + exploration
}

// IsLeaf returns true if this node is a leaf node (no children)
func (n *AGMCTSNode) IsLeaf() bool {
	return len(n.Children) == 0
}

// IsTerminal returns true if this node is a terminal state (game over)
func (n *AGMCTSNode) IsTerminal() bool {
	return n.GameState.IsGameOver()
}

// GetValue returns the average value of this node
func (n *AGMCTSNode) GetValue() float64 {
	if n.Visits == 0 {
		return 0
	}
	return n.TotalValue / float64(n.Visits)
}

// Update updates the node statistics
func (n *AGMCTSNode) Update(value float64) {
	n.Visits++
	n.TotalValue += value
}

// GetMostVisitedChild returns the child with the most visits
func (n *AGMCTSNode) GetMostVisitedChild() *AGMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}

	bestChild := n.Children[0]
	bestVisits := bestChild.Visits

	for _, child := range n.Children[1:] {
		if child.Visits > bestVisits {
			bestChild = child
			bestVisits = child.Visits
		}
	}

	return bestChild
}

// GetBestChild returns the child with the highest value (for final move selection)
func (n *AGMCTSNode) GetBestChild() *AGMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}

	bestChild := n.Children[0]
	bestValue := bestChild.GetValue()

	for _, child := range n.Children[1:] {
		value := child.GetValue()
		if value > bestValue {
			bestChild = child
			bestValue = value
		}
	}

	return bestChild
}

// GetRandomChild returns a random child
func (n *AGMCTSNode) GetRandomChild() *AGMCTSNode {
	if len(n.Children) == 0 {
		return nil
	}
	return n.Children[rand.Intn(len(n.Children))]
}
