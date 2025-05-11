package mcts

import (
	"fmt"
	"math"
	"sync"

	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/neural"
)

// MCTSNode represents a node in the MCTS tree
type MCTSNode struct {
	state     *game.RPSCardGame
	parent    *MCTSNode
	children  []*MCTSNode
	visits    int
	value     float64 // Total value accumulated
	priorProb float64 // Prior probability from policy network
}

// MCTSParams contains parameters for the MCTS algorithm
type MCTSParams struct {
	NumSimulations   int
	ExplorationConst float64
}

// DefaultMCTSParams returns the default MCTS parameters
func DefaultMCTSParams() MCTSParams {
	return MCTSParams{
		NumSimulations:   800,
		ExplorationConst: 1.0,
	}
}

// BatchedMCTS implements Monte Carlo Tree Search with batched neural network evaluations
type BatchedMCTS struct {
	policyNetwork *neural.RPSTFPolicyNetwork
	valueNetwork  *neural.RPSTFPolicyNetwork // Using policy network for both in this example
	root          *MCTSNode
	params        MCTSParams

	// Batching-related fields
	batchSize  int
	positions  [][]float64
	nodesIndex map[int]*MCTSNode // Maps batch index to node
	mu         sync.Mutex
}

// NewBatchedMCTS creates a new MCTS instance with batched evaluation
func NewBatchedMCTS(policyNet, valueNet *neural.RPSTFPolicyNetwork, params MCTSParams, batchSize int) *BatchedMCTS {
	return &BatchedMCTS{
		policyNetwork: policyNet,
		valueNetwork:  valueNet,
		params:        params,
		batchSize:     batchSize,
		positions:     make([][]float64, 0, batchSize),
		nodesIndex:    make(map[int]*MCTSNode),
	}
}

// SetRootState sets the root state for the search
func (mcts *BatchedMCTS) SetRootState(state *game.RPSCardGame) {
	mcts.root = &MCTSNode{
		state:     state.Clone(),
		parent:    nil,
		children:  make([]*MCTSNode, 0),
		visits:    0,
		value:     0,
		priorProb: 1.0,
	}
}

// Search runs the MCTS algorithm and returns the best move
func (mcts *BatchedMCTS) Search() game.RPSCardMove {
	if mcts.root == nil {
		panic("Root state not set")
	}

	// Run simulations in batches
	for i := 0; i < mcts.params.NumSimulations; i += mcts.batchSize {
		// Reset batch containers
		mcts.positions = mcts.positions[:0]
		clear(mcts.nodesIndex)

		// Fill batch with nodes to evaluate
		remaining := min(mcts.batchSize, mcts.params.NumSimulations-i)
		mcts.collectNodesToEvaluate(remaining)

		// If batch is not empty, evaluate all nodes at once
		if len(mcts.positions) > 0 {
			mcts.evaluateAndBackpropagate()
		}
	}

	// Select best child of root based on visits
	return mcts.selectBestMove()
}

// collectNodesToEvaluate selects nodes for the current batch
func (mcts *BatchedMCTS) collectNodesToEvaluate(count int) {
	for len(mcts.positions) < count {
		// Select a node to expand
		node := mcts.selectNode(mcts.root)

		// If node is terminal, backpropagate terminal value and continue
		if node.state.IsGameOver() {
			winner := node.state.GetWinner()
			var value float64
			if winner == game.NoPlayer {
				value = 0.0 // Draw
			} else if winner == node.state.CurrentPlayer {
				value = 1.0 // Win
			} else {
				value = -1.0 // Loss
			}

			mcts.backpropagate(node, value)
			continue
		}

		// If node needs evaluation, add to batch
		if node.visits == 0 {
			features := mcts.extractFeatures(node.state)
			mcts.positions = append(mcts.positions, features)
			mcts.nodesIndex[len(mcts.positions)-1] = node
		}
	}
}

// evaluateAndBackpropagate evaluates all positions in the batch and backpropagates values
func (mcts *BatchedMCTS) evaluateAndBackpropagate() {
	// Evaluate all positions at once
	values, err := mcts.valueNetwork.ForwardBatch(mcts.positions)
	if err != nil {
		panic("Error evaluating positions: " + err.Error())
	}

	policies, err := mcts.policyNetwork.ForwardBatch(mcts.positions)
	if err != nil {
		panic("Error evaluating positions: " + err.Error())
	}

	// Process results and backpropagate
	for i := 0; i < len(mcts.positions); i++ {
		node := mcts.nodesIndex[i]

		// Value from value network (assuming scalar output)
		value := values[i][0]

		// Generate children based on policy
		mcts.expandNode(node, policies[i])

		// Backpropagate value
		mcts.backpropagate(node, value)
	}
}

// selectNode traverses the tree to find a node to evaluate
func (mcts *BatchedMCTS) selectNode(node *MCTSNode) *MCTSNode {
	for !node.state.IsGameOver() && mcts.isFullyExpanded(node) {
		node = mcts.selectBestChild(node)
	}

	// If node is terminal, return it
	if node.state.IsGameOver() {
		return node
	}

	// If node is not fully expanded, expand it
	if !mcts.isFullyExpanded(node) && node.visits > 0 {
		return mcts.expandNode(node, nil)
	}

	return node
}

// isFullyExpanded checks if all possible actions from this node have been explored
func (mcts *BatchedMCTS) isFullyExpanded(node *MCTSNode) bool {
	if node.visits == 0 {
		return false
	}

	// Get legal moves
	legalMoves := node.state.GetLegalMoves()

	// If we have fewer children than legal moves, the node is not fully expanded
	if len(node.children) < len(legalMoves) {
		return false
	}

	return true
}

// selectBestChild selects the best child according to UCB formula
func (mcts *BatchedMCTS) selectBestChild(node *MCTSNode) *MCTSNode {
	bestScore := -math.MaxFloat64
	var bestChild *MCTSNode

	for _, child := range node.children {
		// PUCT formula (used in AlphaZero)
		exploitation := child.value / float64(child.visits)
		exploration := mcts.params.ExplorationConst *
			child.priorProb *
			math.Sqrt(float64(node.visits)) /
			float64(1+child.visits)

		score := exploitation + exploration

		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}

	return bestChild
}

// expandNode expands a node by creating one of its unexplored children
func (mcts *BatchedMCTS) expandNode(node *MCTSNode, policy []float64) *MCTSNode {
	// Get legal moves
	legalMoves := node.state.GetLegalMoves()

	// If no legal moves, return this node
	if len(legalMoves) == 0 {
		return node
	}

	// Find a move that hasn't been tried yet
	for _, move := range legalMoves {
		// Check if this move already has a child
		alreadyExpanded := false
		for _, child := range node.children {
			if movesEqual(child.state.LastMove, move) {
				alreadyExpanded = true
				break
			}
		}

		if !alreadyExpanded {
			// Create new child for this move
			childState := node.state.Clone()
			childState.ApplyMove(move)

			// Calculate prior probability if policy is provided
			priorProb := 1.0 / float64(len(legalMoves))
			if policy != nil {
				moveIdx := mcts.getMoveIndex(move)
				if moveIdx >= 0 && moveIdx < len(policy) {
					priorProb = policy[moveIdx]
				}
			}

			child := &MCTSNode{
				state:     childState,
				parent:    node,
				children:  make([]*MCTSNode, 0),
				visits:    0,
				value:     0,
				priorProb: priorProb,
			}

			node.children = append(node.children, child)
			return child
		}
	}

	// If all children are expanded, just return the node
	return node
}

// backpropagate updates values up the tree
func (mcts *BatchedMCTS) backpropagate(node *MCTSNode, value float64) {
	for node != nil {
		node.visits++
		node.value += value
		value = -value // Flip for opponent
		node = node.parent
	}
}

// selectBestMove returns the best move from the root
func (mcts *BatchedMCTS) selectBestMove() game.RPSCardMove {
	if len(mcts.root.children) == 0 {
		// No children, return a random move
		randomMove, err := mcts.root.state.GetRandomMove()
		if err != nil {
			panic(fmt.Sprintf("Error getting random move: %v", err))
		}
		return randomMove
	}

	// Select based on visit count (most robust policy)
	var bestChild *MCTSNode
	bestVisits := -1

	for _, child := range mcts.root.children {
		if child.visits > bestVisits {
			bestVisits = child.visits
			bestChild = child
		}
	}

	return bestChild.state.LastMove
}

// Helper methods

// extractFeatures converts a game state to neural network input
func (mcts *BatchedMCTS) extractFeatures(state *game.RPSCardGame) []float64 {
	// Implementation depends on specific feature representation
	features := make([]float64, mcts.policyNetwork.InputSize)

	// Fill features based on game state (simplified example)
	idx := 0

	// Board state (9 positions, 3 card types)
	for pos := 0; pos < 9; pos++ {
		cardType := int(state.Board[pos])
		for t := 0; t < 3; t++ {
			if t == cardType {
				features[idx] = 1.0
			} else {
				features[idx] = 0.0
			}
			idx++
		}
	}

	// Board ownership (9 positions, 3 possibilities: no owner, player 1, player 2)
	for pos := 0; pos < 9; pos++ {
		owner := state.BoardOwner[pos]
		for o := game.NoPlayer; o <= game.Player2; o++ {
			if owner == o {
				features[idx] = 1.0
			} else {
				features[idx] = 0.0
			}
			idx++
		}
	}

	// Current player's hand
	playerHand := state.Player1Hand
	if state.CurrentPlayer == game.Player2 {
		playerHand = state.Player2Hand
	}

	for i := 0; i < state.HandSize; i++ {
		if i < len(playerHand) {
			cardType := int(playerHand[i])
			for t := 0; t < 3; t++ {
				if t == cardType {
					features[idx] = 1.0
				} else {
					features[idx] = 0.0
				}
				idx++
			}
		} else {
			// Empty hand slot
			for t := 0; t < 3; t++ {
				features[idx] = 0.0
				idx++
			}
		}
	}

	// Current player indicator
	if state.CurrentPlayer == game.Player1 {
		features[idx] = 1.0
	} else {
		features[idx] = 0.0
	}
	idx++

	// Round number (normalized)
	features[idx] = float64(state.Round) / float64(state.MaxRounds)

	return features
}

// getMoveIndex maps a move to an index in the policy output
func (mcts *BatchedMCTS) getMoveIndex(move game.RPSCardMove) int {
	// Simple mapping: CardIndex * 9 + Position
	// This assumes the policy network outputs probabilities for all card+position combinations
	return move.CardIndex*9 + move.Position
}

// Helper functions
func movesEqual(move1, move2 game.RPSCardMove) bool {
	return move1.CardIndex == move2.CardIndex &&
		move1.Position == move2.Position &&
		move1.Player == move2.Player
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
