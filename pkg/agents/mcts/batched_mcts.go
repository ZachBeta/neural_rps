package mcts

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/zachbeta/neural_rps/pkg/common"
	"github.com/zachbeta/neural_rps/pkg/game"
)

// BatchedMCTS implements Monte Carlo Tree Search with batched neural network evaluations
type BatchedMCTS struct {
	policyNetwork common.BatchedNeuralNetwork
	valueNetwork  common.BatchedNeuralNetwork
	root          *MCTSNode
	params        MCTSParams

	// Batching-related fields
	batchSize  int
	positions  [][]float64
	nodesIndex map[int]*MCTSNode // Maps batch index to node
	mu         sync.Mutex
}

// NewBatchedMCTS creates a new MCTS instance with batched evaluation
func NewBatchedMCTS(policyNet, valueNet common.BatchedNeuralNetwork, params MCTSParams, batchSize int) *BatchedMCTS {
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
func (mcts *BatchedMCTS) SetRootState(state GameState) {
	mcts.root = &MCTSNode{
		State:      state.Clone(),
		Parent:     nil,
		Children:   make([]*MCTSNode, 0),
		Visits:     0,
		TotalValue: 0,
		Prior:      0.0,
	}
}

// GetMove implements the Agent interface
func (mcts *BatchedMCTS) GetMove(gameState interface{}) (interface{}, error) {
	state, ok := gameState.(*game.RPSCardGame)
	if !ok {
		return nil, fmt.Errorf("invalid game state type, expected *game.RPSCardGame")
	}

	mcts.SetRootState(NewRPSGameStateAdapter(state))
	return mcts.Search(), nil
}

// Name returns the name of the agent
func (mcts *BatchedMCTS) Name() string {
	return "BatchedMCTSAgent"
}

// SetSearchDepth sets the number of simulations to perform
func (mcts *BatchedMCTS) SetSearchDepth(depth int) {
	mcts.params.NumSimulations = depth
}

// SetExplorationConstant sets the exploration constant for UCB
func (mcts *BatchedMCTS) SetExplorationConstant(c float64) {
	mcts.params.ExplorationConst = c
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
		for k := range mcts.nodesIndex {
			delete(mcts.nodesIndex, k)
		}

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
		if node.State.IsGameOver() {
			winner := node.State.GetWinner()
			var value float64
			if winner == game.NoPlayer {
				value = 0.0 // Draw
			} else if winner == node.State.GetCurrentPlayer() {
				value = 1.0 // Win
			} else {
				value = -1.0 // Loss
			}

			mcts.backpropagate(node, value)
			continue
		}

		// If node needs evaluation, add to batch
		if node.Visits == 0 {
			rpsAdapter, ok := node.State.(*RPSGameStateAdapter)
			if !ok {
				panic("BatchedMCTS expects node.State to be adaptable to RPSGameStateAdapter for extractFeatures")
			}
			features := mcts.extractFeatures(rpsAdapter.RPSCardGame)
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

		// Value from value network (assuming first output is value)
		value := values[i][0]

		// Generate children based on policy
		mcts.expandNode(node, policies[i])

		// Backpropagate value
		mcts.backpropagate(node, value)
	}
}

// selectNode traverses the tree to find a node to evaluate
func (mcts *BatchedMCTS) selectNode(node *MCTSNode) *MCTSNode {
	for !node.State.IsGameOver() && mcts.isFullyExpanded(node) {
		node = mcts.selectBestChild(node)
	}

	// If node is terminal, return it
	if node.State.IsGameOver() {
		return node
	}

	// If node is not fully expanded, expand it
	if !mcts.isFullyExpanded(node) && node.Visits > 0 {
		return mcts.expandNode(node, nil)
	}

	return node
}

// isFullyExpanded checks if all possible actions from this node have been explored
func (mcts *BatchedMCTS) isFullyExpanded(node *MCTSNode) bool {
	if node.Visits == 0 {
		return false
	}

	// Get legal moves
	legalMoves := node.State.GetLegalMoves()

	// If we have fewer children than legal moves, the node is not fully expanded
	if len(node.Children) < len(legalMoves) {
		return false
	}

	return true
}

// selectBestChild selects the best child according to UCB formula
func (mcts *BatchedMCTS) selectBestChild(node *MCTSNode) *MCTSNode {
	bestScore := -math.MaxFloat64
	var bestChild *MCTSNode

	for _, child := range node.Children {
		// PUCT formula (used in AlphaZero)
		exploitation := child.TotalValue / float64(child.Visits)
		exploration := mcts.params.ExplorationConst *
			float64(child.Prior) *
			math.Sqrt(float64(node.Visits)) /
			float64(1+child.Visits)

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
	legalMoves := node.State.GetLegalMoves()

	// If no legal moves, return this node
	if len(legalMoves) == 0 {
		return node
	}

	// Find a move that hasn't been tried yet
	for moveIdx, move := range legalMoves {
		// Check if this move already has a child
		alreadyExpanded := false
		for _, child := range node.Children {
			if movesEqual(child.Move, move) {
				alreadyExpanded = true
				break
			}
		}

		if !alreadyExpanded {
			// Create new child for this move
			childState := node.State.Clone()
			childState.ApplyMove(move)

			// Calculate prior probability if policy is provided
			priorProb := 1.0 / float64(len(legalMoves))
			if policy != nil {
				if moveIdx >= 0 && moveIdx < len(policy) {
					priorProb = policy[moveIdx]
				}
			}

			// Create child node
			child := &MCTSNode{
				State:      childState,
				Parent:     node,
				Children:   make([]*MCTSNode, 0),
				Visits:     0,
				TotalValue: 0,
				Prior:      float32(priorProb),
				Move:       move,
			}

			// Add child to parent
			node.Children = append(node.Children, child)
			return child
		}
	}

	if len(node.Children) > 0 {
		return node.Children[0]
	}
	return node
}

// backpropagate updates the statistics of nodes on the path from node to root
func (mcts *BatchedMCTS) backpropagate(node *MCTSNode, value float64) {
	current := node
	for current != nil {
		current.Visits++
		current.TotalValue += value
		value = -value // Flip value for opponent
		current = current.Parent
	}
}

// selectBestMove returns the best move based on visit counts
func (mcts *BatchedMCTS) selectBestMove() game.RPSCardMove {
	bestVisits := -1
	var bestMove game.RPSCardMove

	if mcts.root == nil || len(mcts.root.Children) == 0 {
		if mcts.root != nil && mcts.root.State != nil {
			legalMoves := mcts.root.State.GetLegalMoves()
			if len(legalMoves) > 0 {
				return legalMoves[rand.Intn(len(legalMoves))]
			}
		}
		return game.RPSCardMove{}
	}

	for _, child := range mcts.root.Children {
		if child.Visits > bestVisits {
			bestVisits = child.Visits
			bestMove = child.Move
		}
	}

	return bestMove
}

// extractFeatures extracts features for the neural network from the game state
func (mcts *BatchedMCTS) extractFeatures(state *game.RPSCardGame) []float64 {
	// Use the existing method from the game for feature extraction
	return state.GetBoardAsFeatures()
}

// getMoveIndex converts a move to an index in the policy output
func (mcts *BatchedMCTS) getMoveIndex(move game.RPSCardMove) int {
	// Policy output format: [pos0_rock, pos0_paper, pos0_scissors, pos1_rock, ...]
	// 9 positions * 3 card types = 27 possible actions
	return move.Position*3 + int(0) // Replace 0 with the card type when available
}

// movesEqual checks if two moves are equal
func movesEqual(move1, move2 game.RPSCardMove) bool {
	return move1.Position == move2.Position &&
		move1.CardIndex == move2.CardIndex &&
		move1.Player == move2.Player
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Ensure BatchedMCTS implements the TreeSearchAgent interface
var _ common.TreeSearchAgent = (*BatchedMCTS)(nil)
