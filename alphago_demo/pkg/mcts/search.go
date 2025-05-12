package mcts

import (
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

// AGMCTSParams contains parameters for the MCTS algorithm
type AGMCTSParams struct {
	NumSimulations       int     // Number of simulations to run
	ExplorationConstant  float64 // Exploration constant for UCB
	TemperatureInit      float64 // Initial temperature for move selection
	TemperatureFinal     float64 // Final temperature for move selection
	TemperatureThreshold int     // Move threshold to switch to final temperature
}

// DefaultAGMCTSParams returns default MCTS parameters
func DefaultAGMCTSParams() AGMCTSParams {
	return AGMCTSParams{
		NumSimulations:       800,
		ExplorationConstant:  1.25,
		TemperatureInit:      1.0,
		TemperatureFinal:     0.1,
		TemperatureThreshold: 10,
	}
}

// AGMCTS represents the Monte Carlo Tree Search algorithm
type AGMCTS struct {
	policyNetwork *neural.AGPolicyNetwork
	valueNetwork  *neural.AGValueNetwork
	params        AGMCTSParams
	rootNode      *AGMCTSNode
}

// NewAGMCTS creates a new MCTS instance
func NewAGMCTS(policyNetwork *neural.AGPolicyNetwork, valueNetwork *neural.AGValueNetwork, params AGMCTSParams) *AGMCTS {
	return &AGMCTS{
		policyNetwork: policyNetwork,
		valueNetwork:  valueNetwork,
		params:        params,
	}
}

// SetRootState sets the root state for the search
func (mcts *AGMCTS) SetRootState(gameState *game.AGGame) {
	// Get policy predictions for root state
	priors := mcts.policyNetwork.Predict(gameState)

	// Create root node
	mcts.rootNode = NewAGMCTSNode(gameState, nil, nil, priors)
}

// Search runs the MCTS algorithm to find the best move
func (mcts *AGMCTS) Search() *AGMCTSNode {
	// Run simulations
	for i := 0; i < mcts.params.NumSimulations; i++ {
		// Selection phase: select a node to expand
		node := mcts.selectNode(mcts.rootNode)

		// Expansion phase: if the selected node is not terminal, expand it
		if !node.IsTerminal() {
			node = mcts.expand(node)
		}

		// Simulation/Evaluation phase: evaluate the node using the value network
		value := mcts.evaluate(node)

		// Backpropagation phase: update node statistics up the tree
		mcts.backpropagate(node, value)
	}

	// Return the best child of the root node
	return mcts.getBestChild(mcts.rootNode)
}

// selectNode implements the selection phase of MCTS
func (mcts *AGMCTS) selectNode(node *AGMCTSNode) *AGMCTSNode {
	// Keep traversing down the tree until we find a leaf node
	for !node.IsLeaf() && !node.IsTerminal() {
		// If the node is not fully expanded, expand it
		if !node.fullyExpanded {
			return node
		}

		// Find the child with the highest UCB score
		bestChild := node.Children[0]
		bestUCB := bestChild.GetUCB(mcts.params.ExplorationConstant)

		for _, child := range node.Children[1:] {
			ucb := child.GetUCB(mcts.params.ExplorationConstant)
			if ucb > bestUCB {
				bestChild = child
				bestUCB = ucb
			}
		}

		node = bestChild
	}

	return node
}

// expand implements the expansion phase of MCTS
func (mcts *AGMCTS) expand(node *AGMCTSNode) *AGMCTSNode {
	// Get valid moves
	validMoves := node.GameState.GetValidMoves()

	// If there are no valid moves or the node is terminal, just return it
	if len(validMoves) == 0 || node.IsTerminal() {
		node.fullyExpanded = true
		return node
	}

	// Check if all possible moves have already been expanded
	if len(node.Children) == len(validMoves) {
		node.fullyExpanded = true
		return node.GetRandomChild()
	}

	// Find a move that hasn't been expanded yet
	for _, move := range validMoves {
		// Check if this move has already been expanded
		moveAlreadyExpanded := false
		for _, child := range node.Children {
			if child.Move != nil &&
				child.Move.Row == move.Row &&
				child.Move.Col == move.Col {
				moveAlreadyExpanded = true
				break
			}
		}

		// If the move hasn't been expanded, expand it
		if !moveAlreadyExpanded {
			// Create a new game state by applying the move
			newState := node.GameState.Copy()
			newState.MakeMove(move)

			// Get policy predictions for the new state
			priors := mcts.policyNetwork.Predict(newState)

			// Create a new child node
			childNode := NewAGMCTSNode(newState, node, &move, priors)

			// Add the child to the node's children
			node.Children = append(node.Children, childNode)

			return childNode
		}
	}

	// All moves have been expanded
	node.fullyExpanded = true
	return node.GetRandomChild()
}

// evaluate implements the simulation/evaluation phase of MCTS
func (mcts *AGMCTS) evaluate(node *AGMCTSNode) float64 {
	// If the game is over, return the actual result
	if node.IsTerminal() {
		winner := node.GameState.GetWinner()
		if winner == game.Empty {
			return 0.5 // Draw
		} else if winner == node.GameState.CurrentPlayer {
			return 1.0 // Win for current player
		} else {
			return 0.0 // Loss for current player
		}
	}

	// Otherwise, use the value network to estimate the value
	return mcts.valueNetwork.Predict(node.GameState)
}

// backpropagate implements the backpropagation phase of MCTS
func (mcts *AGMCTS) backpropagate(node *AGMCTSNode, value float64) {
	// Backpropagate value up the tree, flipping the perspective at each level
	for node != nil {
		node.Update(value)
		node = node.Parent
		value = 1.0 - value // Flip perspective for the opponent
	}
}

// getBestChild returns the best child of a node
func (mcts *AGMCTS) getBestChild(node *AGMCTSNode) *AGMCTSNode {
	// During actual play, we usually want the most visited child
	// (which is more robust than the highest value child)
	return node.GetMostVisitedChild()
}

// GetBestMove returns the best move according to MCTS
func (mcts *AGMCTS) GetBestMove() game.AGMove {
	bestChild := mcts.Search()
	if bestChild == nil || bestChild.Move == nil {
		// Fallback to a random move if MCTS fails
		randomMove, err := mcts.rootNode.GameState.GetRandomMove()
		if err != nil {
			// If even that fails, return invalid move
			return game.AGMove{Row: -1, Col: -1}
		}
		return randomMove
	}
	return *bestChild.Move
}

// GetActionProbabilities returns a distribution over possible actions based on visit counts
func (mcts *AGMCTS) GetActionProbabilities() []float64 {
	// Make sure search has been performed
	if mcts.rootNode == nil || len(mcts.rootNode.Children) == 0 {
		mcts.Search()
	}

	// Calculate the total number of visits to children
	totalVisits := 0
	for _, child := range mcts.rootNode.Children {
		totalVisits += child.Visits
	}

	// If no visits, return uniform distribution
	if totalVisits == 0 {
		boardSize := 3
		probs := make([]float64, boardSize*boardSize)
		numValidMoves := len(mcts.rootNode.GameState.GetValidMoves())
		if numValidMoves > 0 {
			uniformProb := 1.0 / float64(numValidMoves)
			for _, move := range mcts.rootNode.GameState.GetValidMoves() {
				probs[move.Row*boardSize+move.Col] = uniformProb
			}
		}
		return probs
	}

	// Create probability distribution based on visit counts
	boardSize := 3
	probs := make([]float64, boardSize*boardSize)

	// Set probabilities based on visit counts
	for _, child := range mcts.rootNode.Children {
		if child.Move != nil {
			index := child.Move.Row*boardSize + child.Move.Col
			probs[index] = float64(child.Visits) / float64(totalVisits)
		}
	}

	return probs
}

// GetRootValue returns the estimated value of the root state
func (mcts *AGMCTS) GetRootValue() float64 {
	// Make sure search has been performed
	if mcts.rootNode == nil {
		mcts.Search()
	}

	// If the root node hasn't been visited enough, use value network directly
	if mcts.rootNode.Visits < 10 {
		return mcts.valueNetwork.Predict(mcts.rootNode.GameState)
	}

	// Return the average value from MCTS
	return mcts.rootNode.TotalValue / float64(mcts.rootNode.Visits)
}
