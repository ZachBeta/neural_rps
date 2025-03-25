package mcts

import (
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// RPSMCTSParams contains parameters for the MCTS algorithm
type RPSMCTSParams struct {
	NumSimulations   int
	ExplorationConst float64
	DirichletNoise   bool
	DirichletWeight  float64
	DirichletAlpha   float64
}

// DefaultRPSMCTSParams returns default MCTS parameters
func DefaultRPSMCTSParams() RPSMCTSParams {
	return RPSMCTSParams{
		NumSimulations:   800,
		ExplorationConst: 1.0,
		DirichletNoise:   true,
		DirichletWeight:  0.25,
		DirichletAlpha:   0.03,
	}
}

// RPSMCTS implements the Monte Carlo Tree Search algorithm for RPS
type RPSMCTS struct {
	PolicyNetwork *neural.RPSPolicyNetwork
	ValueNetwork  *neural.RPSValueNetwork
	Params        RPSMCTSParams
	Root          *RPSMCTSNode
}

// NewRPSMCTS creates a new MCTS instance
func NewRPSMCTS(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork, params RPSMCTSParams) *RPSMCTS {
	return &RPSMCTS{
		PolicyNetwork: policyNetwork,
		ValueNetwork:  valueNetwork,
		Params:        params,
		Root:          nil,
	}
}

// SetRootState sets the root state of the search tree
func (mcts *RPSMCTS) SetRootState(state *game.RPSGame) {
	// Get policy priors from the neural network
	priors := mcts.PolicyNetwork.Predict(state)

	// Create a new root node
	mcts.Root = NewRPSMCTSNode(state.Copy(), nil, nil, priors)
}

// Search performs the MCTS algorithm and returns the best move
func (mcts *RPSMCTS) Search() *RPSMCTSNode {
	if mcts.Root == nil {
		return nil
	}

	// Expand the root node if needed
	if len(mcts.Root.Children) == 0 {
		priors := mcts.PolicyNetwork.Predict(mcts.Root.GameState)
		mcts.Root.ExpandAll(priors)
	}

	// Run simulations
	for i := 0; i < mcts.Params.NumSimulations; i++ {
		// Selection phase
		node := mcts.selection(mcts.Root)

		// Expansion phase (if needed)
		if !node.GameState.IsGameOver() && node.Visits > 0 {
			priors := mcts.PolicyNetwork.Predict(node.GameState)
			node.ExpandAll(priors)

			// If expansion created children, select one of them
			if len(node.Children) > 0 {
				node = node.Children[0] // Select first child for simplicity
			}
		}

		// Evaluation phase
		value := mcts.evaluate(node)

		// Backpropagation phase
		node.UpdateRecursive(value)
	}

	// Return the most visited child of the root
	return mcts.Root.MostVisitedChild()
}

// selection traverses the tree to find a node to expand
func (mcts *RPSMCTS) selection(node *RPSMCTSNode) *RPSMCTSNode {
	// Keep traversing until we reach a leaf node or a terminal state
	for len(node.Children) > 0 && !node.GameState.IsGameOver() {
		node = node.SelectChild(mcts.Params.ExplorationConst)
		if node.Visits == 0 {
			// Found an unvisited node, return it
			return node
		}
	}

	return node
}

// evaluate estimates the value of a node
func (mcts *RPSMCTS) evaluate(node *RPSMCTSNode) float64 {
	// If game is over, return the actual outcome
	if node.GameState.IsGameOver() {
		winner := node.GameState.GetWinner()
		currentPlayer := node.GameState.CurrentPlayer

		// Return 1 for win, 0.5 for draw, 0 for loss
		if winner == game.NoPlayer {
			return 0.5 // Draw
		} else if (winner == game.Player1 && currentPlayer == game.Player1) ||
			(winner == game.Player2 && currentPlayer == game.Player2) {
			return 1.0 // Win
		} else {
			return 0.0 // Loss
		}
	}

	// Otherwise, use the value network to predict the outcome
	return mcts.ValueNetwork.Predict(node.GameState)
}

// GetBestMove returns the best move according to the MCTS search
func (mcts *RPSMCTS) GetBestMove() *game.RPSMove {
	if mcts.Root == nil {
		return nil
	}

	bestChild := mcts.Search()
	if bestChild != nil && bestChild.Move != nil {
		return bestChild.Move
	}

	return nil
}
