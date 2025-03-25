package mcts

import (
	"runtime"
	"sync"

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
	// Check if we should use parallel search
	// Use parallel search for large simulation counts on multi-core systems
	if mcts.Params.NumSimulations > 100 && runtime.NumCPU() > 2 {
		return mcts.searchParallel()
	}

	// Use original serial search for small simulation counts or single-core systems
	return mcts.searchSerial()
}

// searchSerial performs serial MCTS (original implementation)
func (mcts *RPSMCTS) searchSerial() *RPSMCTSNode {
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

// searchParallel performs parallel MCTS using multiple goroutines
func (mcts *RPSMCTS) searchParallel() *RPSMCTSNode {
	if mcts.Root == nil {
		return nil
	}

	// Expand the root node if needed (this needs to be done before parallelization)
	if len(mcts.Root.Children) == 0 {
		priors := mcts.PolicyNetwork.Predict(mcts.Root.GameState)
		mcts.Root.ExpandAll(priors)
	}

	// Determine optimal worker count
	// Use n-1 workers to avoid saturating all cores
	numWorkers := runtime.NumCPU() - 1
	if numWorkers < 1 {
		numWorkers = 1
	}

	// Calculate simulations per worker
	simsPerWorker := mcts.Params.NumSimulations / numWorkers
	extraSims := mcts.Params.NumSimulations % numWorkers

	// Create mutex for thread-safe tree access
	treeMutex := &sync.RWMutex{}

	// Create wait group to synchronize workers
	var wg sync.WaitGroup

	// Start worker goroutines
	for i := 0; i < numWorkers; i++ {
		// Calculate how many simulations this worker should perform
		workerSims := simsPerWorker
		if i == 0 {
			// First worker gets any extra simulations
			workerSims += extraSims
		}

		wg.Add(1)
		go func(simCount int) {
			defer wg.Done()

			// Each worker performs its share of simulations
			for j := 0; j < simCount; j++ {
				// Selection phase (with read lock)
				treeMutex.RLock()
				node := mcts.selectionThreadSafe(mcts.Root)
				treeMutex.RUnlock()

				// Local copy of the selected node's game state to avoid locks during evaluation
				localState := node.GameState.Copy()

				// Check if expansion is needed, using a local check to minimize lock time
				needsExpansion := !localState.IsGameOver() && node.Visits > 0 && len(node.Children) == 0

				// Expansion phase (with write lock, only if needed)
				if needsExpansion {
					// Get policy network prediction outside the lock
					priors := mcts.PolicyNetwork.Predict(localState)

					// Take write lock for expansion
					treeMutex.Lock()

					// Double-check that expansion is still needed (another thread might have expanded)
					if !node.GameState.IsGameOver() && node.Visits > 0 && len(node.Children) == 0 {
						node.ExpandAll(priors)

						// If expansion created children, select one of them
						if len(node.Children) > 0 {
							node = node.Children[0]
						}
					}

					treeMutex.Unlock()
				}

				// Evaluation phase (thread-safe, no locks needed)
				// Use the local copy of the game state
				value := mcts.evaluateState(localState)

				// Backpropagation phase (with write lock)
				treeMutex.Lock()
				mcts.backpropagateThreadSafe(node, value)
				treeMutex.Unlock()
			}
		}(workerSims)
	}

	// Wait for all workers to complete
	wg.Wait()

	// Return the most visited child of the root
	return mcts.Root.MostVisitedChild()
}

// selectionThreadSafe is a thread-safe version of selection
// Caller must hold at least a read lock
func (mcts *RPSMCTS) selectionThreadSafe(node *RPSMCTSNode) *RPSMCTSNode {
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

// backpropagateThreadSafe is a thread-safe version of backpropagation
// Caller must hold a write lock
func (mcts *RPSMCTS) backpropagateThreadSafe(node *RPSMCTSNode, value float64) {
	// Update this node
	node.Update(value)

	// Update parent recursively
	if node.Parent != nil {
		// Flip value perspective for parent (from opponent's point of view)
		mcts.backpropagateThreadSafe(node.Parent, 1.0-value)
	}
}

// evaluateState evaluates a game state without using the node
// This avoids needing to lock the tree during evaluation
func (mcts *RPSMCTS) evaluateState(state *game.RPSGame) float64 {
	// If game is over, return actual outcome
	if state.IsGameOver() {
		winner := state.GetWinner()

		if winner == game.NoPlayer {
			return 0.5 // Draw
		} else if winner == state.CurrentPlayer {
			return 1.0 // Win for current player
		} else {
			return 0.0 // Loss for current player
		}
	}

	// Otherwise, use value network for position evaluation
	return mcts.ValueNetwork.Predict(state)
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
	// If game is over, return actual outcome
	if node.GameState.IsGameOver() {
		winner := node.GameState.GetWinner()

		if winner == game.NoPlayer {
			return 0.5 // Draw
		} else if winner == node.GameState.CurrentPlayer {
			return 1.0 // Win for current player
		} else {
			return 0.0 // Loss for current player
		}
	}

	// Otherwise, use value network for position evaluation
	return mcts.ValueNetwork.Predict(node.GameState)
}

// GetBestMove returns the best move according to MCTS
func (mcts *RPSMCTS) GetBestMove() *game.RPSMove {
	bestNode := mcts.Search()
	if bestNode == nil || bestNode.Move == nil {
		return nil
	}
	return bestNode.Move
}
