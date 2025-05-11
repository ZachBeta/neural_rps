package mcts

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/zachbeta/neural_rps/pkg/game"
	"github.com/zachbeta/neural_rps/pkg/neural/gpu"
)

// DefaultBatchSize is the default batch size for neural network evaluations
const DefaultBatchSize = 64

// MaxWaitTime is the maximum time to wait for batch completion
const MaxWaitTime = 5 * time.Millisecond

// GPUBatchedMCTS implements Monte Carlo Tree Search with GPU-accelerated batch evaluation
type GPUBatchedMCTS struct {
	// Base fields replicating BatchedMCTS
	root      *MCTSNode
	params    MCTSParams
	batchSize int

	// GPU clients
	policyClient *gpu.NeuralClient
	valueClient  *gpu.NeuralClient

	// Batch processing
	policyQueue      []PolicyBatchItem
	valueQueue       []ValueBatchItem
	policyQueueMutex sync.Mutex
	valueQueueMutex  sync.Mutex

	// Configuration
	maxWaitTime time.Duration

	// Statistics
	totalPolicyBatches int
	totalValueBatches  int
	totalNodes         int
}

// PolicyBatchItem represents a state waiting for policy network evaluation
type PolicyBatchItem struct {
	state     GameState
	resultCh  chan []float32
	features  []float32
	nodeIndex int
}

// ValueBatchItem represents a state waiting for value network evaluation
type ValueBatchItem struct {
	state     GameState
	resultCh  chan float32
	features  []float32
	nodeIndex int
}

// NewGPUBatchedMCTS creates a new MCTS instance with GPU-accelerated batch evaluation
func NewGPUBatchedMCTS(serviceAddr string, params MCTSParams) (*GPUBatchedMCTS, error) {
	// Create GPU clients
	policyClient, err := gpu.NewNeuralClient(serviceAddr, "policy")
	if err != nil {
		return nil, fmt.Errorf("failed to create policy network client: %v", err)
	}

	valueClient, err := gpu.NewNeuralClient(serviceAddr, "value")
	if err != nil {
		return nil, fmt.Errorf("failed to create value network client: %v", err)
	}

	return &GPUBatchedMCTS{
		params:             params,
		policyClient:       policyClient,
		valueClient:        valueClient,
		policyQueue:        make([]PolicyBatchItem, 0, DefaultBatchSize),
		valueQueue:         make([]ValueBatchItem, 0, DefaultBatchSize),
		batchSize:          DefaultBatchSize,
		maxWaitTime:        MaxWaitTime,
		totalPolicyBatches: 0,
		totalValueBatches:  0,
		totalNodes:         0,
	}, nil
}

// SetRootState sets the root state for the search
func (mcts *GPUBatchedMCTS) SetRootState(state GameState) {
	mcts.root = &MCTSNode{
		State:      state.Clone(),
		Parent:     nil,
		Children:   make([]*MCTSNode, 0),
		Visits:     0,
		TotalValue: 0,
		Prior:      0.0,
	}
}

// SetBatchSize sets the batch size for neural network operations
func (mcts *GPUBatchedMCTS) SetBatchSize(size int) {
	if size < 1 {
		size = 1
	}
	mcts.batchSize = size
}

// SetMaxWaitTime sets the maximum time to wait before flushing non-full batches
func (mcts *GPUBatchedMCTS) SetMaxWaitTime(duration time.Duration) {
	mcts.maxWaitTime = duration
}

// Search runs the MCTS algorithm with GPU batched operations and returns the best move
func (mcts *GPUBatchedMCTS) Search(ctx context.Context) game.RPSCardMove {
	// Start background workers for batch processing
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	go mcts.policyWorker(workerCtx)
	go mcts.valueWorker(workerCtx)

	if mcts.root == nil {
		panic("Root state not set")
	}

	// Run simulations
	for i := 0; i < mcts.params.NumSimulations; i++ {
		select {
		case <-ctx.Done():
			return mcts.selectBestMove(ctx)
		default:
			mcts.iteration(ctx)
		}
	}

	return mcts.selectBestMove(ctx)
}

// iteration performs a single MCTS iteration
func (mcts *GPUBatchedMCTS) iteration(ctx context.Context) {
	node := mcts.selectNode(mcts.root)

	if node.State.IsGameOver() {
		winner := node.State.GetWinner()
		var value float64
		if winner == game.NoPlayer {
			value = 0.0
		} else if winner == node.State.GetCurrentPlayer() {
			value = 1.0
		} else {
			value = -1.0
		}
		mcts.backpropagate(node, value)
		return
	}

	if node.Visits == 0 {
		mcts.queueNodeForEvaluation(ctx, node)
		return
	}

	if len(node.Children) == 0 {
		mcts.backpropagate(node, 0.0)
		return
	}

	childToExplore := mcts.selectBestChild(node)
	if childToExplore == nil {
		mcts.backpropagate(node, 0.0)
		return
	}
}

// queueNodeForEvaluation queues a node for both policy and value evaluation
func (mcts *GPUBatchedMCTS) queueNodeForEvaluation(ctx context.Context, node *MCTSNode) {
	rpsAdapter, ok := node.State.(*RPSGameStateAdapter)
	if !ok {
		panic("GPUBatchedMCTS expects node.State to be adaptable to RPSGameStateAdapter for extractFeatures")
	}
	features := mcts.extractFeatures(rpsAdapter.RPSCardGame)

	policyResultCh := make(chan []float32, 1)
	mcts.policyQueueMutex.Lock()
	mcts.policyQueue = append(mcts.policyQueue, PolicyBatchItem{
		state:     node.State,
		resultCh:  policyResultCh,
		features:  features,
		nodeIndex: mcts.totalNodes,
	})
	shouldFlushPolicy := len(mcts.policyQueue) >= mcts.batchSize
	mcts.policyQueueMutex.Unlock()

	valueResultCh := make(chan float32, 1)
	mcts.valueQueueMutex.Lock()
	mcts.valueQueue = append(mcts.valueQueue, ValueBatchItem{
		state:     node.State,
		resultCh:  valueResultCh,
		features:  features,
		nodeIndex: mcts.totalNodes,
	})
	shouldFlushValue := len(mcts.valueQueue) >= mcts.batchSize
	mcts.valueQueueMutex.Unlock()

	mcts.totalNodes++

	if shouldFlushPolicy {
		go mcts.flushPolicyQueue(ctx)
	}
	if shouldFlushValue {
		go mcts.flushValueQueue(ctx)
	}

	select {
	case policy := <-policyResultCh:
		mcts.expandNode(node, policy)
		select {
		case value := <-valueResultCh:
			mcts.backpropagate(node, float64(value))
		case <-ctx.Done():
			return
		case <-time.After(mcts.maxWaitTime):
			mcts.backpropagate(node, 0.0)
			return
		}
	case <-ctx.Done():
		return
	case <-time.After(mcts.maxWaitTime):
		mcts.expandNode(node, nil)
		select {
		case value := <-valueResultCh:
			mcts.backpropagate(node, float64(value))
		case <-ctx.Done():
			return
		case <-time.After(mcts.maxWaitTime / 2):
			mcts.backpropagate(node, 0.0)
			return
		}
		return
	}
}

// queueForValueEvaluation queues a node for value evaluation only
func (mcts *GPUBatchedMCTS) queueForValueEvaluation(ctx context.Context, node *MCTSNode) {
	rpsAdapter, ok := node.State.(*RPSGameStateAdapter)
	if !ok {
		panic("GPUBatchedMCTS expects node.State to be adaptable to RPSGameStateAdapter for extractFeatures")
	}
	features := mcts.extractFeatures(rpsAdapter.RPSCardGame)
	valueResultCh := make(chan float32, 1)
	mcts.valueQueueMutex.Lock()
	mcts.valueQueue = append(mcts.valueQueue, ValueBatchItem{
		state:     node.State,
		resultCh:  valueResultCh,
		features:  features,
		nodeIndex: mcts.totalNodes,
	})
	shouldFlush := len(mcts.valueQueue) >= mcts.batchSize
	mcts.valueQueueMutex.Unlock()

	if shouldFlush {
		go mcts.flushValueQueue(ctx)
	}
	select {
	case value := <-valueResultCh:
		mcts.backpropagate(node, float64(value))
	case <-ctx.Done():
		return
	case <-time.After(mcts.maxWaitTime):
		mcts.backpropagate(node, 0.0)
		return
	}
}

// policyWorker processes policy evaluation batches periodically
func (mcts *GPUBatchedMCTS) policyWorker(ctx context.Context) {
	ticker := time.NewTicker(mcts.maxWaitTime)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mcts.flushPolicyQueue(ctx)
		}
	}
}

// valueWorker processes value evaluation batches periodically
func (mcts *GPUBatchedMCTS) valueWorker(ctx context.Context) {
	ticker := time.NewTicker(mcts.maxWaitTime)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mcts.flushValueQueue(ctx)
		}
	}
}

// flushPolicyQueue evaluates all queued positions with the policy network
func (mcts *GPUBatchedMCTS) flushPolicyQueue(ctx context.Context) {
	mcts.policyQueueMutex.Lock()
	if len(mcts.policyQueue) == 0 {
		mcts.policyQueueMutex.Unlock()
		return
	}

	batch := mcts.policyQueue
	mcts.policyQueue = make([]PolicyBatchItem, 0, mcts.batchSize)
	mcts.policyQueueMutex.Unlock()

	inputs := make([][]float32, len(batch))
	for i, item := range batch {
		inputs[i] = item.features
	}

	mcts.totalPolicyBatches++
	outputs, err := mcts.policyClient.PredictBatch(ctx, inputs)

	for i, item := range batch {
		var policy []float32
		if err != nil {
			policy, _, err = mcts.policyClient.Predict(ctx, item.features)
			if err != nil {
				policy = makeUniformPolicy(mcts.policyClient.GetOutputSize())
			}
		} else {
			policy = outputs[i].Probabilities
		}

		select {
		case item.resultCh <- policy:
		default:
		}
	}
}

// flushValueQueue evaluates all queued positions with the value network
func (mcts *GPUBatchedMCTS) flushValueQueue(ctx context.Context) {
	mcts.valueQueueMutex.Lock()
	if len(mcts.valueQueue) == 0 {
		mcts.valueQueueMutex.Unlock()
		return
	}

	batch := mcts.valueQueue
	mcts.valueQueue = make([]ValueBatchItem, 0, mcts.batchSize)
	mcts.valueQueueMutex.Unlock()

	inputs := make([][]float32, len(batch))
	for i, item := range batch {
		inputs[i] = item.features
	}

	mcts.totalValueBatches++
	outputs, err := mcts.valueClient.PredictBatch(ctx, inputs)

	for i, item := range batch {
		var value float32
		if err != nil {
			_, value, err = mcts.valueClient.Predict(ctx, item.features)
			if err != nil {
				value = 0
			}
		} else {
			value = outputs[i].Value
		}

		select {
		case item.resultCh <- value:
		default:
		}
	}
}

// GetStats returns performance statistics for the GPU-accelerated MCTS
func (mcts *GPUBatchedMCTS) GetStats() map[string]interface{} {
	policyStats := mcts.policyClient.GetStats()
	valueStats := mcts.valueClient.GetStats()

	return map[string]interface{}{
		"total_simulations":     mcts.params.NumSimulations,
		"total_policy_batches":  mcts.totalPolicyBatches,
		"total_value_batches":   mcts.totalValueBatches,
		"total_nodes":           mcts.totalNodes,
		"avg_policy_batch_size": float64(policyStats.TotalPositions) / float64(policyStats.TotalCalls),
		"avg_value_batch_size":  float64(valueStats.TotalPositions) / float64(valueStats.TotalCalls),
		"avg_policy_latency_us": policyStats.AvgLatencyUs,
		"avg_value_latency_us":  valueStats.AvgLatencyUs,
	}
}

// Close releases resources used by the GPU-accelerated MCTS
func (mcts *GPUBatchedMCTS) Close() {
	mcts.policyClient.Close()
	mcts.valueClient.Close()
}

// selectNode traverses the tree to find a node to evaluate
func (mcts *GPUBatchedMCTS) selectNode(node *MCTSNode) *MCTSNode {
	for node != nil && !node.State.IsGameOver() && mcts.isFullyExpanded(node) {
		selectedChild := mcts.selectBestChild(node)
		if selectedChild == nil {
			break
		}
		node = selectedChild
	}
	return node
}

// isFullyExpanded checks if all possible actions from this node have been explored
func (mcts *GPUBatchedMCTS) isFullyExpanded(node *MCTSNode) bool {
	if node.Visits == 0 {
		return false
	}
	legalMoves := node.State.GetLegalMoves()
	if len(node.Children) < len(legalMoves) {
		return false
	}
	return true
}

// selectBestChild selects the best child according to UCB formula
func (mcts *GPUBatchedMCTS) selectBestChild(node *MCTSNode) *MCTSNode {
	bestScore := -math.MaxFloat64
	var bestChild *MCTSNode

	if len(node.Children) == 0 {
		return nil
	}

	for _, child := range node.Children {
		if child.Visits == 0 {
		}
		exploitation := 0.0
		if child.Visits > 0 {
			exploitation = child.TotalValue / float64(child.Visits)
		}
		exploration := mcts.params.ExplorationConst *
			float64(child.Prior) *
			math.Sqrt(float64(node.Visits)) /
			(1.0 + float64(child.Visits))

		score := exploitation + exploration

		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}
	return bestChild
}

// expandNode expands a node by creating one of its unexplored children
func (mcts *GPUBatchedMCTS) expandNode(node *MCTSNode, policy []float32) *MCTSNode {
	legalMoves := node.State.GetLegalMoves()

	if len(legalMoves) == 0 {
		if node.Visits == 0 {
			node.Visits++
		}
		return node
	}

	if policy == nil {
		policy = makeUniformPolicy(len(legalMoves))
	}
	if len(policy) != len(legalMoves) {
		policy = makeUniformPolicy(len(legalMoves))
	}

	existingMoves := make(map[string]bool)
	for _, childNode := range node.Children {
		existingMoves[childNode.Move.String()] = true
	}

	addedChild := false
	for i, move := range legalMoves {
		if !existingMoves[move.String()] {
			childState := node.State.Clone()
			childState.ApplyMove(move)
			newChild := &MCTSNode{
				State:      childState,
				Parent:     node,
				Children:   make([]*MCTSNode, 0),
				Visits:     0,
				TotalValue: 0,
				Prior:      policy[i],
				Move:       move,
			}
			node.Children = append(node.Children, newChild)
			addedChild = true
			return newChild
		}
	}

	if !addedChild && len(node.Children) > 0 {
		return node
	}
	return node
}

// backpropagate updates values up the tree
func (mcts *GPUBatchedMCTS) backpropagate(node *MCTSNode, value float64) {
	current := node
	flip := 1.0
	for current != nil {
		current.Visits++
		current.TotalValue += value * flip
		current = current.Parent
		flip *= -1.0
	}
}

// selectBestMove selects the best move at the root based on visits
func (mcts *GPUBatchedMCTS) selectBestMove(ctx context.Context) game.RPSCardMove {
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

// extractFeatures converts a game state to neural network input features
func (mcts *GPUBatchedMCTS) extractFeatures(state *game.RPSCardGame) []float32 {
	features := state.ToTensor()
	return features
}

// makeUniformPolicy creates a uniform policy distribution
func makeUniformPolicy(size int) []float32 {
	policy := make([]float32, size)
	uniformProb := float32(1.0 / float64(size))
	for i := range policy {
		policy[i] = uniformProb
	}
	return policy
}

// toFloat32Slice converts a float64 slice to float32
func toFloat32Slice(src []float64) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

// toFloat64Slice converts a float32 slice to float64
func toFloat64Slice(src []float32) []float64 {
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}
	return dst
}
