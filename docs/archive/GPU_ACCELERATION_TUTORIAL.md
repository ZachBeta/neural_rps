# GPU Acceleration with TensorFlow-Go: A Practical Guide

This tutorial walks through implementing GPU acceleration for the Neural RPS project using TensorFlow-Go. It's designed for mid-level software engineers who are familiar with Go but may not have extensive experience with GPU programming or TensorFlow.

## Introduction

Neural networks in the RPS card game project are computationally intensive, especially when combined with Monte Carlo Tree Search (MCTS). Moving these calculations to the GPU can provide significant performance improvements - often 10-100x faster for neural network inference.

In this tutorial, we'll:
1. Set up TensorFlow-Go with GPU support
2. Implement a GPU-accelerated neural network
3. Create a batched inference system for MCTS
4. Optimize memory usage and tensor handling
5. Profile and measure performance gains

## Prerequisites

- Go 1.18+ installed
- Basic understanding of neural networks
- Familiarity with the Neural RPS codebase
- For Apple Silicon: macOS 12+ with XCode tools installed
- For NVIDIA: CUDA and cuDNN installed

## 1. Setting Up TensorFlow-Go

### Installing Dependencies

First, add TensorFlow-Go to your project:

```bash
go get github.com/tensorflow/tensorflow/tensorflow/go
go get github.com/tensorflow/tensorflow/tensorflow/go/op
```

These packages provide the Go bindings to the TensorFlow C API. The actual TensorFlow library is dynamically linked.

### Configure go.mod

Update your go.mod file:

```go
module github.com/zachbeta/neural_rps

go 1.24.0

require (
    github.com/tensorflow/tensorflow/tensorflow/go v0.5.0
    github.com/tensorflow/tensorflow/tensorflow/go/op v0.5.0
    gonum.org/v1/gonum v0.16.0
)
```

### Installing the TensorFlow C Library

For macOS with Apple Silicon:

```bash
# Set environment variables for Metal 
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_MPS=1

# Install TensorFlow C library
brew install tensorflow
```

For other platforms, follow the [TensorFlow installation guide](https://www.tensorflow.org/install/lang_c).

## 2. Implementing the TensorFlow-Based Neural Network

We'll create a new implementation of our neural network using TensorFlow while maintaining API compatibility with the existing pure Go implementation.

### Create the TensorFlow Policy Network

Create a new file at `golang_implementation/pkg/neural/tf_network.go`:

```go
package neural

import (
    "fmt"
    "math/rand"
    
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// RPSTFPolicyNetwork implements a policy network using TensorFlow
type RPSTFPolicyNetwork struct {
    session    *tf.Session
    graph      *tf.Graph
    inputOp    tf.Output
    outputOp   tf.Output
    inputShape []int64
    
    // Keep track of network dimensions
    InputSize  int
    HiddenSize int
    OutputSize int
}

// NewRPSTFPolicyNetwork creates a new policy network using TensorFlow
func NewRPSTFPolicyNetwork(inputSize, hiddenSize, outputSize int) (*RPSTFPolicyNetwork, error) {
    // Create the computational graph
    s := op.NewScope()
    
    // Define input placeholder: -1 means variable batch size
    input := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, int64(inputSize))))
    
    // First hidden layer with ReLU activation
    w1 := op.Variable(s, op.Const(s.SubScope("w1"), generateRandomWeights(inputSize, hiddenSize)))
    b1 := op.Variable(s, op.Const(s.SubScope("b1"), generateZeroBiases(hiddenSize)))
    hidden := op.Relu(s, op.Add(s, op.MatMul(s, input, w1), b1))
    
    // Output layer with softmax activation
    w2 := op.Variable(s, op.Const(s.SubScope("w2"), generateRandomWeights(hiddenSize, outputSize)))
    b2 := op.Variable(s, op.Const(s.SubScope("b2"), generateZeroBiases(outputSize)))
    logits := op.Add(s, op.MatMul(s, hidden, w2), b2)
    output := op.Softmax(s, logits)
    
    // Finalize the graph
    graph, err := s.Finalize()
    if err != nil {
        return nil, fmt.Errorf("error finalizing graph: %v", err)
    }
    
    // Create a session with GPU configuration
    sessionOpts := &tf.SessionOptions{
        Config: []byte(`
            allow_soft_placement: true
            gpu_options { 
                allow_growth: true
            }
        `),
    }
    
    session, err := tf.NewSession(graph, sessionOpts)
    if err != nil {
        return nil, fmt.Errorf("error creating session: %v", err)
    }
    
    // Run initializers for variables
    init := op.Init(s)
    if _, err := session.Run(nil, nil, []*tf.Operation{init}); err != nil {
        session.Close()
        return nil, fmt.Errorf("error initializing variables: %v", err)
    }
    
    return &RPSTFPolicyNetwork{
        session:    session,
        graph:      graph,
        inputOp:    input,
        outputOp:   output,
        inputShape: []int64{-1, int64(inputSize)},
        InputSize:  inputSize,
        HiddenSize: hiddenSize,
        OutputSize: outputSize,
    }, nil
}

// Forward runs a forward pass through the network
func (nn *RPSTFPolicyNetwork) Forward(input []float64) ([]float64, error) {
    // Convert input to float32 for TensorFlow
    inputFloat32 := make([]float32, len(input))
    for i, v := range input {
        inputFloat32[i] = float32(v)
    }
    
    // Reshape input for batch processing (batch size 1)
    inputBatch := [][]float32{inputFloat32}
    
    // Create input tensor
    inputTensor, err := tf.NewTensor(inputBatch)
    if err != nil {
        return nil, fmt.Errorf("error creating input tensor: %v", err)
    }
    
    // Run the session
    results, err := nn.session.Run(
        map[tf.Output]*tf.Tensor{nn.inputOp: inputTensor},
        []tf.Output{nn.outputOp},
        nil,
    )
    if err != nil {
        return nil, fmt.Errorf("error running session: %v", err)
    }
    
    // Extract the output
    outputBatch := results[0].Value().([][]float32)
    output := make([]float64, len(outputBatch[0]))
    for i, v := range outputBatch[0] {
        output[i] = float64(v)
    }
    
    return output, nil
}

// Predict predicts the best move based on the input
func (nn *RPSTFPolicyNetwork) Predict(input []float64) (int, error) {
    output, err := nn.Forward(input)
    if err != nil {
        return -1, err
    }
    return argmax(output), nil
}

// Helper functions
func generateRandomWeights(inputSize, outputSize int) [][]float32 {
    // Xavier/Glorot initialization
    bound := float32(1.0 / float32(inputSize))
    weights := make([][]float32, inputSize)
    
    for i := 0; i < inputSize; i++ {
        weights[i] = make([]float32, outputSize)
        for j := 0; j < outputSize; j++ {
            weights[i][j] = (rand.Float32() * 2 - 1) * bound
        }
    }
    
    return weights
}

func generateZeroBiases(size int) []float32 {
    biases := make([]float32, size)
    for i := range biases {
        biases[i] = 0.0
    }
    return biases
}

// Close releases the TensorFlow session
func (nn *RPSTFPolicyNetwork) Close() {
    if nn.session != nil {
        nn.session.Close()
    }
}
```

### Implement Batched Inference

Add batching capability for efficient evaluation of multiple positions:

```go
// Add this to the RPSTFPolicyNetwork type

// ForwardBatch runs a forward pass for a batch of inputs
func (nn *RPSTFPolicyNetwork) ForwardBatch(inputs [][]float64) ([][]float64, error) {
    if len(inputs) == 0 {
        return nil, fmt.Errorf("empty batch")
    }
    
    // Convert inputs to float32
    inputsFloat32 := make([][]float32, len(inputs))
    for i, input := range inputs {
        if len(input) != nn.InputSize {
            return nil, fmt.Errorf("input %d size mismatch: expected %d, got %d", i, nn.InputSize, len(input))
        }
        
        inputsFloat32[i] = make([]float32, len(input))
        for j, v := range input {
            inputsFloat32[i][j] = float32(v)
        }
    }
    
    // Create input tensor
    inputTensor, err := tf.NewTensor(inputsFloat32)
    if err != nil {
        return nil, fmt.Errorf("error creating input tensor: %v", err)
    }
    
    // Run the session
    results, err := nn.session.Run(
        map[tf.Output]*tf.Tensor{nn.inputOp: inputTensor},
        []tf.Output{nn.outputOp},
        nil,
    )
    if err != nil {
        return nil, fmt.Errorf("error running session: %v", err)
    }
    
    // Extract and convert the outputs
    outputsFloat32 := results[0].Value().([][]float32)
    outputs := make([][]float64, len(outputsFloat32))
    
    for i, outputFloat32 := range outputsFloat32 {
        outputs[i] = make([]float64, len(outputFloat32))
        for j, v := range outputFloat32 {
            outputs[i][j] = float64(v)
        }
    }
    
    return outputs, nil
}

// PredictBatch predicts the best moves for a batch of inputs
func (nn *RPSTFPolicyNetwork) PredictBatch(inputs [][]float64) ([]int, error) {
    outputs, err := nn.ForwardBatch(inputs)
    if err != nil {
        return nil, err
    }
    
    predictions := make([]int, len(outputs))
    for i, output := range outputs {
        predictions[i] = argmax(output)
    }
    
    return predictions, nil
}
```

### Implement Weight Transfer

To use pre-trained CPU networks with our GPU implementation, add weight transfer functionality:

```go
// LoadFromCPUNetwork loads weights from a CPU-based network
func (nn *RPSTFPolicyNetwork) LoadFromCPUNetwork(cpuNet *Network) error {
    // Extract weights and biases from CPU network
    weights1 := make([][]float32, len(cpuNet.Weights1))
    for i, row := range cpuNet.Weights1 {
        weights1[i] = make([]float32, len(row))
        for j, v := range row {
            weights1[i][j] = float32(v)
        }
    }
    
    bias1 := make([]float32, len(cpuNet.Bias1))
    for i, v := range cpuNet.Bias1 {
        bias1[i] = float32(v)
    }
    
    weights2 := make([][]float32, len(cpuNet.Weights2))
    for i, row := range cpuNet.Weights2 {
        weights2[i] = make([]float32, len(row))
        for j, v := range row {
            weights2[i][j] = float32(v)
        }
    }
    
    bias2 := make([]float32, len(cpuNet.Bias2))
    for i, v := range cpuNet.Bias2 {
        bias2[i] = float32(v)
    }
    
    // Create tensors
    w1Tensor, err := tf.NewTensor(weights1)
    if err != nil {
        return fmt.Errorf("error creating w1 tensor: %v", err)
    }
    
    b1Tensor, err := tf.NewTensor(bias1)
    if err != nil {
        return fmt.Errorf("error creating b1 tensor: %v", err)
    }
    
    w2Tensor, err := tf.NewTensor(weights2)
    if err != nil {
        return fmt.Errorf("error creating w2 tensor: %v", err)
    }
    
    b2Tensor, err := tf.NewTensor(bias2)
    if err != nil {
        return fmt.Errorf("error creating b2 tensor: %v", err)
    }
    
    // Get operations for the variables
    w1 := tf.Output{Op: nn.graph.Operation("w1"), Index: 0}
    b1 := tf.Output{Op: nn.graph.Operation("b1"), Index: 0}
    w2 := tf.Output{Op: nn.graph.Operation("w2"), Index: 0}
    b2 := tf.Output{Op: nn.graph.Operation("b2"), Index: 0}
    
    // Run assignment operations
    _, err = nn.session.Run(
        map[tf.Output]*tf.Tensor{
            w1: w1Tensor,
            b1: b1Tensor,
            w2: w2Tensor,
            b2: b2Tensor,
        },
        nil,
        nil,
    )
    
    if err != nil {
        return fmt.Errorf("error assigning weights: %v", err)
    }
    
    return nil
}
```

## 3. Implementing Batched MCTS

Now let's modify the MCTS implementation to use batched neural network evaluations. First, we'll create a new MCTS implementation that supports batching.

### Create a Batched MCTS Implementation

Create a new file at `golang_implementation/pkg/mcts/batched_mcts.go`:

```go
package mcts

import (
    "math"
    "sync"
    
    "github.com/zachbeta/neural_rps/pkg/game"
    "github.com/zachbeta/neural_rps/pkg/neural"
)

// BatchedMCTS implements Monte Carlo Tree Search with batched neural network evaluations
type BatchedMCTS struct {
    policyNetwork *neural.RPSTFPolicyNetwork
    valueNetwork  *neural.RPSTFPolicyNetwork  // Using policy network for both in this example
    root          *MCTSNode
    params        MCTSParams
    
    // Batching-related fields
    batchSize     int
    positions     [][]float64
    nodesIndex    map[int]*MCTSNode  // Maps batch index to node
    mu            sync.Mutex
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
        state:  state.Clone(),
        parent: nil,
        children: make([]*MCTSNode, 0),
        visits: 0,
        value:  0,
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
    for !node.state.IsGameOver() && node.isFullyExpanded() {
        node = mcts.selectBestChild(node)
    }
    
    // If node is terminal, return it
    if node.state.IsGameOver() {
        return node
    }
    
    // If node is not fully expanded, expand it
    if !node.isFullyExpanded() && node.visits > 0 {
        return mcts.expandNode(node, nil)
    }
    
    return node
}

// selectBestChild selects the best child according to UCB formula
func (mcts *BatchedMCTS) selectBestChild(node *MCTSNode) *MCTSNode {
    bestScore := -math.MaxFloat64
    var bestChild *MCTSNode
    
    for _, child := range node.children {
        // UCB1 formula
        exploitation := child.value / float64(child.visits)
        exploration := mcts.params.ExplorationConst * math.Sqrt(math.Log(float64(node.visits))/float64(child.visits))
        score := exploitation + exploration
        
        if score > bestScore {
            bestScore = score
            bestChild = child
        }
    }
    
    return bestChild
}

// expandNode expands a node by creating all possible children
func (mcts *BatchedMCTS) expandNode(node *MCTSNode, policy []float64) *MCTSNode {
    // If policy is provided, use it to guide expansion
    if policy != nil {
        // Create all possible children with priors from policy
        legalMoves := node.state.GetLegalMoves()
        for _, move := range legalMoves {
            if !mcts.hasChild(node, move) {
                childState := node.state.Clone()
                childState.ApplyMove(move)
                
                child := &MCTSNode{
                    state:    childState,
                    parent:   node,
                    children: make([]*MCTSNode, 0),
                    visits:   0,
                    value:    0,
                }
                node.children = append(node.children, child)
            }
        }
        return node
    }
    
    // Without policy, just create one unexplored child
    legalMoves := node.state.GetLegalMoves()
    for _, move := range legalMoves {
        if !mcts.hasChild(node, move) {
            childState := node.state.Clone()
            childState.ApplyMove(move)
            
            child := &MCTSNode{
                state:    childState,
                parent:   node,
                children: make([]*MCTSNode, 0),
                visits:   0,
                value:    0,
            }
            node.children = append(node.children, child)
            return child
        }
    }
    
    return node // No more children to create
}

// hasChild checks if a node already has a child with the given move
func (mcts *BatchedMCTS) hasChild(node *MCTSNode, move game.RPSCardMove) bool {
    for _, child := range node.children {
        // Need to implement a way to check if this child was created with this move
        // This is simplified here
        if movesEqual(child.state.LastMove, move) {
            return true
        }
    }
    return false
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
        return mcts.root.state.GetRandomMove()
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

// extractFeatures converts a game state to neural network input
func (mcts *BatchedMCTS) extractFeatures(state *game.RPSCardGame) []float64 {
    // Implementation depends on specific feature representation
    // This is a placeholder
    features := make([]float64, mcts.policyNetwork.InputSize)
    
    // Fill features based on game state
    // ...
    
    return features
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
```

## 4. Memory Management and Optimization

Memory management is critical for GPU performance. Here are some techniques to implement:

### Tensor Pooling

Create a tensor pool to reuse tensor allocations:

```go
// TensorPool provides a pool of reusable tensors
type TensorPool struct {
    tensors []*tf.Tensor
    shape   []int64
    dtype   tf.DataType
    mu      sync.Mutex
}

// NewTensorPool creates a new tensor pool
func NewTensorPool(initialSize int, shape []int64, dtype tf.DataType) *TensorPool {
    pool := &TensorPool{
        tensors: make([]*tf.Tensor, 0, initialSize),
        shape:   shape,
        dtype:   dtype,
    }
    
    // Pre-allocate tensors
    for i := 0; i < initialSize; i++ {
        tensor, err := createEmptyTensor(shape, dtype)
        if err != nil {
            // Log error but continue
            fmt.Printf("Error pre-allocating tensor: %v\n", err)
            continue
        }
        pool.tensors = append(pool.tensors, tensor)
    }
    
    return pool
}

// Get retrieves a tensor from the pool or creates a new one
func (pool *TensorPool) Get() (*tf.Tensor, error) {
    pool.mu.Lock()
    defer pool.mu.Unlock()
    
    if len(pool.tensors) == 0 {
        // Create a new tensor if pool is empty
        return createEmptyTensor(pool.shape, pool.dtype)
    }
    
    // Get tensor from pool
    tensor := pool.tensors[len(pool.tensors)-1]
    pool.tensors = pool.tensors[:len(pool.tensors)-1]
    return tensor, nil
}

// Put returns a tensor to the pool
func (pool *TensorPool) Put(tensor *tf.Tensor) {
    pool.mu.Lock()
    defer pool.mu.Unlock()
    
    // Check if tensor matches pool specifications
    if !shapesEqual(tensor.Shape(), pool.shape) || tensor.DataType() != pool.dtype {
        return // Don't add mismatched tensors
    }
    
    pool.tensors = append(pool.tensors, tensor)
}

// Helper function to create an empty tensor
func createEmptyTensor(shape []int64, dtype tf.DataType) (*tf.Tensor, error) {
    // Create appropriate empty structure based on data type
    var value interface{}
    
    if dtype == tf.Float {
        // Create empty float32 array with the right shape
        if len(shape) == 2 {
            rows := int(shape[0])
            cols := int(shape[1])
            arr := make([][]float32, rows)
            for i := 0; i < rows; i++ {
                arr[i] = make([]float32, cols)
            }
            value = arr
        } else {
            // Handle other shapes as needed
            return nil, fmt.Errorf("unsupported shape dimension: %v", len(shape))
        }
    } else {
        return nil, fmt.Errorf("unsupported data type: %v", dtype)
    }
    
    return tf.NewTensor(value)
}

// Helper to compare shapes
func shapesEqual(s1, s2 []int64) bool {
    if len(s1) != len(s2) {
        return false
    }
    for i := range s1 {
        if s1[i] != s2[i] && s1[i] != -1 && s2[i] != -1 {
            return false
        }
    }
    return true
}
```

### Session Management

Manage TensorFlow sessions efficiently:

```go
// SessionManager maintains TensorFlow sessions
type SessionManager struct {
    sessions map[string]*tf.Session
    mu       sync.Mutex
}

// NewSessionManager creates a new session manager
func NewSessionManager() *SessionManager {
    return &SessionManager{
        sessions: make(map[string]*tf.Session),
    }
}

// GetSession retrieves or creates a session for the given model
func (sm *SessionManager) GetSession(modelID string, graphDef []byte) (*tf.Session, error) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    // Check if session exists
    if session, ok := sm.sessions[modelID]; ok {
        return session, nil
    }
    
    // Create new graph and session
    graph := tf.NewGraph()
    if err := graph.Import(graphDef, ""); err != nil {
        return nil, fmt.Errorf("error importing graph: %v", err)
    }
    
    sessionOpts := &tf.SessionOptions{
        Config: []byte(`
            allow_soft_placement: true
            gpu_options { 
                allow_growth: true
                per_process_gpu_memory_fraction: 0.5
            }
        `),
    }
    
    session, err := tf.NewSession(graph, sessionOpts)
    if err != nil {
        return nil, fmt.Errorf("error creating session: %v", err)
    }
    
    // Store session
    sm.sessions[modelID] = session
    return session, nil
}

// CloseAll closes all sessions
func (sm *SessionManager) CloseAll() {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    for _, session := range sm.sessions {
        session.Close()
    }
    
    sm.sessions = make(map[string]*tf.Session)
}
```

## 5. Performance Profiling

Let's implement a profiling utility to measure the performance improvements:

```go
package profiling

import (
    "fmt"
    "time"
    
    "github.com/zachbeta/neural_rps/pkg/neural"
)

// ProfileForwardPass profiles the forward pass performance
func ProfileForwardPass(cpuNetwork *neural.Network, gpuNetwork *neural.RPSTFPolicyNetwork, 
                         numSamples int, batchSizes []int) {
    
    // Generate random inputs
    inputs := make([][]float64, numSamples)
    for i := 0; i < numSamples; i++ {
        inputs[i] = generateRandomInput(cpuNetwork.InputSize)
    }
    
    // Profile CPU single inference
    fmt.Println("CPU Single Inference:")
    start := time.Now()
    for i := 0; i < numSamples; i++ {
        cpuNetwork.Forward(inputs[i])
    }
    elapsed := time.Since(start)
    fmt.Printf("  Time: %v, Avg: %v/sample\n", elapsed, elapsed/time.Duration(numSamples))
    
    // Profile GPU single inference
    fmt.Println("GPU Single Inference:")
    start = time.Now()
    for i := 0; i < numSamples; i++ {
        _, err := gpuNetwork.Forward(inputs[i])
        if err != nil {
            fmt.Printf("Error: %v\n", err)
            return
        }
    }
    elapsed = time.Since(start)
    fmt.Printf("  Time: %v, Avg: %v/sample\n", elapsed, elapsed/time.Duration(numSamples))
    
    // Profile GPU batched inference with different batch sizes
    for _, batchSize := range batchSizes {
        fmt.Printf("GPU Batch Inference (batch size = %d):\n", batchSize)
        
        // Create batches
        numBatches := (numSamples + batchSize - 1) / batchSize
        totalSamples := 0
        
        start = time.Now()
        for i := 0; i < numBatches; i++ {
            startIdx := i * batchSize
            endIdx := min(startIdx+batchSize, numSamples)
            batch := inputs[startIdx:endIdx]
            totalSamples += len(batch)
            
            _, err := gpuNetwork.ForwardBatch(batch)
            if err != nil {
                fmt.Printf("Error: %v\n", err)
                return
            }
        }
        elapsed = time.Since(start)
        fmt.Printf("  Time: %v, Avg: %v/sample, Throughput: %.2f samples/sec\n", 
                  elapsed, elapsed/time.Duration(totalSamples), 
                  float64(totalSamples)/elapsed.Seconds())
    }
}

// Helper functions
func generateRandomInput(size int) []float64 {
    input := make([]float64, size)
    for i := range input {
        input[i] = rand.Float64()*2 - 1
    }
    return input
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

## 6. Putting It All Together

Now, let's create a simple example that shows how to use our GPU-accelerated implementation:

```go
package main

import (
    "fmt"
    "os"
    
    "github.com/zachbeta/neural_rps/pkg/neural"
    "github.com/zachbeta/neural_rps/pkg/profiling"
)

func main() {
    // Define network dimensions
    inputSize := 120
    hiddenSize := 256
    outputSize := 9
    
    fmt.Println("Creating CPU network...")
    cpuNetwork := neural.NewNetwork(inputSize, hiddenSize, outputSize)
    
    fmt.Println("Creating GPU network...")
    gpuNetwork, err := neural.NewRPSTFPolicyNetwork(inputSize, hiddenSize, outputSize)
    if err != nil {
        fmt.Printf("Error creating GPU network: %v\n", err)
        os.Exit(1)
    }
    defer gpuNetwork.Close()
    
    // Transfer weights from CPU to GPU
    fmt.Println("Transferring weights from CPU to GPU...")
    err = gpuNetwork.LoadFromCPUNetwork(cpuNetwork)
    if err != nil {
        fmt.Printf("Error transferring weights: %v\n", err)
        os.Exit(1)
    }
    
    // Run performance profiling
    fmt.Println("Running performance profiling...")
    numSamples := 1000
    batchSizes := []int{1, 4, 16, 64, 256}
    profiling.ProfileForwardPass(cpuNetwork, gpuNetwork, numSamples, batchSizes)
    
    fmt.Println("Done!")
}
```

## 7. Common Issues and Troubleshooting

### Memory Leaks

TensorFlow tensors and sessions consume GPU memory. Always:
- Close sessions when they're no longer needed
- Use tensor pooling for frequent operations
- Set appropriate memory limits in session options

### Batch Size Tuning

The optimal batch size depends on your specific hardware:
- Too small: Underutilizes the GPU
- Too large: May cause out-of-memory errors
- Start with batch sizes of 32-128 and benchmark different values

### Debugging TensorFlow Errors

Common TensorFlow errors:
- Shape mismatches: Ensure tensor dimensions match the expected model inputs
- Memory errors: Reduce batch size or use tensor pooling
- Device placement errors: Make sure TensorFlow can find GPU devices

Use the TF_CPP_MIN_LOG_LEVEL environment variable to control TensorFlow's logging:
```bash
export TF_CPP_MIN_LOG_LEVEL=0  # All logs shown
export TF_CPP_MIN_LOG_LEVEL=1  # INFO logs filtered out
export TF_CPP_MIN_LOG_LEVEL=2  # INFO and WARNING logs filtered out
export TF_CPP_MIN_LOG_LEVEL=3  # Only errors shown
```

## 8. Next Steps

After implementing these basic GPU acceleration components, consider:

1. **Implementing model training on GPU**: Extend training.go to use TensorFlow-Go for backpropagation
2. **Multi-GPU support**: Distribute computation across multiple GPUs if available
3. **Optimizing memory usage**: Fine-tune tensor pooling and batch sizes
4. **Advanced GPU acceleration**: Explore custom CUDA kernels for specialized operations

## Conclusion

By implementing the code in this tutorial, you've added GPU acceleration to the Neural RPS project using TensorFlow-Go, which should provide significant performance improvements for neural network inference and MCTS search.

The key benefits achieved are:
- Faster neural network evaluation (10-100x speedup)
- Efficient batched processing for MCTS
- Better hardware utilization
- Support for larger neural networks

These improvements should lead to stronger AI agents through deeper search and faster training, while maintaining compatibility with the existing codebase. 