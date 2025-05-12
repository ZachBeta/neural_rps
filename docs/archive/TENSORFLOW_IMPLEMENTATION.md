# TensorFlow-Go Implementation Plan for Neural RPS

This document outlines the implementation strategy for integrating TensorFlow-Go into the Neural RPS project to achieve GPU acceleration.

## Overview

We'll use the core TensorFlow-Go bindings to directly leverage GPU acceleration while maintaining precise control over execution, memory management, and optimization. This approach prioritizes performance and reliability over higher-level abstractions.

## Phase 1: Core Neural Network Acceleration

### TensorFlow Model Definition

```go
package neural

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
    "math"
    "math/rand"
)

// RPSTFPolicyNetwork implements a policy network using TensorFlow
type RPSTFPolicyNetwork struct {
    session    *tf.Session
    graph      *tf.Graph
    inputOp    tf.Output
    outputOp   tf.Output
    inputShape []int64
}

func NewRPSTFPolicyNetwork(inputSize, hiddenSize, outputSize int) (*RPSTFPolicyNetwork, error) {
    // Create the graph
    s := op.NewScope()
    
    // Define input placeholder
    input := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, int64(inputSize))))
    
    // First hidden layer
    w1 := op.Variable(s, op.Const(s.SubScope("w1"), generateWeights(inputSize, hiddenSize)))
    b1 := op.Variable(s, op.Const(s.SubScope("b1"), generateBiases(hiddenSize)))
    hidden := op.Relu(s, op.Add(s, op.MatMul(s, input, w1), b1))
    
    // Output layer
    w2 := op.Variable(s, op.Const(s.SubScope("w2"), generateWeights(hiddenSize, outputSize)))
    b2 := op.Variable(s, op.Const(s.SubScope("b2"), generateBiases(outputSize)))
    logits := op.Add(s, op.MatMul(s, hidden, w2), b2)
    output := op.Softmax(s, logits)
    
    // Finalize the graph
    graph, err := s.Finalize()
    if err != nil {
        return nil, err
    }
    
    // Create the session with GPU configuration
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
        return nil, err
    }
    
    // Run initializers
    init := op.Init(s)
    _, err = session.Run(nil, nil, []*tf.Operation{init})
    if err != nil {
        session.Close()
        return nil, err
    }
    
    return &RPSTFPolicyNetwork{
        session:    session,
        graph:      graph,
        inputOp:    input,
        outputOp:   output,
        inputShape: []int64{-1, int64(inputSize)},
    }, nil
}

// Helper functions for network initialization
func generateWeights(inputSize, outputSize int) [][]float32 {
    // Xavier/Glorot initialization
    limit := float32(math.Sqrt(6.0 / float32(inputSize + outputSize)))
    weights := make([][]float32, inputSize)
    
    for i := range weights {
        weights[i] = make([]float32, outputSize)
        for j := range weights[i] {
            weights[i][j] = (rand.Float32() * 2 * limit) - limit
        }
    }
    return weights
}

func generateBiases(size int) []float32 {
    biases := make([]float32, size)
    // Initialize with small values near zero
    for i := range biases {
        biases[i] = rand.Float32() * 0.1
    }
    return biases
}
```

### Batched Inference Implementation

```go
// PredictBatch runs a batch of positions through the network
func (n *RPSTFPolicyNetwork) PredictBatch(positions [][]float32) ([][]float32, error) {
    // Create input tensor from positions
    inputTensor, err := tf.NewTensor(positions)
    if err != nil {
        return nil, err
    }
    
    // Run the session
    results, err := n.session.Run(
        map[tf.Output]*tf.Tensor{
            n.inputOp: inputTensor,
        },
        []tf.Output{n.outputOp},
        nil,
    )
    if err != nil {
        return nil, err
    }
    
    // Extract and return the predictions
    return results[0].Value().([][]float32), nil
}

// Single position prediction
func (n *RPSTFPolicyNetwork) Predict(position []float32) ([]float32, error) {
    batch := [][]float32{position}
    results, err := n.PredictBatch(batch)
    if err != nil {
        return nil, err
    }
    return results[0], nil
}
```

### Weight Transfer from CPU Models

```go
// LoadWeightsFrom loads weights from a CPU-based neural network
func (n *RPSTFPolicyNetwork) LoadWeightsFrom(cpuNetwork *RPSPolicyNetwork) error {
    // Extract weights from CPU network
    w1, b1, w2, b2 := cpuNetwork.GetWeights()
    
    // Convert to tensors
    w1Tensor, err := tf.NewTensor(w1)
    if err != nil {
        return err
    }
    
    b1Tensor, err := tf.NewTensor(b1)
    if err != nil {
        return err
    }
    
    w2Tensor, err := tf.NewTensor(w2)
    if err != nil {
        return err
    }
    
    b2Tensor, err := tf.NewTensor(b2)
    if err != nil {
        return err
    }
    
    // Create assign operations
    s := op.NewScope()
    
    w1Var := n.graph.Operation("w1")
    w1Assign := op.Assign(s, tf.Output{Op: w1Var, Index: 0}, op.Const(s, w1))
    
    b1Var := n.graph.Operation("b1")
    b1Assign := op.Assign(s, tf.Output{Op: b1Var, Index: 0}, op.Const(s, b1))
    
    w2Var := n.graph.Operation("w2")
    w2Assign := op.Assign(s, tf.Output{Op: w2Var, Index: 0}, op.Const(s, w2))
    
    b2Var := n.graph.Operation("b2")
    b2Assign := op.Assign(s, tf.Output{Op: b2Var, Index: 0}, op.Const(s, b2))
    
    // Run assign operations
    assignOps := []*tf.Operation{w1Assign.Op, b1Assign.Op, w2Assign.Op, b2Assign.Op}
    _, err = n.session.Run(nil, nil, assignOps)
    return err
}

// SaveModel saves the TensorFlow model to a file
func (n *RPSTFPolicyNetwork) SaveModel(path string) error {
    // Create a TensorFlow SavedModel
    s := op.NewScope()
    
    // Define input and output signature
    input := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, n.inputShape[1])))
    
    // Build the signature definition
    signatureInputs := map[string]tf.Output{
        "input": input,
    }
    signatureOutputs := map[string]tf.Output{
        "output": n.outputOp,
    }
    
    // Create a SavedModel builder
    builder := tf.NewSavedModelBuilder(path)
    builder.AddSignature("serve", signatureInputs, signatureOutputs)
    
    // Save the model
    return builder.Save(n.session)
}

// LoadModel loads a TensorFlow model from a file
func LoadRPSTFPolicyNetwork(path string) (*RPSTFPolicyNetwork, error) {
    // Load the model
    bundle, err := tf.LoadSavedModel(path, []string{"serve"}, nil)
    if err != nil {
        return nil, err
    }
    
    // Get the input and output operations
    input, err := bundle.Graph.Operation("input").Output(0), nil
    if err != nil {
        return nil, err
    }
    
    output, err := bundle.Graph.Operation("output").Output(0), nil
    if err != nil {
        return nil, err
    }
    
    // Create the network
    return &RPSTFPolicyNetwork{
        session:    bundle.Session,
        graph:      bundle.Graph,
        inputOp:    input,
        outputOp:   output,
        inputShape: []int64{-1, input.Shape()[1]},
    }, nil
}
```

## Phase 2: MCTS Acceleration

### Batched MCTS Implementation

```go
// BatchedMCTS collects positions for batch evaluation
type BatchedMCTS struct {
    network        *RPSTFPolicyNetwork
    batchSize      int
    positionBuffer [][]float32
    stateMap       map[int]MCTSNode // Maps buffer indices to nodes
    rootNode       *MCTSNode
    cpuct          float32          // Exploration constant
}

func NewBatchedMCTS(network *RPSTFPolicyNetwork, batchSize int) *BatchedMCTS {
    return &BatchedMCTS{
        network:        network,
        batchSize:      batchSize,
        positionBuffer: make([][]float32, 0, batchSize),
        stateMap:       make(map[int]MCTSNode),
        cpuct:          1.0, // Default exploration constant
    }
}

func (mcts *BatchedMCTS) Search(root *GameState, numSimulations int) *Action {
    // Create root node
    mcts.rootNode = &MCTSNode{
        state:       root,
        parent:      nil,
        children:    make([]*MCTSNode, 0),
        visits:      0,
        value:       0,
        priorProb:   1.0,
        isEvaluated: false,
    }
    
    // Evaluate root node if needed
    if !mcts.rootNode.isEvaluated {
        rootFeatures := root.ToFeatures()
        prediction, err := mcts.network.Predict(rootFeatures)
        if err != nil {
            log.Printf("Root evaluation error: %v", err)
            return nil
        }
        
        mcts.rootNode.processEvaluation(prediction)
    }
    
    for i := 0; i < numSimulations; i += mcts.batchSize {
        // Clear the batch buffers
        mcts.positionBuffer = mcts.positionBuffer[:0]
        clear(mcts.stateMap)
        
        // Fill batch with positions to evaluate
        count := 0
        for j := 0; j < mcts.batchSize && i+j < numSimulations; j++ {
            node := mcts.selectNode(mcts.rootNode)
            if node.needsEvaluation() {
                mcts.positionBuffer = append(mcts.positionBuffer, node.state.ToFeatures())
                mcts.stateMap[count] = node
                count++
            }
        }
        
        // Evaluate batch
        if count > 0 {
            predictions, err := mcts.network.PredictBatch(mcts.positionBuffer[:count])
            if err != nil {
                log.Printf("Batch evaluation error: %v", err)
                continue
            }
            
            // Process predictions
            for idx, pred := range predictions {
                node := mcts.stateMap[idx]
                node.processEvaluation(pred)
                mcts.backpropagate(node)
            }
        }
    }
    
    // Return best action
    return mcts.rootNode.bestAction()
}

// Functions for tree traversal and node selection
func (mcts *BatchedMCTS) selectNode(rootNode *MCTSNode) *MCTSNode {
    node := rootNode
    
    // Traverse tree until we find a leaf node
    for len(node.children) > 0 && node.isFullyExpanded() {
        node = mcts.selectBestChild(node)
    }
    
    // If the node is not terminal and not fully expanded, expand it
    if !node.state.IsTerminal() && !node.isFullyExpanded() {
        return mcts.expandNode(node)
    }
    
    return node
}

// selectBestChild uses PUCT formula (used in AlphaZero) to select the best child
func (mcts *BatchedMCTS) selectBestChild(node *MCTSNode) *MCTSNode {
    bestValue := float32(-math.MaxFloat32)
    var bestChild *MCTSNode
    
    for _, child := range node.children {
        // Calculate PUCT value
        // Q(s,a) + cpuct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        exploitation := child.value / float32(math.Max(1, float64(child.visits)))
        exploration := mcts.cpuct * child.priorProb * float32(math.Sqrt(float64(node.visits))) / float32(1+child.visits)
        puct := exploitation + exploration
        
        if puct > bestValue {
            bestValue = puct
            bestChild = child
        }
    }
    
    return bestChild
}

// expandNode creates a new child node
func (mcts *BatchedMCTS) expandNode(node *MCTSNode) *MCTSNode {
    // Get legal actions from this state
    actions := node.state.GetLegalActions()
    
    // If no legal actions, return this node
    if len(actions) == 0 {
        return node
    }
    
    // Find an action that hasn't been tried yet
    for _, action := range actions {
        // Check if this action has already been tried
        alreadyTried := false
        for _, child := range node.children {
            if actionsEqual(action, actionBetween(node.state, child.state)) {
                alreadyTried = true
                break
            }
        }
        
        if !alreadyTried {
            // Apply the action to get a new state
            newState := node.state.Clone()
            newState.ApplyAction(action)
            
            // Create a new child node
            childNode := &MCTSNode{
                state:       newState,
                parent:      node,
                children:    make([]*MCTSNode, 0),
                visits:      0,
                value:       0,
                priorProb:   1.0 / float32(len(actions)), // Uniform prior
                isEvaluated: false,
            }
            
            // Add the child to the node's children
            node.children = append(node.children, childNode)
            
            return childNode
        }
    }
    
    // If all actions have been tried, return this node
    return node
}

func (mcts *BatchedMCTS) backpropagate(node *MCTSNode) {
    value := node.value
    current := node
    
    for current != nil {
        current.visits++
        current.value += value
        value = -value // Flip the value for the opponent's perspective
        current = current.parent
    }
}
```

### MCTSNode Implementation

```go
// MCTSNode represents a node in the MCTS tree
type MCTSNode struct {
    state       *GameState
    parent      *MCTSNode
    children    []*MCTSNode
    visits      int
    value       float32    // Total value
    priorProb   float32    // Prior probability from policy network
    isEvaluated bool
    policyProbs []float32  // Policy probabilities for all actions
    valueEst    float32    // Value estimate from value network
}

func (n *MCTSNode) needsEvaluation() bool {
    return !n.isEvaluated && n.visits == 0
}

func (n *MCTSNode) processEvaluation(prediction []float32) {
    n.isEvaluated = true
    
    // Assuming prediction format: [policy_probs..., value]
    // Last element is the value, rest are policy probabilities
    n.valueEst = prediction[len(prediction)-1]
    n.policyProbs = prediction[:len(prediction)-1]
    
    // Initialize children with prior probabilities from policy
    actions := n.state.GetLegalActions()
    for _, action := range actions {
        actionIdx := action.GetIndex()
        if actionIdx < len(n.policyProbs) {
            // Create child node with prior probability from policy
            childState := n.state.Clone()
            childState.ApplyAction(action)
            
            child := &MCTSNode{
                state:       childState,
                parent:      n,
                children:    make([]*MCTSNode, 0),
                visits:      0,
                value:       0,
                priorProb:   n.policyProbs[actionIdx],
                isEvaluated: false,
            }
            
            n.children = append(n.children, child)
        }
    }
}

func (n *MCTSNode) isFullyExpanded() bool {
    if !n.isEvaluated {
        return false
    }
    
    // Get legal actions from this state
    actions := n.state.GetLegalActions()
    
    // If we have as many children as there are legal actions, 
    // the node is fully expanded
    return len(n.children) == len(actions)
}

func (n *MCTSNode) bestAction() *Action {
    // Select the child with the highest visit count
    var bestChild *MCTSNode
    bestVisits := -1
    
    for _, child := range n.children {
        if child.visits > bestVisits {
            bestVisits = child.visits
            bestChild = child
        }
    }
    
    if bestChild == nil {
        // No best child found, return a random legal action
        actions := n.state.GetLegalActions()
        if len(actions) > 0 {
            return actions[0]
        }
        return nil
    }
    
    // Get the action that leads to the best child
    return actionBetween(n.state, bestChild.state)
}

// Helper functions for MCTS
func actionBetween(parent, child *GameState) *Action {
    // Get legal actions from parent state
    actions := parent.GetLegalActions()
    
    // Try each action to see which one leads to the child state
    for _, action := range actions {
        testState := parent.Clone()
        testState.ApplyAction(action)
        
        if statesEqual(testState, child) {
            return action
        }
    }
    
    // If no matching action found, return nil
    return nil
}

func statesEqual(state1, state2 *GameState) bool {
    // Compare state representations
    // This implementation depends on how game states are represented
    return state1.GetHash() == state2.GetHash()
}

func actionsEqual(a1, a2 *Action) bool {
    // Implementation depends on how actions are represented
    return a1.GetIndex() == a2.GetIndex()
}
```

### Position Collection Optimization

```go
// Efficient position collection with preallocated memory
func (mcts *BatchedMCTS) collectPositionsForEvaluation(numPositions int) {
    // Preallocate if needed
    if cap(mcts.positionBuffer) < numPositions {
        mcts.positionBuffer = make([][]float32, 0, numPositions)
    }
    
    // Try to fill the buffer with positions that need evaluation
    nodesQueue := []*MCTSNode{mcts.rootNode}
    for len(mcts.positionBuffer) < numPositions && len(nodesQueue) > 0 {
        node := nodesQueue[0]
        nodesQueue = nodesQueue[1:]
        
        if node.needsEvaluation() {
            features := node.state.ToFeatures()
            mcts.positionBuffer = append(mcts.positionBuffer, features)
            mcts.stateMap[len(mcts.positionBuffer)-1] = node
        } else {
            // Add children to queue
            for _, child := range node.children {
                if !child.isFullyExpanded() {
                    nodesQueue = append(nodesQueue, child)
                }
            }
        }
    }
}

// Performance monitoring for MCTS
func (mcts *BatchedMCTS) GetStats() map[string]interface{} {
    stats := make(map[string]interface{})
    
    // Count total nodes in tree
    var countNodes func(*MCTSNode) int
    countNodes = func(node *MCTSNode) int {
        count := 1 // Count this node
        for _, child := range node.children {
            count += countNodes(child)
        }
        return count
    }
    
    // Calculate max depth of tree
    var maxDepth func(*MCTSNode, int) int
    maxDepth = func(node *MCTSNode, depth int) int {
        if len(node.children) == 0 {
            return depth
        }
        
        maxChildDepth := depth
        for _, child := range node.children {
            childDepth := maxDepth(child, depth+1)
            if childDepth > maxChildDepth {
                maxChildDepth = childDepth
            }
        }
        
        return maxChildDepth
    }
    
    // Collect statistics
    stats["total_nodes"] = countNodes(mcts.rootNode)
    stats["max_depth"] = maxDepth(mcts.rootNode, 0)
    stats["root_visits"] = mcts.rootNode.visits
    
    // Calculate branching factor
    if mcts.rootNode.visits > 0 {
        stats["avg_branching_factor"] = float64(stats["total_nodes"].(int)-1) / float64(stats["total_nodes"].(int) - len(mcts.rootNode.children))
    } else {
        stats["avg_branching_factor"] = 0.0
    }
    
    return stats
}

// Helper for selecting actions from predictions
func selectActionFromPrediction(prediction []float32) *Action {
    // Get the index of the highest probability
    var bestIdx int
    bestProb := float32(-1.0)
    
    for i, prob := range prediction {
        if prob > bestProb {
            bestProb = prob
            bestIdx = i
        }
    }
    
    // Convert index to action
    return ActionFromIndex(bestIdx)
}

// Helper to convert action index to Action object
func ActionFromIndex(idx int) *Action {
    // This implementation depends on how actions are represented
    return &Action{index: idx}
}
```

## Phase 3: NEAT Training Optimization

### Parallel Fitness Evaluation with GPU

```go
// Parallel evaluation of genomes using GPU batching
func (e *NEATEvaluator) EvaluatePopulation(population []*neat.Genome) {
    // Create TensorFlow policy networks for each genome
    networks := make([]*RPSTFPolicyNetwork, len(population))
    for i, genome := range population {
        networks[i] = convertGenomeToTFNetwork(genome)
    }
    
    // Create worker pool
    numWorkers := runtime.NumCPU()
    workerPool := make(chan int, numWorkers)
    var wg sync.WaitGroup
    
    // Distribute evaluation work
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            
            // Each worker gets a subset of genomes to evaluate
            for j := workerID; j < len(population); j += numWorkers {
                // Run tournament games in batches
                fitness := e.evaluateGenomeFitness(networks[j])
                population[j].Fitness = fitness
            }
        }(i)
    }
    
    wg.Wait()
}

// Convert a NEAT genome to a TensorFlow network
func convertGenomeToTFNetwork(genome *neat.Genome) (*RPSTFPolicyNetwork, error) {
    // Extract network topology from genome
    inputSize, hiddenSize, outputSize := extractNetworkTopology(genome)
    
    // Create a new TensorFlow network
    network, err := NewRPSTFPolicyNetwork(inputSize, hiddenSize, outputSize)
    if err != nil {
        return nil, err
    }
    
    // Extract weights from genome and load into network
    err = loadGenomeWeightsIntoTFNetwork(genome, network)
    if err != nil {
        return nil, err
    }
    
    return network, nil
}

// Extract network topology from a NEAT genome
func extractNetworkTopology(genome *neat.Genome) (int, int, int) {
    // Count input, hidden, and output nodes
    inputNodes := 0
    hiddenNodes := 0
    outputNodes := 0
    
    for _, node := range genome.Nodes {
        switch node.Type {
        case neat.InputNode:
            inputNodes++
        case neat.HiddenNode:
            hiddenNodes++
        case neat.OutputNode:
            outputNodes++
        }
    }
    
    // If no hidden nodes, use a default size
    if hiddenNodes == 0 {
        hiddenNodes = 10
    }
    
    return inputNodes, hiddenNodes, outputNodes
}

// Load genome weights into a TensorFlow network
func loadGenomeWeightsIntoTFNetwork(genome *neat.Genome, network *RPSTFPolicyNetwork) error {
    // Create weight matrices from NEAT genome
    inputSize, hiddenSize, outputSize := extractNetworkTopology(genome)
    
    // Initialize weight matrices with zeros
    w1 := make([][]float32, inputSize)
    for i := range w1 {
        w1[i] = make([]float32, hiddenSize)
    }
    
    w2 := make([][]float32, hiddenSize)
    for i := range w2 {
        w2[i] = make([]float32, outputSize)
    }
    
    // Initialize bias vectors with zeros
    b1 := make([]float32, hiddenSize)
    b2 := make([]float32, outputSize)
    
    // Create node ID to index mappings
    inputNodeIDs := make(map[int]int)
    hiddenNodeIDs := make(map[int]int)
    outputNodeIDs := make(map[int]int)
    
    inputIdx := 0
    hiddenIdx := 0
    outputIdx := 0
    
    for _, node := range genome.Nodes {
        switch node.Type {
        case neat.InputNode:
            inputNodeIDs[node.ID] = inputIdx
            inputIdx++
        case neat.HiddenNode:
            hiddenNodeIDs[node.ID] = hiddenIdx
            hiddenIdx++
        case neat.OutputNode:
            outputNodeIDs[node.ID] = outputIdx
            outputIdx++
        }
    }
    
    // Map NEAT connections to weight matrices
    for _, conn := range genome.Connections {
        if !conn.Enabled {
            continue
        }
        
        // Determine connection type (input->hidden, hidden->output)
        fromInput := false
        fromHidden := false
        toHidden := false
        toOutput := false
        
        if _, exists := inputNodeIDs[conn.InNode]; exists {
            fromInput = true
        }
        if _, exists := hiddenNodeIDs[conn.InNode]; exists {
            fromHidden = true
        }
        if _, exists := hiddenNodeIDs[conn.OutNode]; exists {
            toHidden = true
        }
        if _, exists := outputNodeIDs[conn.OutNode]; exists {
            toOutput = true
        }
        
        weight := float32(conn.Weight)
        
        // Assign weight to appropriate matrix
        if fromInput && toHidden {
            inIdx := inputNodeIDs[conn.InNode]
            outIdx := hiddenNodeIDs[conn.OutNode]
            w1[inIdx][outIdx] = weight
        } else if fromHidden && toOutput {
            inIdx := hiddenNodeIDs[conn.InNode]
            outIdx := outputNodeIDs[conn.OutNode]
            w2[inIdx][outIdx] = weight
        }
        // Skip other connections (input->output, etc.)
    }
    
    // Transfer weights to TensorFlow network
    s := op.NewScope()
    
    // Create tensors for weights and biases
    w1Tensor, err := tf.NewTensor(w1)
    if err != nil {
        return err
    }
    
    b1Tensor, err := tf.NewTensor(b1)
    if err != nil {
        return err
    }
    
    w2Tensor, err := tf.NewTensor(w2)
    if err != nil {
        return err
    }
    
    b2Tensor, err := tf.NewTensor(b2)
    if err != nil {
        return err
    }
    
    // Create assign operations
    w1Var := network.graph.Operation("w1")
    w1Assign := op.Assign(s, tf.Output{Op: w1Var, Index: 0}, op.Const(s, w1))
    
    b1Var := network.graph.Operation("b1")
    b1Assign := op.Assign(s, tf.Output{Op: b1Var, Index: 0}, op.Const(s, b1))
    
    w2Var := network.graph.Operation("w2")
    w2Assign := op.Assign(s, tf.Output{Op: w2Var, Index: 0}, op.Const(s, w2))
    
    b2Var := network.graph.Operation("b2")
    b2Assign := op.Assign(s, tf.Output{Op: b2Var, Index: 0}, op.Const(s, b2))
    
    // Run assign operations
    assignOps := []*tf.Operation{w1Assign.Op, b1Assign.Op, w2Assign.Op, b2Assign.Op}
    _, err = network.session.Run(nil, nil, assignOps)
    
    return err
}

// Convert TensorFlow network weights back to NEAT genome
func updateGenomeFromTFNetwork(genome *neat.Genome, network *RPSTFPolicyNetwork) error {
    // Fetch weights from TensorFlow network
    w1, b1, w2, b2, err := network.GetWeights()
    if err != nil {
        return err
    }
    
    // Create node ID to index mappings
    inputNodeIDs := make(map[int]int)
    hiddenNodeIDs := make(map[int]int)
    outputNodeIDs := make(map[int]int)
    
    inputIdx := 0
    hiddenIdx := 0
    outputIdx := 0
    
    for _, node := range genome.Nodes {
        switch node.Type {
        case neat.InputNode:
            inputNodeIDs[node.ID] = inputIdx
            inputIdx++
        case neat.HiddenNode:
            hiddenNodeIDs[node.ID] = hiddenIdx
            hiddenIdx++
        case neat.OutputNode:
            outputNodeIDs[node.ID] = outputIdx
            outputIdx++
        }
    }
    
    // Map weight matrices back to NEAT connections
    for _, conn := range genome.Connections {
        if !conn.Enabled {
            continue
        }
        
        // Determine connection type (input->hidden, hidden->output)
        fromInput := false
        fromHidden := false
        toHidden := false
        toOutput := false
        
        if _, exists := inputNodeIDs[conn.InNode]; exists {
            fromInput = true
        }
        if _, exists := hiddenNodeIDs[conn.InNode]; exists {
            fromHidden = true
        }
        if _, exists := hiddenNodeIDs[conn.OutNode]; exists {
            toHidden = true
        }
        if _, exists := outputNodeIDs[conn.OutNode]; exists {
            toOutput = true
        }
        
        // Update connection weight from appropriate matrix
        if fromInput && toHidden {
            inIdx := inputNodeIDs[conn.InNode]
            outIdx := hiddenNodeIDs[conn.OutNode]
            conn.Weight = float64(w1[inIdx][outIdx])
        } else if fromHidden && toOutput {
            inIdx := hiddenNodeIDs[conn.InNode]
            outIdx := outputNodeIDs[conn.OutNode]
            conn.Weight = float64(w2[inIdx][outIdx])
        }
        // Skip other connections (input->output, etc.)
    }
    
    return nil
}

// Helper function to get weights from TensorFlow network
func (n *RPSTFPolicyNetwork) GetWeights() ([][]float32, []float32, [][]float32, []float32, error) {
    // Get the weight operations
    w1Op := tf.Output{Op: n.graph.Operation("w1"), Index: 0}
    b1Op := tf.Output{Op: n.graph.Operation("b1"), Index: 0}
    w2Op := tf.Output{Op: n.graph.Operation("w2"), Index: 0}
    b2Op := tf.Output{Op: n.graph.Operation("b2"), Index: 0}
    
    // Run the session to get the weights
    results, err := n.session.Run(
        nil,
        []tf.Output{w1Op, b1Op, w2Op, b2Op},
        nil,
    )
    if err != nil {
        return nil, nil, nil, nil, err
    }
    
    // Extract the weights
    w1 := results[0].Value().([][]float32)
    b1 := results[1].Value().([]float32)
    w2 := results[2].Value().([][]float32)
    b2 := results[3].Value().([]float32)
    
    return w1, b1, w2, b2, nil
}

// Batched game simulation for fitness calculation
func (e *NEATEvaluator) evaluateGenomeFitness(network *RPSTFPolicyNetwork) float64 {
    const gamesPerBatch = 32
    const totalGames = 200
    
    opponent := e.getStandardOpponent()
    wins := 0
    
    // Run games in batches
    for gameStart := 0; gameStart < totalGames; gameStart += gamesPerBatch {
        // Initialize batch of games
        games := make([]*GameState, gamesPerBatch)
        for i := range games {
            games[i] = NewGameState()
        }
        
        // Play all games to completion
        activeBatchSize := gamesPerBatch
        for activeBatchSize > 0 {
            // Collect states for active games
            positions := make([][]float32, 0, activeBatchSize)
            gameIndices := make([]int, 0, activeBatchSize)
            
            for i, game := range games {
                if !game.IsTerminal() {
                    positions = append(positions, game.ToFeatures())
                    gameIndices = append(gameIndices, i)
                }
            }
            
            // No more active games
            if len(positions) == 0 {
                break
            }
            
            // Evaluate all positions in a single batch
            predictions, err := network.PredictBatch(positions)
            if err != nil {
                log.Printf("Batch prediction error: %v", err)
                continue
            }
            
            // Apply actions to games
            for i, idx := range gameIndices {
                action := selectActionFromPrediction(predictions[i])
                games[idx].ApplyAction(action)
                
                // Check if game is now terminal
                if games[idx].IsTerminal() {
                    if games[idx].Winner() == PlayerAgent {
                        wins++
                    }
                    activeBatchSize--
                }
            }
        }
    }
    
    return float64(wins) / totalGames
}
```

## Phase 4: Tournament System

```go
// Efficient tournament with GPU acceleration
func RunGPUTournament(agents []*Agent, gamesPerMatchup int) *TournamentResults {
    numAgents := len(agents)
    results := NewTournamentResults(numAgents)
    
    // Convert all agents to TensorFlow networks
    networks := make([]*RPSTFPolicyNetwork, numAgents)
    for i, agent := range agents {
        networks[i] = convertAgentToTFNetwork(agent)
    }
    
    // Define batch size for games
    const batchSize = 64
    
    // For each pair of agents
    for i := 0; i < numAgents; i++ {
        for j := i+1; j < numAgents; j++ {
            winsI := 0
            winsJ := 0
            
            // Run games in batches
            for gameStart := 0; gameStart < gamesPerMatchup; gameStart += batchSize {
                gamesToPlay := min(batchSize, gamesPerMatchup-gameStart)
                wins := playBatchedGames(networks[i], networks[j], gamesToPlay)
                winsI += wins
                winsJ += gamesToPlay - wins
            }
            
            // Record results
            results.RecordResult(i, j, winsI, winsJ)
        }
    }
    
    return results
}

// Play a batch of games between two networks
func playBatchedGames(network1, network2 *RPSTFPolicyNetwork, numGames int) int {
    // Initialize game states
    games := make([]*GameState, numGames)
    activePlayerIdx := make([]int, numGames) // 0 for network1, 1 for network2
    
    for i := range games {
        games[i] = NewGameState()
        activePlayerIdx[i] = 0 // network1 starts
    }
    
    network1Wins := 0
    activeGames := numGames
    
    // Play all games to completion
    for activeGames > 0 {
        // Collect states for network1's turn
        network1Positions := make([][]float32, 0, activeGames)
        network1GameIndices := make([]int, 0, activeGames)
        
        // Collect states for network2's turn
        network2Positions := make([][]float32, 0, activeGames)
        network2GameIndices := make([]int, 0, activeGames)
        
        // Sort games by active player
        for i, game := range games {
            if game.IsTerminal() {
                continue
            }
            
            if activePlayerIdx[i] == 0 {
                network1Positions = append(network1Positions, game.ToFeatures())
                network1GameIndices = append(network1GameIndices, i)
            } else {
                network2Positions = append(network2Positions, game.ToFeatures())
                network2GameIndices = append(network2GameIndices, i)
            }
        }
        
        // Process network1's moves
        if len(network1Positions) > 0 {
            predictions, _ := network1.PredictBatch(network1Positions)
            for i, idx := range network1GameIndices {
                action := selectActionFromPrediction(predictions[i])
                games[idx].ApplyAction(action)
                activePlayerIdx[idx] = 1 // Switch to network2
                
                // Check if game ended
                if games[idx].IsTerminal() {
                    if games[idx].Winner() == 0 {
                        network1Wins++
                    }
                    activeGames--
                }
            }
        }
        
        // Process network2's moves
        if len(network2Positions) > 0 {
            predictions, _ := network2.PredictBatch(network2Positions)
            for i, idx := range network2GameIndices {
                action := selectActionFromPrediction(predictions[i])
                games[idx].ApplyAction(action)
                activePlayerIdx[idx] = 0 // Switch to network1
                
                // Check if game ended
                if games[idx].IsTerminal() {
                    if games[idx].Winner() == 1 {
                        // Player 1 (network2) won
                    } else {
                        // Draw or player 0 (network1) won
                        network1Wins++
                    }
                    activeGames--
                }
            }
        }
        
        // If no moves were made, we're done
        if len(network1Positions) == 0 && len(network2Positions) == 0 {
            break
        }
    }
    
    return network1Wins
}

// Tournament results tracking
type TournamentResults struct {
    numAgents int
    wins      [][]int
    games     [][]int
}

func NewTournamentResults(numAgents int) *TournamentResults {
    wins := make([][]int, numAgents)
    games := make([][]int, numAgents)
    
    for i := range wins {
        wins[i] = make([]int, numAgents)
        games[i] = make([]int, numAgents)
    }
    
    return &TournamentResults{
        numAgents: numAgents,
        wins:      wins,
        games:     games,
    }
}

func (r *TournamentResults) RecordResult(i, j, winsI, winsJ int) {
    r.wins[i][j] = winsI
    r.wins[j][i] = winsJ
    r.games[i][j] = winsI + winsJ
    r.games[j][i] = winsI + winsJ
}

func (r *TournamentResults) GetWinRate(i, j int) float64 {
    if r.games[i][j] == 0 {
        return 0.5 // No games played
    }
    return float64(r.wins[i][j]) / float64(r.games[i][j])
}

func (r *TournamentResults) GetOverallWinRate(i int) float64 {
    totalWins := 0
    totalGames := 0
    
    for j := 0; j < r.numAgents; j++ {
        if i == j {
            continue // Skip self
        }
        totalWins += r.wins[i][j]
        totalGames += r.games[i][j]
    }
    
    if totalGames == 0 {
        return 0.5 // No games played
    }
    return float64(totalWins) / float64(totalGames)
}
```

## Memory Management & Performance Optimizations

### Tensor Pooling

```go
// TensorPool for reusing tensors
type TensorPool struct {
    inputTensors  []*tf.Tensor
    mutex         sync.Mutex
}

func NewTensorPool(capacity int, shape []int64) *TensorPool {
    pool := &TensorPool{}
    for i := 0; i < capacity; i++ {
        // Pre-allocate tensors with zeros
        tensor, _ := tf.NewTensor(make([][]float32, shape[0], shape[1]))
        pool.inputTensors = append(pool.inputTensors, tensor)
    }
    return pool
}

func (p *TensorPool) GetTensor() *tf.Tensor {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    if len(p.inputTensors) == 0 {
        // Create a new tensor if pool is empty
        tensor, _ := tf.NewTensor(make([][]float32, 1, 100)) // Default size
        return tensor
    }
    
    // Get tensor from pool
    tensor := p.inputTensors[len(p.inputTensors)-1]
    p.inputTensors = p.inputTensors[:len(p.inputTensors)-1]
    return tensor
}

func (p *TensorPool) ReleaseTensor(tensor *tf.Tensor) {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    p.inputTensors = append(p.inputTensors, tensor)
}
```

### GPU Initialization & Configuration

```go
// Manage TensorFlow sessions efficiently
func InitializeGPU() {
    // Set environment variable to enable MPS backend for Metal
    os.Setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    os.Setenv("TF_MPS_ENABLE_METAL", "1")
    
    // Verify GPU availability
    log.Println("TensorFlow version:", tf.Version())
    
    // Test session
    s := op.NewScope()
    hello := op.Const(s, "Hello TensorFlow on GPU!")
    graph, err := s.Finalize()
    if err != nil {
        log.Fatalf("Failed to finalize graph: %v", err)
    }
    
    sessionOpts := &tf.SessionOptions{
        Config: []byte(`
            allow_soft_placement: true
            gpu_options { 
                allow_growth: true
                per_process_gpu_memory_fraction: 0.8
            }
        `),
    }
    
    session, err := tf.NewSession(graph, sessionOpts)
    if err != nil {
        log.Fatalf("Failed to create session: %v", err)
    }
    defer session.Close()
    
    results, err := session.Run(nil, []tf.Output{hello}, nil)
    if err != nil {
        log.Fatalf("Failed to run session: %v", err)
    }
    
    log.Println(results[0].Value().(string))
}
```

## Benchmark & Performance Evaluation

To measure the performance gains, we'll implement profiling tools that compare:

1. **Forward Pass Speed**:
   - CPU vs. GPU for single inference
   - Various batch sizes (1, 8, 32, 128, 512)
   - Memory usage patterns

2. **MCTS Performance**:
   - Nodes/second throughput
   - Search depth achievable in fixed time
   - Position evaluation efficiency

3. **NEAT Training**:
   - Time per generation
   - Evaluation throughput
   - Memory usage during evolution

4. **Tournament Efficiency**:
   - Games/second
   - CPU vs. GPU utilization
   - Memory footprint

Benchmarks will be conducted on Apple Silicon M1 with the MPS backend, and results will be documented for comparison with the original CPU implementation.

## Expected Performance

| Component | Current Performance | Expected with GPU | Improvement Factor |
|-----------|---------------------|-------------------|-------------------|
| Forward Pass | ~500-1000 pos/sec | ~50,000-100,000 pos/sec | 40-100x |
| MCTS Search | ~500-1000 nodes/sec | ~5,000-10,000 nodes/sec | 5-15x |
| NEAT Generation | ~30 seconds/gen | ~2-5 seconds/gen | 8-20x |
| Tournament | ~1-2 games/sec | ~5-10 games/sec | 3-8x | 