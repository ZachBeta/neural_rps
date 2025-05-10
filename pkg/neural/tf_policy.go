package neural

import (
	"fmt"
	"math/rand/v2"
	"sync"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// RPSTFPolicyNetwork is a policy network implemented using TensorFlow with GPU support
type RPSTFPolicyNetwork struct {
	// Network architecture
	inputSize  int
	hiddenSize int
	outputSize int

	// TensorFlow session and operations
	session  *tf.Session
	graph    *tf.Graph
	inputOp  tf.Output
	outputOp tf.Output

	// Synchronization for concurrent access
	mu sync.Mutex

	// Batch handling
	batchSize     int
	currentBatch  int
	inputBatch    [][]float64
	outputBatch   [][]float64
	batchComplete chan bool
}

// NewRPSTFPolicyNetwork creates a new TensorFlow-based policy network
func NewRPSTFPolicyNetwork(hiddenSize int) *RPSTFPolicyNetwork {
	inputSize := 81 // 9 positions * 9 features
	outputSize := 9 // 9 positions

	// Create TensorFlow graph for the neural network
	graph := tf.NewGraph()

	// Input placeholder
	input, err := tf.Placeholder(tf.Float, tf.MakeShape(-1, int64(inputSize)))
	if err != nil {
		fmt.Printf("Error creating input placeholder: %v\n", err)
		return nil
	}

	// Hidden layer
	wHidden, err := tf.Variable(tf.Const(randomWeights(hiddenSize, inputSize), tf.Float))
	if err != nil {
		fmt.Printf("Error creating hidden weights: %v\n", err)
		return nil
	}

	bHidden, err := tf.Variable(tf.Const(zeroWeights(hiddenSize), tf.Float))
	if err != nil {
		fmt.Printf("Error creating hidden biases: %v\n", err)
		return nil
	}

	hiddenLayer, err := tf.Add(tf.MatMul(input, wHidden), bHidden)
	if err != nil {
		fmt.Printf("Error creating hidden layer: %v\n", err)
		return nil
	}

	hiddenActivation, err := tf.Relu(hiddenLayer)
	if err != nil {
		fmt.Printf("Error applying ReLU: %v\n", err)
		return nil
	}

	// Output layer
	wOutput, err := tf.Variable(tf.Const(randomWeights(outputSize, hiddenSize), tf.Float))
	if err != nil {
		fmt.Printf("Error creating output weights: %v\n", err)
		return nil
	}

	bOutput, err := tf.Variable(tf.Const(zeroWeights(outputSize), tf.Float))
	if err != nil {
		fmt.Printf("Error creating output biases: %v\n", err)
		return nil
	}

	logits, err := tf.Add(tf.MatMul(hiddenActivation, wOutput), bOutput)
	if err != nil {
		fmt.Printf("Error creating output layer: %v\n", err)
		return nil
	}

	// Apply softmax
	output, err := tf.Softmax(logits)
	if err != nil {
		fmt.Printf("Error applying softmax: %v\n", err)
		return nil
	}

	// Create TensorFlow session
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Printf("Error creating TensorFlow session: %v\n", err)
		return nil
	}

	// Initialize variables
	init, err := tf.NewOperation(graph, "global_variables_initializer", "init")
	if err != nil {
		fmt.Printf("Error creating initializer: %v\n", err)
		return nil
	}

	if _, err := sess.Run(nil, nil, []*tf.Operation{init}); err != nil {
		fmt.Printf("Error initializing variables: %v\n", err)
		return nil
	}

	// Create network
	return &RPSTFPolicyNetwork{
		inputSize:     inputSize,
		hiddenSize:    hiddenSize,
		outputSize:    outputSize,
		session:       sess,
		graph:         graph,
		inputOp:       input,
		outputOp:      output,
		batchSize:     64, // Default batch size
		inputBatch:    make([][]float64, 0, 64),
		outputBatch:   make([][]float64, 0, 64),
		batchComplete: make(chan bool),
	}
}

// Predict returns the position probabilities for a given game state
func (n *RPSTFPolicyNetwork) Predict(gameState *game.RPSGame) []float64 {
	// Convert game state to input features
	input := gameState.GetBoardAsFeatures()

	// Use TensorFlow session to run inference
	n.mu.Lock()
	defer n.mu.Unlock()

	inputTensor, err := tf.NewTensor([][]float64{input})
	if err != nil {
		fmt.Printf("Error creating input tensor: %v\n", err)
		return make([]float64, n.outputSize)
	}

	// Run inference
	result, err := n.session.Run(
		map[tf.Output]*tf.Tensor{
			n.inputOp: inputTensor,
		},
		[]tf.Output{n.outputOp},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running inference: %v\n", err)
		return make([]float64, n.outputSize)
	}

	// Extract and return results
	outputArray := result[0].Value().([][]float64)
	return outputArray[0]
}

// PredictBatch performs prediction on a batch of game states
func (n *RPSTFPolicyNetwork) PredictBatch(inputs [][]float64) [][]float64 {
	n.mu.Lock()
	defer n.mu.Unlock()

	inputTensor, err := tf.NewTensor(inputs)
	if err != nil {
		fmt.Printf("Error creating batch input tensor: %v\n", err)
		return make([][]float64, len(inputs))
	}

	// Run inference
	result, err := n.session.Run(
		map[tf.Output]*tf.Tensor{
			n.inputOp: inputTensor,
		},
		[]tf.Output{n.outputOp},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running batch inference: %v\n", err)
		return make([][]float64, len(inputs))
	}

	// Extract and return results
	return result[0].Value().([][]float64)
}

// PredictMove returns the best move according to the policy network
func (n *RPSTFPolicyNetwork) PredictMove(gameState *game.RPSGame) game.RPSMove {
	// Get valid moves
	validMoves := gameState.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{} // No valid moves
	}

	// Get position probabilities
	positionProbs := n.Predict(gameState)

	// Group moves by position
	movesByPosition := make(map[int][]game.RPSMove)
	for _, move := range validMoves {
		movesByPosition[move.Position] = append(movesByPosition[move.Position], move)
	}

	// Find the best position according to the policy network
	bestPosition := 0
	bestProb := positionProbs[0]
	for pos, prob := range positionProbs {
		if prob > bestProb && len(movesByPosition[pos]) > 0 {
			bestProb = prob
			bestPosition = pos
		}
	}

	// Choose the first move that places a card at the best position
	possibleMoves := movesByPosition[bestPosition]
	if len(possibleMoves) > 0 {
		return possibleMoves[0]
	}

	// Fallback: return the first valid move
	return validMoves[0]
}

// SaveToFile saves the network weights to a file
func (n *RPSTFPolicyNetwork) SaveToFile(filename string) error {
	// Not implemented yet
	return fmt.Errorf("SaveToFile not implemented for TensorFlow network")
}

// LoadFromFile loads the network weights from a file
func (n *RPSTFPolicyNetwork) LoadFromFile(filename string) error {
	// Not implemented yet
	return fmt.Errorf("LoadFromFile not implemented for TensorFlow network")
}

// Helper functions for weight initialization
func randomWeights(rows, cols int) [][][]float64 {
	weights := make([][][]float64, 1)
	weights[0] = make([][]float64, rows)

	for i := range weights[0] {
		weights[0][i] = make([]float64, cols)
		for j := range weights[0][i] {
			weights[0][i][j] = (rand.Float64()*2 - 1) * 0.1
		}
	}

	return weights
}

func zeroWeights(size int) [][][]float64 {
	weights := make([][][]float64, 1)
	weights[0] = make([][]float64, size)

	for i := range weights[0] {
		weights[0][i] = make([]float64, 1)
		weights[0][i][0] = 0.0
	}

	return weights
}
