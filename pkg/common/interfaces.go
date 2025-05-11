package common

// NeuralNetwork defines the interface for all neural network implementations
type NeuralNetwork interface {
	// Forward runs a forward pass through the network
	Forward(input []float64) ([]float64, error)

	// GetInputSize returns the input size of the network
	GetInputSize() int

	// GetOutputSize returns the output size of the network
	GetOutputSize() int

	// Close releases any resources used by the network
	Close() error
}

// BatchedNeuralNetwork extends the NeuralNetwork interface with batch processing
type BatchedNeuralNetwork interface {
	NeuralNetwork

	// ForwardBatch runs a forward pass for a batch of inputs
	ForwardBatch(inputs [][]float64) ([][]float64, error)

	// PredictBatch returns the index of the highest output value for a batch of inputs
	PredictBatch(inputs [][]float64) ([]int, error)
}

// NetworkStats contains performance statistics for neural network execution
type NetworkStats struct {
	// TotalCalls is the number of network calls (single or batch)
	TotalCalls int

	// TotalBatchSize is the total number of positions evaluated
	TotalBatchSize int

	// AvgLatencyUs is the average latency per call in microseconds
	AvgLatencyUs float64

	// AvgBatchSize is the average batch size per call
	AvgBatchSize float64
}

// Agent defines the interface for all game-playing agents
type Agent interface {
	// GetMove returns the best move for the current game state
	GetMove(gameState interface{}) (interface{}, error)

	// Name returns the name of the agent
	Name() string
}

// TreeSearchAgent defines the interface for tree-search based agents
type TreeSearchAgent interface {
	Agent

	// SetSearchDepth sets the search depth/simulations
	SetSearchDepth(depth int)

	// SetExplorationConstant sets the exploration constant
	SetExplorationConstant(c float64)
}

// Option defines a functional option pattern for configuring components
type Option func(interface{}) error
