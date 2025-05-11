package neural

import (
	"fmt"
	"math"
	"math/rand"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	op "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// RPSTFPolicyNetwork implements a policy network using TensorFlow
type RPSTFPolicyNetwork struct {
	session    *tf.Session
	graph      *tf.Graph
	inputOp    tf.Output
	outputOp   tf.Output
	inputShape []int64

	// Keep track of network dimensions (for compatibility with the original Network)
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
	w1 := op.Variable(s, op.Const(s.SubScope("w1"), generateWeights(inputSize, hiddenSize)))
	b1 := op.Variable(s, op.Const(s.SubScope("b1"), generateBiases(hiddenSize)))
	hidden := op.Relu(s, op.Add(s, op.MatMul(s, input, w1), b1))

	// Output layer with softmax activation
	w2 := op.Variable(s, op.Const(s.SubScope("w2"), generateWeights(hiddenSize, outputSize)))
	b2 := op.Variable(s, op.Const(s.SubScope("b2"), generateBiases(outputSize)))
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

// Predict predicts the best move based on the input (returns index of highest probability)
func (nn *RPSTFPolicyNetwork) Predict(input []float64) (int, error) {
	output, err := nn.Forward(input)
	if err != nil {
		return -1, err
	}
	return argmax(output), nil
}

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

// Helper functions for network initialization
func generateWeights(inputSize, outputSize int) [][]float32 {
	// Xavier/Glorot initialization
	limit := float32(math.Sqrt(6.0 / float64(inputSize+outputSize)))
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

// Close releases the TensorFlow session
func (nn *RPSTFPolicyNetwork) Close() {
	if nn.session != nil {
		nn.session.Close()
	}
}

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

	// Create assign operations directly (avoids unused variables)
	s := op.NewScope()

	// Get variables from graph and create assign ops
	w1Var := nn.graph.Operation("w1")
	w1AssignOp := op.Assign(s, tf.Output{Op: w1Var, Index: 0}, op.Const(s, weights1))

	b1Var := nn.graph.Operation("b1")
	b1AssignOp := op.Assign(s, tf.Output{Op: b1Var, Index: 0}, op.Const(s, bias1))

	w2Var := nn.graph.Operation("w2")
	w2AssignOp := op.Assign(s, tf.Output{Op: w2Var, Index: 0}, op.Const(s, weights2))

	b2Var := nn.graph.Operation("b2")
	b2AssignOp := op.Assign(s, tf.Output{Op: b2Var, Index: 0}, op.Const(s, bias2))

	// Run assign operations
	assignOps := []*tf.Operation{w1AssignOp.Op, b1AssignOp.Op, w2AssignOp.Op, b2AssignOp.Op}
	_, err := nn.session.Run(nil, nil, assignOps)
	if err != nil {
		return fmt.Errorf("error assigning weights: %v", err)
	}

	return nil
}

// SaveModel saves the TensorFlow model to a file
func (nn *RPSTFPolicyNetwork) SaveModel(path string) error {
	// Create a TensorFlow SavedModel
	s := op.NewScope()

	// Define input and output signature
	input := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, int64(nn.InputSize))))

	// Build the signature definition
	signatureInputs := map[string]tf.Output{
		"input": input,
	}
	signatureOutputs := map[string]tf.Output{
		"output": nn.outputOp,
	}

	// Create a SavedModel builder
	builder := tf.NewSavedModelBuilder(path)
	builder.AddSignature("serve", signatureInputs, signatureOutputs)

	// Save the model
	return builder.Save(nn.session)
}

// LoadModel loads a TensorFlow model from a file
func LoadRPSTFPolicyNetwork(path string) (*RPSTFPolicyNetwork, error) {
	// Load the model
	bundle, err := tf.LoadSavedModel(path, []string{"serve"}, nil)
	if err != nil {
		return nil, fmt.Errorf("error loading model: %v", err)
	}

	// Extract input and output operations
	inputOp := tf.Output{Op: bundle.Graph.Operation("input"), Index: 0}
	outputOp := tf.Output{Op: bundle.Graph.Operation("output"), Index: 0}

	// Determine input size from the shape
	shape := inputOp.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("unexpected input shape: %v", shape)
	}

	inputSize := int(shape[1])

	// Create the network
	return &RPSTFPolicyNetwork{
		session:    bundle.Session,
		graph:      bundle.Graph,
		inputOp:    inputOp,
		outputOp:   outputOp,
		inputShape: shape,
		InputSize:  inputSize,
		// Note: Hidden size can't be determined from the loaded model
		HiddenSize: 0,
		// Output size could be determined similarly to input size
		OutputSize: 0,
	}, nil
}
