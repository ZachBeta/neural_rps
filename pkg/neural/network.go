package neural

import (
	"encoding/binary"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Network represents a neural network
type Network struct {
	inputSize   int
	hiddenSize  int
	outputSize  int
	hiddenLayer *Layer
	outputLayer *Layer
}

// NewNetwork creates a new neural network
func NewNetwork(inputSize, hiddenSize, outputSize int) *Network {
	return &Network{
		inputSize:   inputSize,
		hiddenSize:  hiddenSize,
		outputSize:  outputSize,
		hiddenLayer: NewLayer(inputSize, hiddenSize),
		outputLayer: NewLayer(hiddenSize, outputSize),
	}
}

// relu implements the ReLU activation function
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDerivative implements the derivative of ReLU
func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// softmax implements the softmax activation function
func softmax(x []float64) []float64 {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := 0.0
	exp := make([]float64, len(x))
	for i, v := range x {
		exp[i] = math.Exp(v - max)
		sum += exp[i]
	}

	for i := range exp {
		exp[i] /= sum
	}
	return exp
}

// Forward performs a forward pass through the network
func (n *Network) Forward(input []float64) []float64 {
	// Convert input to vector
	inputVec := mat.NewVecDense(len(input), input)

	// Forward pass through hidden layer
	hiddenOutput := n.hiddenLayer.Forward(inputVec)

	// Forward pass through output layer
	output := n.outputLayer.Forward(hiddenOutput)

	// Convert output to slice
	outputSlice := make([]float64, n.outputSize)
	for i := 0; i < n.outputSize; i++ {
		outputSlice[i] = output.AtVec(i)
	}

	return outputSlice
}

// Backward performs backpropagation through the network
func (n *Network) Backward(input []float64, output []float64, gradients []float64, learningRate float64) {
	// Convert input to vector format
	inputVec := mat.NewVecDense(len(input), input)
	outputVec := mat.NewVecDense(len(output), output)
	gradientVec := mat.NewVecDense(len(gradients), gradients)

	// Forward pass to compute hidden layer activations
	hiddenOutput := n.hiddenLayer.Forward(inputVec)

	// Output layer gradients
	outputGrads := mat.NewVecDense(n.outputSize, nil)
	outputGrads.SubVec(outputVec, gradientVec)
	outputGrads.ScaleVec(-1, outputGrads)

	// Backward pass through output layer
	hiddenGrads, _ := n.outputLayer.Backward(hiddenOutput, outputVec, outputGrads, learningRate)

	// Apply ReLU derivative to hidden gradients
	for i := 0; i < n.hiddenSize; i++ {
		if hiddenOutput.AtVec(i) <= 0 {
			hiddenGrads.SetVec(i, 0)
		}
	}

	// Backward pass through hidden layer
	_, _ = n.hiddenLayer.Backward(inputVec, hiddenOutput, hiddenGrads, learningRate)
}

// dotProduct computes the dot product of two slices
func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// SaveWeights saves the network weights to a file
func (n *Network) SaveWeights(filename string) error {
	// TODO: Implement weight saving
	return nil
}

// LoadWeights loads the network weights from a file
func (n *Network) LoadWeights(filename string) error {
	// TODO: Implement weight loading
	return nil
}

// Helper functions for file I/O
func writeInt(file *os.File, value int) error {
	return binary.Write(file, binary.LittleEndian, int32(value))
}

func readInt(file *os.File) int {
	var value int32
	binary.Read(file, binary.LittleEndian, &value)
	return int(value)
}

func writeMatrix(file *os.File, m mat.Matrix) error {
	r, c := m.Dims()
	writeInt(file, r)
	writeInt(file, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if err := binary.Write(file, binary.LittleEndian, m.At(i, j)); err != nil {
				return err
			}
		}
	}
	return nil
}

func readMatrix(file *os.File, rows, cols int) *mat.Dense {
	m := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var value float64
			binary.Read(file, binary.LittleEndian, &value)
			m.Set(i, j, value)
		}
	}
	return m
}

func writeVector(file *os.File, v mat.Vector) error {
	writeInt(file, v.Len())
	for i := 0; i < v.Len(); i++ {
		if err := binary.Write(file, binary.LittleEndian, v.AtVec(i)); err != nil {
			return err
		}
	}
	return nil
}

func readVector(file *os.File, length int) *mat.VecDense {
	v := mat.NewVecDense(length, nil)
	for i := 0; i < length; i++ {
		var value float64
		binary.Read(file, binary.LittleEndian, &value)
		v.SetVec(i, value)
	}
	return v
}
