package neural

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Network represents a neural network with one hidden layer
type Network struct {
	inputSize  int
	hiddenSize int
	outputSize int
	weights1   *mat.Dense // hidden layer weights
	bias1      *mat.VecDense
	weights2   *mat.Dense // output layer weights
	bias2      *mat.VecDense
}

// NewNetwork creates a new neural network with Xavier initialization
func NewNetwork(inputSize, hiddenSize, outputSize int) *Network {
	// Xavier initialization bounds
	w1Bound := math.Sqrt(6.0 / float64(inputSize+hiddenSize))
	w2Bound := math.Sqrt(6.0 / float64(hiddenSize+outputSize))

	// Initialize weights and biases
	weights1 := mat.NewDense(hiddenSize, inputSize, nil)
	bias1 := mat.NewVecDense(hiddenSize, nil)
	weights2 := mat.NewDense(outputSize, hiddenSize, nil)
	bias2 := mat.NewVecDense(outputSize, nil)

	// Random initialization
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < inputSize; j++ {
			weights1.Set(i, j, (rand.Float64()*2-1)*w1Bound)
		}
	}

	for i := 0; i < outputSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			weights2.Set(i, j, (rand.Float64()*2-1)*w2Bound)
		}
	}

	return &Network{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		weights1:   weights1,
		bias1:      bias1,
		weights2:   weights2,
		bias2:      bias2,
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
	// Convert input to matrix
	inputVec := mat.NewVecDense(n.inputSize, input)

	// Hidden layer
	hidden := mat.NewVecDense(n.hiddenSize, nil)
	hidden.MulVec(n.weights1, inputVec)
	hidden.AddVec(hidden, n.bias1)

	// Apply ReLU to hidden layer
	for i := 0; i < n.hiddenSize; i++ {
		hidden.SetVec(i, relu(hidden.AtVec(i)))
	}

	// Output layer
	output := mat.NewVecDense(n.outputSize, nil)
	output.MulVec(n.weights2, hidden)
	output.AddVec(output, n.bias2)

	// Convert output to slice and apply softmax
	outputSlice := make([]float64, n.outputSize)
	for i := 0; i < n.outputSize; i++ {
		outputSlice[i] = output.AtVec(i)
	}
	return softmax(outputSlice)
}

// SaveWeights saves the network weights to a file
func (n *Network) SaveWeights(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Save dimensions
	writeInt(file, n.inputSize)
	writeInt(file, n.hiddenSize)
	writeInt(file, n.outputSize)

	// Save weights and biases
	writeMatrix(file, n.weights1)
	writeVector(file, n.bias1)
	writeMatrix(file, n.weights2)
	writeVector(file, n.bias2)

	return nil
}

// LoadWeights loads the network weights from a file
func (n *Network) LoadWeights(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Load and verify dimensions
	inputSize := readInt(file)
	hiddenSize := readInt(file)
	outputSize := readInt(file)

	if inputSize != n.inputSize || hiddenSize != n.hiddenSize || outputSize != n.outputSize {
		return fmt.Errorf("model architecture mismatch in file: %s", filename)
	}

	// Load weights and biases
	n.weights1 = readMatrix(file, hiddenSize, inputSize)
	n.bias1 = readVector(file, hiddenSize)
	n.weights2 = readMatrix(file, outputSize, hiddenSize)
	n.bias2 = readVector(file, outputSize)

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
