package neural

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Layer represents a neural network layer
type Layer struct {
	weights    *mat.Dense
	biases     *mat.VecDense
	activation Activation
	inputSize  int
	outputSize int
}

// Activation represents a neural network activation function
type Activation interface {
	Forward(x float64) float64
	Backward(x float64) float64
}

// ReLU implements the ReLU activation function
type ReLU struct{}

func (r *ReLU) Forward(x float64) float64 {
	return relu(x)
}

func (r *ReLU) Backward(x float64) float64 {
	return reluDerivative(x)
}

// NewLayer creates a new layer with the specified input and output sizes
func NewLayer(inputSize, outputSize int) *Layer {
	// Initialize weights using He initialization
	weights := mat.NewDense(outputSize, inputSize, nil)
	scale := math.Sqrt(2.0 / float64(inputSize))
	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			// Generate two random numbers and use them to create a zero-mean pair
			r1 := rand.NormFloat64()
			r2 := -r1
			// Use one of them randomly
			if rand.Float64() < 0.5 {
				weights.Set(i, j, r1*scale)
			} else {
				weights.Set(i, j, r2*scale)
			}
		}
	}

	// Initialize biases to zero
	biases := mat.NewVecDense(outputSize, nil)

	return &Layer{
		weights:    weights,
		biases:     biases,
		inputSize:  inputSize,
		outputSize: outputSize,
	}
}

// Forward performs a forward pass through the layer
func (l *Layer) Forward(input *mat.VecDense) *mat.VecDense {
	// Compute z = Wx + b
	z := mat.NewVecDense(l.outputSize, nil)
	z.MulVec(l.weights, input)
	z.AddVec(z, l.biases)

	// Apply activation function
	output := mat.NewVecDense(l.outputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		output.SetVec(i, l.activation.Forward(z.AtVec(i)))
	}

	return output
}

// Backward performs backpropagation through the layer
func (l *Layer) Backward(input *mat.VecDense, output *mat.VecDense, gradients *mat.VecDense, learningRate float64) (*mat.VecDense, *mat.VecDense) {
	// Compute input gradients
	inputGrads := mat.NewVecDense(l.inputSize, nil)
	inputGrads.MulVec(l.weights.T(), gradients)

	// Compute weight gradients
	weightGrads := mat.NewDense(l.outputSize, l.inputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightGrads.Set(i, j, gradients.AtVec(i)*input.AtVec(j))
		}
	}

	// Update weights and biases
	scaledGrads := mat.NewDense(l.outputSize, l.inputSize, nil)
	scaledGrads.Scale(learningRate, weightGrads)
	l.weights.Add(l.weights, scaledGrads)
	l.biases.AddScaledVec(l.biases, learningRate, gradients)

	return inputGrads, gradients
}

// GetWeights returns the layer's weight matrix
func (l *Layer) GetWeights() *mat.Dense {
	return l.weights
}

// GetBiases returns the layer's bias vector
func (l *Layer) GetBiases() *mat.VecDense {
	return l.biases
}
