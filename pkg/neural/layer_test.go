package neural

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayerInitialization(t *testing.T) {
	inputSize := 2
	outputSize := 2
	layer := NewLayer(inputSize, outputSize)

	// Check dimensions
	weights := layer.GetWeights()
	biases := layer.GetBiases()

	if r, c := weights.Dims(); r != outputSize || c != inputSize {
		t.Errorf("Wrong weight dimensions: got (%d, %d), want (%d, %d)", r, c, outputSize, inputSize)
	}

	if r := biases.Len(); r != outputSize {
		t.Errorf("Wrong bias dimensions: got %d, want %d", r, outputSize)
	}

	// Check weight initialization mean
	sum := 0.0
	count := 0
	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			sum += weights.At(i, j)
			count++
		}
	}
	mean := sum / float64(count)

	if math.Abs(mean) > 0.1 {
		t.Errorf("Weight initialization mean too far from zero: got %f", mean)
	}
}

func TestLayerForwardPass(t *testing.T) {
	layer := NewLayer(2, 2)

	input := mat.NewVecDense(2, []float64{1, 2})
	output := layer.Forward(input)

	if output.Len() != 2 {
		t.Errorf("Wrong output size: got %d, want 2", output.Len())
	}
}

func TestLayerBackwardPass(t *testing.T) {
	layer := NewLayer(2, 2)

	// Set known weights and biases
	weights := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	biases := mat.NewVecDense(2, []float64{1, 2})
	layer.weights = weights
	layer.biases = biases

	t.Logf("Initial weights: %v", layer)
	t.Logf("Initial biases: %v", layer.biases)

	input := mat.NewVecDense(2, []float64{1, 2})
	t.Logf("Input: %v", input)

	output := layer.Forward(input)
	t.Logf("Output: %v", output)

	gradients := mat.NewVecDense(2, []float64{1, 2})
	t.Logf("Gradients: %v", gradients)

	inputGrads, biasGrads := layer.Backward(input, output, gradients, 0.1)
	t.Logf("Input gradients: %v", inputGrads)
	t.Logf("Bias gradients: %v", biasGrads)

	newWeights := layer.GetWeights()
	newBiases := layer.GetBiases()

	t.Logf("New weights: %v", newWeights)
	t.Logf("New biases: %v", newBiases)
}

func TestLayerGradientScaling(t *testing.T) {
	layer := NewLayer(2, 2)

	input := mat.NewVecDense(2, []float64{1, 2})
	output := layer.Forward(input)
	gradients := mat.NewVecDense(2, []float64{1, 2})

	// Test different learning rates
	learningRates := []float64{0.001, 0.01, 0.1}
	for _, lr := range learningRates {
		layer.Backward(input, output, gradients, lr)
	}
}

func TestLayerNumericalStability(t *testing.T) {
	layer := NewLayer(2, 2)

	// Test with very small and very large inputs
	smallInput := mat.NewVecDense(2, []float64{1e-10, 1e-10})
	largeInput := mat.NewVecDense(2, []float64{1e10, 1e10})

	smallOutput := layer.Forward(smallInput)
	largeOutput := layer.Forward(largeInput)

	if math.IsNaN(smallOutput.AtVec(0)) || math.IsNaN(smallOutput.AtVec(1)) {
		t.Error("Forward pass produced NaN with small input")
	}

	if math.IsNaN(largeOutput.AtVec(0)) || math.IsNaN(largeOutput.AtVec(1)) {
		t.Error("Forward pass produced NaN with large input")
	}
}
