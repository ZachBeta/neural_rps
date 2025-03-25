package neural

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestSimpleBackprop creates a minimal working example of backpropagation
func TestSimpleBackprop(t *testing.T) {
	// Create a simple layer with 2 inputs and 1 output
	layer := NewLayer(2, 1)

	// Set known weights and biases
	weights := mat.NewDense(1, 2, []float64{1.0, 2.0})
	biases := mat.NewVecDense(1, []float64{1.0})
	layer.weights = weights
	layer.biases = biases

	// Create test input
	input := mat.NewVecDense(2, []float64{1.0, 2.0})
	t.Logf("Input: %v", input)
	t.Logf("Weights: %v", layer)
	t.Logf("Biases: %v", layer.biases)

	// Forward pass
	output := layer.Forward(input)
	t.Logf("Output: %v", output)

	// Expected output:
	// z = 1.0 * 1.0 + 2.0 * 2.0 + 1.0 = 6.0
	// ReLU(6.0) = 6.0
	if math.Abs(output.AtVec(0)-6.0) > 1e-6 {
		t.Errorf("Forward pass output incorrect: got %f, want 6.0", output.AtVec(0))
	}

	// Compute gradients using MSE loss
	target := 7.0
	t.Logf("Target: %f", target)
	gradient := 2.0 * (output.AtVec(0) - target)
	t.Logf("Gradient: %f", gradient)

	// Backward pass
	inputGrads, biasGrads := layer.Backward(input, output, mat.NewVecDense(1, []float64{gradient}), 0.1)
	t.Logf("Input gradients: %v", inputGrads)
	t.Logf("Bias gradients: %v", biasGrads)

	// Verify weight updates
	expectedWeightGrad1 := 2.0 * (output.AtVec(0) - target) * input.AtVec(0) // 2 * (6 - 7) * 1 = -2
	expectedWeightGrad2 := 2.0 * (output.AtVec(0) - target) * input.AtVec(1) // 2 * (6 - 7) * 2 = -4
	expectedBiasGrad := 2.0 * (output.AtVec(0) - target)                     // 2 * (6 - 7) = -2

	// Check weight gradients
	if math.Abs(weights.At(0, 0)-1.0+0.1*expectedWeightGrad1) > 1e-6 {
		t.Errorf("Weight 1 update incorrect: got %f, want %f", weights.At(0, 0), 1.0-0.1*expectedWeightGrad1)
	}
	if math.Abs(weights.At(0, 1)-2.0+0.1*expectedWeightGrad2) > 1e-6 {
		t.Errorf("Weight 2 update incorrect: got %f, want %f", weights.At(0, 1), 2.0-0.1*expectedWeightGrad2)
	}
	if math.Abs(biases.AtVec(0)-1.0+0.1*expectedBiasGrad) > 1e-6 {
		t.Errorf("Bias update incorrect: got %f, want %f", biases.AtVec(0), 1.0-0.1*expectedBiasGrad)
	}

	// Check input gradients
	expectedInputGrad1 := 2.0 * (output.AtVec(0) - target) * weights.At(0, 0) // 2 * (6 - 7) * 1 = -2
	expectedInputGrad2 := 2.0 * (output.AtVec(0) - target) * weights.At(0, 1) // 2 * (6 - 7) * 2 = -4

	if math.Abs(inputGrads.AtVec(0)-expectedInputGrad1) > 1e-6 {
		t.Errorf("Input gradient 1 incorrect: got %f, want %f", inputGrads.AtVec(0), expectedInputGrad1)
	}
	if math.Abs(inputGrads.AtVec(1)-expectedInputGrad2) > 1e-6 {
		t.Errorf("Input gradient 2 incorrect: got %f, want %f", inputGrads.AtVec(1), expectedInputGrad2)
	}
}
