package neural

import (
	"math"
	"testing"
)

func TestNetworkInitialization(t *testing.T) {
	nn := NewNetwork(9, 64, 3)

	// Test layer sizes
	r1, c1 := nn.weights1.Dims()
	if r1 != 64 || c1 != 9 {
		t.Errorf("weights1 dimensions = %dx%d, want 64x9", r1, c1)
	}
	r2, c2 := nn.weights2.Dims()
	if r2 != 3 || c2 != 64 {
		t.Errorf("weights2 dimensions = %dx%d, want 3x64", r2, c2)
	}

	// Test weight initialization (should be small random values)
	maxWeight := 0.0
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			w := math.Abs(nn.weights1.At(i, j))
			if w > maxWeight {
				maxWeight = w
			}
		}
	}
	if maxWeight > 1.0 {
		t.Errorf("weights1 maximum absolute value = %f, want <= 1.0", maxWeight)
	}
}

func TestForwardPass(t *testing.T) {
	nn := NewNetwork(9, 64, 3)
	input := make([]float64, 9)
	input[0] = 1.0 // Set one input to 1, rest to 0

	// Test forward pass
	output := nn.Forward(input)

	// Check output dimensions
	if len(output) != 3 {
		t.Errorf("output length = %d, want 3", len(output))
	}

	// Check output is valid probability distribution
	sum := 0.0
	for _, v := range output {
		if v < 0 || v > 1 {
			t.Errorf("output value %f is not between 0 and 1", v)
		}
		sum += v
	}
	if math.Abs(sum-1.0) > 0.0001 {
		t.Errorf("output probabilities sum = %f, want 1.0", sum)
	}
}

func TestBackwardPass(t *testing.T) {
	nn := NewNetwork(9, 64, 3)
	input := make([]float64, 9)
	input[0] = 1.0

	// Store initial weights
	initialWeights1 := nn.weights1.RawMatrix().Data
	initialWeights2 := nn.weights2.RawMatrix().Data

	// Perform backward pass
	output := nn.Forward(input)
	gradients := make([]float64, 3)
	gradients[0] = 1.0 // Target action 0
	nn.Backward(input, output, gradients, 0.01)

	// Check that weights were updated
	weightsChanged := false
	for i, w := range nn.weights1.RawMatrix().Data {
		if w != initialWeights1[i] {
			weightsChanged = true
			break
		}
	}
	if !weightsChanged {
		t.Error("weights1 were not updated during backward pass")
	}

	weightsChanged = false
	for i, w := range nn.weights2.RawMatrix().Data {
		if w != initialWeights2[i] {
			weightsChanged = true
			break
		}
	}
	if !weightsChanged {
		t.Error("weights2 were not updated during backward pass")
	}
}

func TestNumericalStability(t *testing.T) {
	nn := NewNetwork(9, 64, 3)

	// Test with extreme input values
	inputs := [][]float64{
		make([]float64, 9), // All zeros
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, // All ones
		{100.0, 0, 0, 0, 0, 0, 0, 0, 0},               // Extreme value
	}

	for _, input := range inputs {
		output := nn.Forward(input)

		// Check for NaN or Inf values
		for i, v := range output {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("output[%d] is NaN or Inf for input %v", i, input)
			}
		}

		// Check output is valid probability distribution
		sum := 0.0
		for _, v := range output {
			sum += v
		}
		if math.Abs(sum-1.0) > 0.0001 {
			t.Errorf("output probabilities sum = %f for input %v, want 1.0", sum, input)
		}
	}
}

func TestLearningBehavior(t *testing.T) {
	nn := NewNetwork(9, 64, 3)
	input := make([]float64, 9)
	input[0] = 1.0

	// Test multiple forward and backward passes
	initialOutput := nn.Forward(input)
	for i := 0; i < 100; i++ {
		output := nn.Forward(input)
		gradients := make([]float64, 3)
		gradients[0] = 1.0 // Always target action 0
		nn.Backward(input, output, gradients, 0.01)
	}
	finalOutput := nn.Forward(input)

	// Check that the network learned to prefer action 0
	if finalOutput[0] <= initialOutput[0] {
		t.Errorf("network did not learn to prefer action 0: initial=%f, final=%f",
			initialOutput[0], finalOutput[0])
	}
}

func TestGradientScaling(t *testing.T) {
	nn := NewNetwork(9, 64, 3)
	input := make([]float64, 9)
	input[0] = 1.0

	// Test different learning rates
	learningRates := []float64{0.001, 0.01, 0.1}
	for _, lr := range learningRates {
		// Reset network
		nn = NewNetwork(9, 64, 3)

		// Store initial weights
		initialWeights := make([]float64, len(nn.weights1.RawMatrix().Data))
		copy(initialWeights, nn.weights1.RawMatrix().Data)

		// Perform backward pass
		output := nn.Forward(input)
		gradients := make([]float64, 3)
		gradients[0] = 1.0
		nn.Backward(input, output, gradients, lr)

		// Calculate weight changes
		maxChange := 0.0
		for i, w := range nn.weights1.RawMatrix().Data {
			change := math.Abs(w - initialWeights[i])
			if change > maxChange {
				maxChange = change
			}
		}

		// Check that weight changes scale with learning rate
		expectedScale := lr / 0.01 // Relative to baseline learning rate
		if maxChange < expectedScale*0.01 {
			t.Errorf("weight changes do not scale with learning rate %f: got %f, want > %f",
				lr, maxChange, expectedScale*0.01)
		}
	}
}
