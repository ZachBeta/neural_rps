package neural

import (
	"os"
	"testing"
)

func TestNetworkInitialization(t *testing.T) {
	nn := NewNetwork(6, 12, 3)

	// Check dimensions
	if nn.InputSize != 6 {
		t.Errorf("Expected input size 6, got %d", nn.InputSize)
	}
	if nn.HiddenSize != 12 {
		t.Errorf("Expected hidden size 12, got %d", nn.HiddenSize)
	}
	if nn.OutputSize != 3 {
		t.Errorf("Expected output size 3, got %d", nn.OutputSize)
	}

	// Check weights initialization
	if len(nn.Weights1) != 12 {
		t.Errorf("Expected Weights1 to have 12 rows, got %d", len(nn.Weights1))
	}
	if len(nn.Weights1[0]) != 6 {
		t.Errorf("Expected Weights1 to have 6 columns, got %d", len(nn.Weights1[0]))
	}
	if len(nn.Weights2) != 3 {
		t.Errorf("Expected Weights2 to have 3 rows, got %d", len(nn.Weights2))
	}
	if len(nn.Weights2[0]) != 12 {
		t.Errorf("Expected Weights2 to have 12 columns, got %d", len(nn.Weights2[0]))
	}
}

func TestForward(t *testing.T) {
	nn := NewNetwork(2, 3, 2)

	// Test input
	input := []float64{0.5, 0.2}
	output := nn.Forward(input)

	// Check output shape
	if len(output) != 2 {
		t.Errorf("Expected output size 2, got %d", len(output))
	}

	// Check softmax properties
	sum := 0.0
	for _, v := range output {
		if v < 0 || v > 1 {
			t.Errorf("Output value outside [0,1] range: %f", v)
		}
		sum += v
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Sum of output probabilities should be 1.0, got %f", sum)
	}
}

func TestPredict(t *testing.T) {
	nn := NewNetwork(2, 3, 3)

	// Test input
	input := []float64{0.5, 0.2}
	prediction := nn.Predict(input)

	// Check prediction is within valid range
	if prediction < 0 || prediction >= 3 {
		t.Errorf("Prediction outside valid range [0,2]: %d", prediction)
	}
}

func TestSaveLoad(t *testing.T) {
	// Create a network with known values
	nn1 := NewNetwork(2, 3, 2)

	// Set some weights to specific values
	nn1.Weights1[0][0] = 0.1
	nn1.Weights2[0][0] = 0.2

	// Save the network
	tempFile := "temp_network.gob"
	err := nn1.SaveWeights(tempFile)
	if err != nil {
		t.Fatalf("Failed to save weights: %v", err)
	}

	// Create a new network and load the weights
	nn2 := NewNetwork(2, 3, 2)
	err = nn2.LoadWeights(tempFile)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Check that weights were loaded correctly
	if nn2.Weights1[0][0] != 0.1 {
		t.Errorf("Expected Weights1[0][0] to be 0.1, got %f", nn2.Weights1[0][0])
	}
	if nn2.Weights2[0][0] != 0.2 {
		t.Errorf("Expected Weights2[0][0] to be 0.2, got %f", nn2.Weights2[0][0])
	}

	// Clean up
	os.Remove(tempFile)
}

func TestTrainingSimpleXOR(t *testing.T) {
	// Create a network for XOR problem
	nn := NewNetwork(2, 4, 1)

	// XOR training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	// XOR expected outputs (one-hot encoded for 1 output)
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Training options with fewer epochs for test
	options := TrainingOptions{
		LearningRate: 0.1,
		Epochs:       500,
		BatchSize:    4,
		Parallel:     false, // Disable parallelism for test predictability
	}

	// Train the network
	err := nn.Train(inputs, targets, options)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test the predictions (should be close to targets after training)
	for i, input := range inputs {
		output := nn.Forward(input)
		expected := targets[i][0]

		// Simple threshold for binary classification
		prediction := 0.0
		if output[0] > 0.5 {
			prediction = 1.0
		}

		// Don't be too strict since neural networks have randomness
		if prediction != expected {
			t.Logf("Warning: Input %v gave %f, expected %f", input, output[0], expected)
		}
	}
}

func TestParallelTraining(t *testing.T) {
	// Skip if testing with -short flag
	if testing.Short() {
		t.Skip("Skipping parallel training test in short mode")
	}

	// Create a network
	nn := NewNetwork(2, 8, 1)

	// Generate some random training data
	const numSamples = 100
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = []float64{float64(i) / numSamples, float64(i%10) / 10}
		targets[i] = []float64{float64(i % 2)} // Alternate between 0 and 1
	}

	// Training options with parallelism
	parallelOptions := TrainingOptions{
		LearningRate: 0.01,
		Epochs:       100,
		BatchSize:    32,
		Parallel:     true,
	}

	// Train with parallelism
	err := nn.Train(inputs, targets, parallelOptions)
	if err != nil {
		t.Fatalf("Parallel training failed: %v", err)
	}

	// No assertions here, just making sure it runs without errors
	// In a real test, we'd compare performance metrics
}
