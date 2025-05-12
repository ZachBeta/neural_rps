package neural

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func TestNewRPSValueNetwork(t *testing.T) {
	network := NewRPSValueNetwork(64)

	// Check network structure
	if network.inputSize != 81 {
		t.Errorf("Expected input size 81, got %d", network.inputSize)
	}

	if network.hiddenSize != 64 {
		t.Errorf("Expected hidden size 64, got %d", network.hiddenSize)
	}

	if network.outputSize != 1 {
		t.Errorf("Expected output size 1, got %d", network.outputSize)
	}

	// Check that weights and biases are initialized
	if len(network.weightsInputHidden) != 64 {
		t.Errorf("Expected 64 input-hidden weight vectors, got %d", len(network.weightsInputHidden))
	}

	if len(network.biasesHidden) != 64 {
		t.Errorf("Expected 64 hidden biases, got %d", len(network.biasesHidden))
	}

	if len(network.weightsHiddenOutput) != 1 {
		t.Errorf("Expected 1 hidden-output weight vector, got %d", len(network.weightsHiddenOutput))
	}

	if len(network.biasesOutput) != 1 {
		t.Errorf("Expected 1 output bias, got %d", len(network.biasesOutput))
	}
}

func TestRPSValuePredict(t *testing.T) {
	network := NewRPSValueNetwork(64)
	gameState := game.NewRPSGame(21, 5, 10)

	// Prediction should be between 0 and 1 (sigmoid output)
	value := network.Predict(gameState)
	if value < 0 || value > 1 {
		t.Errorf("Value prediction outside [0,1] range: %f", value)
	}
}

func TestRPSValueTrain(t *testing.T) {
	network := NewRPSValueNetwork(64)

	// Create some training data
	inputFeatures := make([][]float64, 10)
	targetValues := make([]float64, 10)

	for i := 0; i < 10; i++ {
		inputFeatures[i] = make([]float64, 81)
		// Fill with random values
		for j := 0; j < 81; j++ {
			inputFeatures[i][j] = rand.Float64()*2 - 1
		}
		// Target between 0 and 1
		targetValues[i] = rand.Float64()
	}

	// Initial prediction
	initialPrediction := network.forward(inputFeatures[0])

	// Train for a few epochs
	loss := network.Train(inputFeatures, targetValues, 0.1)

	// Check that loss is reasonable
	if loss < 0 {
		t.Errorf("Loss should be non-negative, got %f", loss)
	}

	// Check that network weights have changed
	finalPrediction := network.forward(inputFeatures[0])
	if initialPrediction == finalPrediction {
		t.Errorf("Network prediction unchanged after training")
	}
}

func TestRPSValueSaveLoadFile(t *testing.T) {
	// Create a temporary file
	tmpfile, err := os.CreateTemp("", "value_test_*.model")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	tmpPath := tmpfile.Name()
	tmpfile.Close()
	defer os.Remove(tmpPath) // Clean up

	// Create two networks: one to save, one to load
	originalNetwork := NewRPSValueNetwork(64)
	loadedNetwork := NewRPSValueNetwork(32) // Intentionally different size

	// Generate some sample data and train the original network a bit
	// to make the weights different from random initialization
	inputFeatures := make([][]float64, 10)
	targetValues := make([]float64, 10)

	for i := 0; i < 10; i++ {
		inputFeatures[i] = make([]float64, 81)
		// Fill with random data
		for j := 0; j < 81; j++ {
			inputFeatures[i][j] = rand.Float64()
		}
		// Make target a value between 0 and 1
		targetValues[i] = rand.Float64()
	}

	// Train the original network a bit
	originalNetwork.Train(inputFeatures, targetValues, 0.01)

	// Predict a value with original network
	testInput := make([]float64, 81)
	for i := 0; i < 81; i++ {
		testInput[i] = rand.Float64()
	}
	originalPrediction := originalNetwork.forward(testInput)

	// Save the original network
	err = originalNetwork.SaveToFile(tmpPath)
	if err != nil {
		t.Fatalf("Failed to save network: %v", err)
	}

	// Load into the other network
	err = loadedNetwork.LoadFromFile(tmpPath)
	if err != nil {
		t.Fatalf("Failed to load network: %v", err)
	}

	// Check that the networks have the same parameters now
	if loadedNetwork.inputSize != originalNetwork.inputSize {
		t.Errorf("Input size mismatch: got %d, want %d", loadedNetwork.inputSize, originalNetwork.inputSize)
	}

	if loadedNetwork.hiddenSize != originalNetwork.hiddenSize {
		t.Errorf("Hidden size mismatch: got %d, want %d", loadedNetwork.hiddenSize, originalNetwork.hiddenSize)
	}

	if loadedNetwork.outputSize != originalNetwork.outputSize {
		t.Errorf("Output size mismatch: got %d, want %d", loadedNetwork.outputSize, originalNetwork.outputSize)
	}

	// Check that the loaded network produces the same prediction
	loadedPrediction := loadedNetwork.forward(testInput)

	// Compare predictions (allow for small floating-point differences)
	if math.Abs(originalPrediction-loadedPrediction) > 1e-6 {
		t.Errorf("Prediction mismatch: got %f, want %f", loadedPrediction, originalPrediction)
	}
}

func TestRPSValueSigmoid(t *testing.T) {
	// Test sigmoid functionality
	testCases := []struct {
		input    float64
		expected float64
	}{
		{-10.0, 0.0000453978},
		{-1.0, 0.2689414},
		{0.0, 0.5},
		{1.0, 0.7310586},
		{10.0, 0.9999546},
	}

	for _, tc := range testCases {
		result := sigmoid(tc.input)
		// Allow for small floating point differences
		if result < tc.expected-0.0001 || result > tc.expected+0.0001 {
			t.Errorf("sigmoid(%f) = %f, expected approximately %f",
				tc.input, result, tc.expected)
		}
	}

	// Check that sigmoid outputs are in the range [0, 1]
	for i := -100; i <= 100; i += 10 {
		value := sigmoid(float64(i))
		if value < 0.0 || value > 1.0 {
			t.Errorf("sigmoid(%d) = %f, which is outside the range [0, 1]", i, value)
		}
	}
}

func TestRPSValueForward(t *testing.T) {
	network := NewRPSValueNetwork(16)

	// Create a test input
	input := make([]float64, 81)
	for i := 0; i < 81; i++ {
		input[i] = float64(i%3) * 0.1
	}

	// Test that forward pass works
	value := network.forward(input)

	// Value should be in range [0, 1]
	if value < 0.0 || value > 1.0 {
		t.Errorf("Expected forward pass to return value in range [0, 1], got %f", value)
	}
}
