package neural

import (
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func TestNewRPSValueNetwork(t *testing.T) {
	network := NewRPSValueNetwork(32)

	if network.inputSize != 81 {
		t.Errorf("Expected input size to be 81, got %d", network.inputSize)
	}

	if network.hiddenSize != 32 {
		t.Errorf("Expected hidden size to be 32, got %d", network.hiddenSize)
	}

	if network.outputSize != 1 {
		t.Errorf("Expected output size to be 1, got %d", network.outputSize)
	}

	// Check that weights and biases were properly initialized
	if len(network.weightsInputHidden) != network.hiddenSize {
		t.Errorf("Expected weightsInputHidden to have size %d, got %d",
			network.hiddenSize, len(network.weightsInputHidden))
	}

	if len(network.biasesHidden) != network.hiddenSize {
		t.Errorf("Expected biasesHidden to have size %d, got %d",
			network.hiddenSize, len(network.biasesHidden))
	}

	if len(network.weightsHiddenOutput) != network.outputSize {
		t.Errorf("Expected weightsHiddenOutput to have size %d, got %d",
			network.outputSize, len(network.weightsHiddenOutput))
	}

	if len(network.biasesOutput) != network.outputSize {
		t.Errorf("Expected biasesOutput to have size %d, got %d",
			network.outputSize, len(network.biasesOutput))
	}
}

func TestRPSValuePredict(t *testing.T) {
	network := NewRPSValueNetwork(32)
	gameInstance := game.NewRPSGame(15, 5, 10)

	// Test that prediction returns a valid value
	value := network.Predict(gameInstance)

	// Value should be in range [0, 1]
	if value < 0.0 || value > 1.0 {
		t.Errorf("Expected value to be in range [0, 1], got %f", value)
	}

	// Make different game states to ensure values change
	// Game state 1: Empty board
	value1 := network.Predict(gameInstance)

	// Game state 2: Add a card to the board
	// Place a Rock card for Player1 at position 0
	gameInstance.Board[0] = game.RPSCard{Type: game.Rock, Owner: game.Player1}

	// Now predict again
	value2 := network.Predict(gameInstance)

	// Game state 3: Add another card to the board
	// Place a Paper card for Player2 at position 4
	gameInstance.Board[4] = game.RPSCard{Type: game.Paper, Owner: game.Player2}

	// Now predict again
	value3 := network.Predict(gameInstance)

	// The values should be different for different game states
	// This is a basic test to make sure the network is responsive to the game state
	// We don't check specific values because they depend on random weight initialization
	if value1 == value2 && value2 == value3 {
		t.Errorf("Expected different values for different game states, got %f, %f, %f",
			value1, value2, value3)
	}
}

func TestRPSValueTrain(t *testing.T) {
	network := NewRPSValueNetwork(16)

	// Create test input and target data
	batchSize := 10
	inputFeatures := make([][]float64, batchSize)
	targetValues := make([]float64, batchSize)

	// Initialize with some test data
	for i := 0; i < batchSize; i++ {
		inputFeatures[i] = make([]float64, 81)

		// Set some features
		for j := 0; j < 81; j++ {
			inputFeatures[i][j] = float64(j%3) * 0.1
		}

		// Set target values (alternating between 0 and 1)
		if i%2 == 0 {
			targetValues[i] = 0.0
		} else {
			targetValues[i] = 1.0
		}
	}

	// Test that training completes without error
	learningRate := 0.01
	loss := network.Train(inputFeatures, targetValues, learningRate)

	// Loss should be positive
	if loss < 0.0 {
		t.Errorf("Expected positive loss value, got %f", loss)
	}

	// Train for a few iterations and check that loss decreases
	initialLoss := loss
	for i := 0; i < 5; i++ {
		loss = network.Train(inputFeatures, targetValues, learningRate)
	}

	if loss >= initialLoss {
		t.Errorf("Expected loss to decrease after training, initial: %f, final: %f",
			initialLoss, loss)
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
