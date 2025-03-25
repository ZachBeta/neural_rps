package neural

import (
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

func TestNewRPSPolicyNetwork(t *testing.T) {
	network := NewRPSPolicyNetwork(32)

	if network.inputSize != 81 {
		t.Errorf("Expected input size to be 81, got %d", network.inputSize)
	}

	if network.hiddenSize != 32 {
		t.Errorf("Expected hidden size to be 32, got %d", network.hiddenSize)
	}

	if network.outputSize != 9 {
		t.Errorf("Expected output size to be 9, got %d", network.outputSize)
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

func TestRPSPolicyPredict(t *testing.T) {
	network := NewRPSPolicyNetwork(32)
	gameInstance := game.NewRPSGame(15, 5, 10)

	// Test that prediction returns the correct shape
	probs := network.Predict(gameInstance)

	if len(probs) != 9 {
		t.Errorf("Expected prediction to have length 9, got %d", len(probs))
	}

	// Test that probabilities sum to approximately 1.0
	sum := 0.0
	for _, p := range probs {
		sum += p
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Expected probabilities to sum to approximately 1.0, got %f", sum)
	}

	// Test that all probabilities are in the range [0, 1]
	for i, p := range probs {
		if p < 0.0 || p > 1.0 {
			t.Errorf("Probability at index %d is out of range [0, 1]: %f", i, p)
		}
	}
}

func TestRPSPolicyPredictMove(t *testing.T) {
	network := NewRPSPolicyNetwork(32)
	gameInstance := game.NewRPSGame(15, 5, 10)

	// Test that the network can predict a move
	move := network.PredictMove(gameInstance)

	// Verify that the move is valid
	if move.Player != gameInstance.CurrentPlayer {
		t.Errorf("Expected move for player %v, got %v",
			gameInstance.CurrentPlayer, move.Player)
	}

	if move.CardIndex < 0 || move.CardIndex >= len(gameInstance.Player1Hand) {
		t.Errorf("Card index out of range: %d", move.CardIndex)
	}

	if move.Position < 0 || move.Position >= 9 {
		t.Errorf("Position out of range: %d", move.Position)
	}

	// Verify that the move is actually in the list of valid moves
	validMoves := gameInstance.GetValidMoves()
	isValid := false
	for _, validMove := range validMoves {
		if validMove.Position == move.Position && validMove.CardIndex == move.CardIndex {
			isValid = true
			break
		}
	}

	if !isValid {
		t.Errorf("Predicted move is not in the list of valid moves: %+v", move)
	}
}

func TestRPSPolicyTrain(t *testing.T) {
	network := NewRPSPolicyNetwork(16)

	// Create test input and target data
	batchSize := 10
	inputFeatures := make([][]float64, batchSize)
	targetProbs := make([][]float64, batchSize)

	// Initialize with some test data
	for i := 0; i < batchSize; i++ {
		inputFeatures[i] = make([]float64, 81)
		targetProbs[i] = make([]float64, 9)

		// Set some features
		for j := 0; j < 81; j++ {
			inputFeatures[i][j] = float64(j%3) * 0.1
		}

		// Set uniform target probabilities
		for j := 0; j < 9; j++ {
			targetProbs[i][j] = 1.0 / 9.0
		}
	}

	// Test that training completes without error
	learningRate := 0.01
	loss := network.Train(inputFeatures, targetProbs, learningRate)

	// Loss should be positive
	if loss < 0.0 {
		t.Errorf("Expected positive loss value, got %f", loss)
	}

	// Train for a few iterations and check that loss decreases
	initialLoss := loss
	for i := 0; i < 5; i++ {
		loss = network.Train(inputFeatures, targetProbs, learningRate)
	}

	if loss >= initialLoss {
		t.Errorf("Expected loss to decrease after training, initial: %f, final: %f",
			initialLoss, loss)
	}
}

func TestRPSPolicySoftmax(t *testing.T) {
	// Test softmax functionality
	values := []float64{1.0, 2.0, 3.0, 4.0}
	probs := softmax(values)

	// Check that output has the same length
	if len(probs) != len(values) {
		t.Errorf("Expected softmax output to have same length as input, got %d vs %d",
			len(probs), len(values))
	}

	// Check that probabilities sum to approximately 1.0
	sum := 0.0
	for _, p := range probs {
		sum += p
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Expected softmax probabilities to sum to approximately 1.0, got %f", sum)
	}

	// Check that probabilities are in the correct order (monotonically increasing)
	for i := 1; i < len(probs); i++ {
		if probs[i] <= probs[i-1] {
			t.Errorf("Expected softmax to preserve ordering of values, but %f <= %f",
				probs[i], probs[i-1])
		}
	}
}

func TestRPSPolicyReLU(t *testing.T) {
	// Test ReLU functionality
	testCases := []struct {
		input    float64
		expected float64
	}{
		{-2.0, 0.0},
		{-1.0, 0.0},
		{0.0, 0.0},
		{1.0, 1.0},
		{2.0, 2.0},
	}

	for _, tc := range testCases {
		result := relu(tc.input)
		if result != tc.expected {
			t.Errorf("ReLU(%f) = %f, expected %f", tc.input, result, tc.expected)
		}
	}
}
