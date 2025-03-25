package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/zachbeta/neural_rps/pkg/neural"
)

const (
	Rock     = 0
	Paper    = 1
	Scissors = 2
)

var moveNames = []string{"Rock", "Paper", "Scissors"}

// Create one-hot encoding for a move
func encodeMove(move int) []float64 {
	encoding := make([]float64, 3)
	encoding[move] = 1.0
	return encoding
}

// Generate training data for the neural network
func generateTrainingData(numSamples int) ([][]float64, [][]float64) {
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	// Create some patterns
	// Pattern 1: If opponent played rock, play paper
	// Pattern 2: If opponent played paper, play scissors
	// Pattern 3: If opponent played scissors, play rock
	for i := 0; i < numSamples; i++ {
		// Randomly select opponent's previous move
		prevOpponentMove := rand.Intn(3)

		// Select the winning move based on the pattern
		var bestMove int
		switch prevOpponentMove {
		case Rock:
			bestMove = Paper
		case Paper:
			bestMove = Scissors
		case Scissors:
			bestMove = Rock
		}

		// Create input as one-hot encoding of opponent's previous move
		input := make([]float64, 6)     // room for prev player and opponent moves
		input[prevOpponentMove+3] = 1.0 // opponent's move in second half

		// Create target as one-hot encoding of best move
		target := encodeMove(bestMove)

		inputs[i] = input
		targets[i] = target
	}

	return inputs, targets
}

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create a neural network
	nn := neural.NewNetwork(6, 12, 3)
	fmt.Println("Neural network created!")

	// Create a visualizer
	visualizer, err := neural.NewFileVisualizer("training_output.txt")
	if err != nil {
		fmt.Printf("Error creating visualizer: %v\n", err)
		os.Exit(1)
	}
	defer visualizer.Close()

	// Visualize network architecture
	visualizer.VisualizeArchitecture(nn, []string{"Input", "Hidden", "Output"})
	visualizer.VisualizeNetworkGraphical(nn)

	// Generate training data
	fmt.Println("Generating training data...")
	inputs, targets := generateTrainingData(1000)
	fmt.Printf("Generated %d training samples\n", len(inputs))

	// Set up training options
	options := neural.TrainingOptions{
		LearningRate: 0.01,
		Epochs:       500,
		BatchSize:    32,
		Parallel:     true,
	}

	// Train the network
	fmt.Println("Training neural network...")
	err = nn.Train(inputs, targets, options)
	if err != nil {
		fmt.Printf("Error training network: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Training complete!")

	// Save the network weights
	err = nn.SaveWeights("neural_rps_model.gob")
	if err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Model saved to neural_rps_model.gob")

	// Test the model with some examples
	fmt.Println("\nTesting the model:")
	testExamples := []int{Rock, Paper, Scissors}
	outputLabels := []string{"Rock", "Paper", "Scissors"}

	for _, prevOpponentMove := range testExamples {
		// Create input with opponent's previous move
		input := make([]float64, 6)     // room for prev player and opponent moves
		input[prevOpponentMove+3] = 1.0 // opponent's move in second half

		// Get the network's prediction
		output := nn.Forward(input)
		prediction := nn.Predict(input)

		fmt.Printf("If opponent played %s, AI predicts: %s\n",
			moveNames[prevOpponentMove],
			moveNames[prediction])

		// Visualize the prediction
		visualizer.VisualizePrediction(nn, input, output,
			[]string{"PlayerRock", "PlayerPaper", "PlayerScissors", "OpponentRock", "OpponentPaper", "OpponentScissors"},
			outputLabels)
	}

	// Visualize the weights
	visualizer.VisualizeWeights(nn,
		[]string{"PlayerRock", "PlayerPaper", "PlayerScissors", "OpponentRock", "OpponentPaper", "OpponentScissors"},
		nil, // default hidden labels
		[]string{"Rock", "Paper", "Scissors"})

	fmt.Println("\nCheck training_output.txt for detailed visualization")
}
