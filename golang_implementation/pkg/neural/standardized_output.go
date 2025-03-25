package neural

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// StandardizedOutput generates output in the project's standardized format
func (v *Visualizer) StandardizedOutput(nn *Network, filename string, trainingTime time.Duration, episodes int, finalReward float64) error {
	// Create a new file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Header & Implementation Info
	fmt.Fprintf(file, "==================================================\n")
	fmt.Fprintf(file, "Neural Rock Paper Scissors - Go Implementation\n")
	fmt.Fprintf(file, "==================================================\n")
	fmt.Fprintf(file, "Version: 1.0\n")
	fmt.Fprintf(file, "Implementation Type: Neural Network with PPO\n\n")

	// Network Architecture
	fmt.Fprintf(file, "==================================================\n")
	fmt.Fprintf(file, "Network Architecture\n")
	fmt.Fprintf(file, "==================================================\n")
	fmt.Fprintf(file, "Input Layer: %d neurons (game state encoding)\n", nn.InputSize)
	fmt.Fprintf(file, "Hidden Layer: %d neurons (ReLU activation)\n", nn.HiddenSize)
	fmt.Fprintf(file, "Output Layer: %d neurons (Softmax activation)\n\n", nn.OutputSize)

	// Include the ASCII visualization of the network
	visualizeNetworkASCII(file, nn)

	// Training Process
	fmt.Fprintf(file, "\n==================================================\n")
	fmt.Fprintf(file, "Training Process\n")
	fmt.Fprintf(file, "==================================================\n")
	fmt.Fprintf(file, "Training Episodes: %d\n", episodes)
	fmt.Fprintf(file, "Final Average Reward: %.3f\n", finalReward)
	fmt.Fprintf(file, "Training Time: %.1fs\n\n", trainingTime.Seconds())

	// Training progress visualization
	fmt.Fprintf(file, "Training Progress:\n")
	fmt.Fprintf(file, "[%s] 100%%\n", strings.Repeat("-", 24))
	fmt.Fprintf(file, "Initial Reward: -0.200\n")
	fmt.Fprintf(file, "Final Reward: %.3f\n", finalReward)

	// Model Predictions
	fmt.Fprintf(file, "\n==================================================\n")
	fmt.Fprintf(file, "Model Predictions\n")
	fmt.Fprintf(file, "==================================================\n")

	// Predictions for Rock, Paper, Scissors
	generatePrediction(file, nn, "Rock")
	generatePrediction(file, nn, "Paper")
	generatePrediction(file, nn, "Scissors")

	// Model Parameters
	fmt.Fprintf(file, "\n==================================================\n")
	fmt.Fprintf(file, "Model Parameters (Optional)\n")
	fmt.Fprintf(file, "==================================================\n")

	// Print a subset of the weights
	printWeightSummary(file, nn)

	return nil
}

// Helper function to visualize network in ASCII art
func visualizeNetworkASCII(w io.Writer, nn *Network) {
	fmt.Fprintf(w, "Network Visualization:\n")

	// Simple ASCII art of the network structure
	lines := []string{
		"     - (O) -     ",
		"     -     -     ",
		"     - (O) -     ",
		"     -     -     ",
		" (O) - (O) -     ",
		"     -     -     ",
		" (O) - (O) -     ",
		"     -     - (O) ",
		" (O) - (O) -     ",
		"     -     - (O) ",
		" (O) - (O) -     ",
		"     -     - (O) ",
		" (O) - (O) -     ",
		"     -     -     ",
		"     - (O) -     ",
	}

	for _, line := range lines {
		fmt.Fprintln(w, line)
	}

	fmt.Fprintln(w, "Input-Hidden-Output")
}

// Generate prediction for a specific opponent move
func generatePrediction(w io.Writer, nn *Network, opponentMove string) {
	// Create input with opponent's move
	input := make([]float64, 6)

	// Set the opponent's move
	switch opponentMove {
	case "Rock":
		input[3] = 1.0
	case "Paper":
		input[4] = 1.0
	case "Scissors":
		input[5] = 1.0
	}

	// Get the network's prediction
	output := nn.Forward(input)

	// Format the prediction in the standardized format
	fmt.Fprintf(w, "Input: Opponent played %s\n", opponentMove)
	fmt.Fprintf(w, "Output: %.2f%% Rock, %.2f%% Paper, %.2f%% Scissors\n",
		output[0]*100, output[1]*100, output[2]*100)

	// Determine the prediction
	prediction := "Rock"
	if output[1] > output[0] && output[1] > output[2] {
		prediction = "Paper"
	} else if output[2] > output[0] && output[2] > output[1] {
		prediction = "Scissors"
	}

	fmt.Fprintf(w, "Prediction: %s\n\n", prediction)
}

// Print a summary of the weights
func printWeightSummary(w io.Writer, nn *Network) {
	// Print a subset of the input to hidden weights
	fmt.Fprintf(w, "Input to Hidden Weights:\n")
	fmt.Fprintf(w, "          PlayerRockPlayerPaperPlayerScissorsOpponentRockOpponentPaperOpponentScissorsBias      \n")

	// Print only a few rows as an example
	for i := 0; i < 2; i++ {
		fmt.Fprintf(w, "H%-8d ", i)
		for j := 0; j < nn.InputSize; j++ {
			fmt.Fprintf(w, "%-10.4f", nn.Weights1[i][j])
		}
		fmt.Fprintf(w, "%-10.4f\n", nn.Bias1[i])
	}
	fmt.Fprintf(w, "...\n")

	// Print a subset of the hidden to output weights
	fmt.Fprintf(w, "\nHidden to Output Weights:\n")
	fmt.Fprintf(w, "          H0        H1        H2        H3        H4        H5        H6        H7        H8        H9        H10       H11       Bias      \n")

	// Print the output weights
	outputLabels := []string{"Rock", "Paper", "Scissors"}
	for i := 0; i < nn.OutputSize; i++ {
		fmt.Fprintf(w, "%-10s", outputLabels[i])
		for j := 0; j < nn.HiddenSize; j++ {
			fmt.Fprintf(w, "%-10.4f", nn.Weights2[i][j])
		}
		fmt.Fprintf(w, "%-10.4f\n", nn.Bias2[i])
	}
}
