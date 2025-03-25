package neural

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestVisualizer(t *testing.T) {
	// Create a network for testing
	nn := NewNetwork(3, 4, 2)

	// Create a visualizer that writes to a buffer for testing
	var buf bytes.Buffer
	visualizer := NewVisualizer(&buf)

	// Test architecture visualization
	visualizer.VisualizeArchitecture(nn, []string{"Input", "Hidden", "Output"})
	output := buf.String()

	if !strings.Contains(output, "Network Architecture") {
		t.Error("VisualizeArchitecture should output network architecture header")
	}
	if !strings.Contains(output, "Input Layer: 3 neurons") {
		t.Error("VisualizeArchitecture should show input layer size")
	}
	if !strings.Contains(output, "Hidden Layer: 4 neurons") {
		t.Error("VisualizeArchitecture should show hidden layer size")
	}
	if !strings.Contains(output, "Output Layer: 2 neurons") {
		t.Error("VisualizeArchitecture should show output layer size")
	}

	// Clear buffer for next test
	buf.Reset()

	// Test graphical visualization
	visualizer.VisualizeNetworkGraphical(nn)
	output = buf.String()

	if !strings.Contains(output, "Graphical Network View") {
		t.Error("VisualizeNetworkGraphical should output graphical view header")
	}
	if !strings.Contains(output, "(O)") {
		t.Error("VisualizeNetworkGraphical should contain neuron representations")
	}

	// Clear buffer for next test
	buf.Reset()

	// Test weights visualization
	inputLabels := []string{"Rock", "Paper", "Scissors"}
	hiddenLabels := []string{"H1", "H2", "H3", "H4"}
	outputLabels := []string{"Win", "Lose"}

	visualizer.VisualizeWeights(nn, inputLabels, hiddenLabels, outputLabels)
	output = buf.String()

	if !strings.Contains(output, "Input to Hidden Weights") {
		t.Error("VisualizeWeights should output input to hidden weights header")
	}
	if !strings.Contains(output, "Hidden to Output Weights") {
		t.Error("VisualizeWeights should output hidden to output weights header")
	}
	for _, label := range inputLabels {
		if !strings.Contains(output, label) {
			t.Errorf("VisualizeWeights should contain input label %s", label)
		}
	}
	for _, label := range outputLabels {
		if !strings.Contains(output, label) {
			t.Errorf("VisualizeWeights should contain output label %s", label)
		}
	}

	// Clear buffer for next test
	buf.Reset()

	// Test prediction visualization
	input := []float64{1.0, 0.0, 0.0} // One-hot encoding for Rock
	prediction := nn.Forward(input)

	visualizer.VisualizePrediction(nn, input, prediction, inputLabels, outputLabels)
	output = buf.String()

	if !strings.Contains(output, "Prediction") {
		t.Error("VisualizePrediction should output prediction header")
	}
	if !strings.Contains(output, "Inputs") {
		t.Error("VisualizePrediction should show inputs section")
	}
	if !strings.Contains(output, "Outputs") {
		t.Error("VisualizePrediction should show outputs section")
	}

	// Clear buffer for next test
	buf.Reset()

	// Test training progress visualization
	visualizer.VisualizeTrainingProgress(50, 100, 0.25)
	output = buf.String()

	if !strings.Contains(output, "Epoch 50/100") {
		t.Error("VisualizeTrainingProgress should show epoch information")
	}
	if !strings.Contains(output, "Loss: 0.250000") {
		t.Error("VisualizeTrainingProgress should show loss value")
	}
}

func TestFileVisualizer(t *testing.T) {
	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "neural_test_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	tmpFilename := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpFilename)

	// Create a file visualizer
	visualizer, err := NewFileVisualizer(tmpFilename)
	if err != nil {
		t.Fatalf("Failed to create file visualizer: %v", err)
	}

	// Create a network for testing
	nn := NewNetwork(2, 3, 2)

	// Visualize the network architecture
	visualizer.VisualizeArchitecture(nn, nil)

	// Close the visualizer
	if err := visualizer.Close(); err != nil {
		t.Fatalf("Failed to close visualizer: %v", err)
	}

	// Read the file contents
	content, err := os.ReadFile(tmpFilename)
	if err != nil {
		t.Fatalf("Failed to read temp file: %v", err)
	}

	// Check the content
	if !strings.Contains(string(content), "Network Architecture") {
		t.Error("File should contain network architecture visualization")
	}
	if !strings.Contains(string(content), "Input Layer: 2 neurons") {
		t.Error("File should contain correct input layer size")
	}
}
