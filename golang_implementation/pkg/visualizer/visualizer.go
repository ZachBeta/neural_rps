package visualizer

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Visualizer handles visualization of the neural network and game state
type Visualizer struct {
	outputDir  string
	outputFile *os.File
}

// NewVisualizer creates a new visualizer
func NewVisualizer(outputDir string) (*Visualizer, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, err
	}

	// Create output file with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(outputDir, fmt.Sprintf("training_%s.txt", timestamp))
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	return &Visualizer{
		outputDir:  outputDir,
		outputFile: file,
	}, nil
}

// Close closes the output file
func (v *Visualizer) Close() error {
	if v.outputFile != nil {
		return v.outputFile.Close()
	}
	return nil
}

// VisualizeArchitecture visualizes the neural network architecture
func (v *Visualizer) VisualizeArchitecture(layerSizes []int, layerNames []string) error {
	content := "Neural Network Architecture:\n"
	for i, size := range layerSizes {
		name := "Layer"
		if i < len(layerNames) {
			name = layerNames[i]
		}
		content += fmt.Sprintf("%s: %d neurons\n", name, size)
		if i < len(layerSizes)-1 {
			content += "  ↓\n"
		}
	}
	return v.WriteToFile(content)
}

// VisualizeWeights visualizes the network weights
func (v *Visualizer) VisualizeWeights(weights [][]float64, inputLabels, outputLabels []string) error {
	content := "Network Weights:\n"
	for i, row := range weights {
		outputLabel := "Output"
		if i < len(outputLabels) {
			outputLabel = outputLabels[i]
		}
		content += fmt.Sprintf("%s:\n", outputLabel)
		for j, weight := range row {
			inputLabel := "Input"
			if j < len(inputLabels) {
				inputLabel = inputLabels[j]
			}
			content += fmt.Sprintf("  %s: %.4f\n", inputLabel, weight)
		}
	}
	return v.WriteToFile(content)
}

// VisualizeActionProbs visualizes the action probabilities
func (v *Visualizer) VisualizeActionProbs(probs []float64, actionLabels []string) error {
	content := "Action Probabilities:\n"
	for i, prob := range probs {
		label := "Action"
		if i < len(actionLabels) {
			label = actionLabels[i]
		}
		content += fmt.Sprintf("%s: %.4f\n", label, prob)
	}
	return v.WriteToFile(content)
}

// VisualizeTrainingProgress visualizes the training progress
func (v *Visualizer) VisualizeTrainingProgress(rewards []float64, windowSize int) error {
	if len(rewards) == 0 {
		return nil
	}

	// Calculate moving average
	avgReward := 0.0
	window := windowSize
	if window > len(rewards) {
		window = len(rewards)
	}
	for i := len(rewards) - window; i < len(rewards); i++ {
		avgReward += rewards[i]
	}
	avgReward /= float64(window)

	content := fmt.Sprintf("Training Progress:\n")
	content += fmt.Sprintf("Last %d episodes average reward: %.4f\n", window, avgReward)
	content += fmt.Sprintf("Total episodes: %d\n", len(rewards))
	return v.WriteToFile(content)
}

// WriteToFile writes content to the output file
func (v *Visualizer) WriteToFile(content string) error {
	if v.outputFile == nil {
		return fmt.Errorf("output file not initialized")
	}
	_, err := v.outputFile.WriteString(content)
	return err
}
