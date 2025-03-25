package neural

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// Visualizer provides methods to visualize neural network structure and training
type Visualizer struct {
	writer io.Writer
}

// NewVisualizer creates a new visualizer that writes to the given writer
func NewVisualizer(w io.Writer) *Visualizer {
	return &Visualizer{writer: w}
}

// NewFileVisualizer creates a new visualizer that writes to a file
func NewFileVisualizer(filename string) (*Visualizer, error) {
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	return &Visualizer{writer: file}, nil
}

// VisualizeArchitecture visualizes the network architecture
func (v *Visualizer) VisualizeArchitecture(nn *Network, layerNames []string) {
	if len(layerNames) < 3 {
		layerNames = []string{"Input", "Hidden", "Output"}
	}

	fmt.Fprintf(v.writer, "\nNetwork Architecture:\n")
	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("=", 50))

	fmt.Fprintf(v.writer, "%s Layer: %d neurons\n", layerNames[0], nn.InputSize)
	fmt.Fprintf(v.writer, "  ↓\n")
	fmt.Fprintf(v.writer, "%s Layer: %d neurons\n", layerNames[1], nn.HiddenSize)
	fmt.Fprintf(v.writer, "  ↓\n")
	fmt.Fprintf(v.writer, "%s Layer: %d neurons\n", layerNames[2], nn.OutputSize)

	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("=", 50))
}

// VisualizeNetworkGraphical creates a simple ASCII visualization of the network
func (v *Visualizer) VisualizeNetworkGraphical(nn *Network) {
	// Get the maximum layer size to determine height
	maxLayerSize := max(nn.InputSize, max(nn.HiddenSize, nn.OutputSize))
	maxHeight := maxLayerSize*2 + 1

	// Layer widths for spacing
	layerWidths := []int{nn.InputSize, nn.HiddenSize, nn.OutputSize}
	layerNames := []string{"Input", "Hidden", "Output"}

	fmt.Fprintf(v.writer, "\nGraphical Network View:\n")
	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("=", 50))

	// Print the network visualization
	for h := 0; h < maxHeight; h++ {
		for l := 0; l < len(layerWidths); l++ {
			layerSize := layerWidths[l]
			start := (maxHeight - layerSize*2) / 2
			end := start + layerSize*2

			if h >= start && h < end && (h-start)%2 == 0 {
				fmt.Fprintf(v.writer, " (O) ")
			} else if h == maxHeight-1 {
				fmt.Fprintf(v.writer, "%-5s", layerNames[l])
			} else {
				fmt.Fprintf(v.writer, "     ")
			}

			// Add connecting lines between layers
			if l < len(layerWidths)-1 {
				fmt.Fprintf(v.writer, "-")
			}
		}
		fmt.Fprintf(v.writer, "\n")
	}

	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("=", 50))
}

// VisualizeWeights visualizes the network weights
func (v *Visualizer) VisualizeWeights(nn *Network, inputLabels, hiddenLabels, outputLabels []string) {
	// Generate default labels if not provided
	if len(inputLabels) < nn.InputSize {
		inputLabels = make([]string, nn.InputSize)
		for i := 0; i < nn.InputSize; i++ {
			inputLabels[i] = fmt.Sprintf("In%d", i)
		}
	}

	if len(hiddenLabels) < nn.HiddenSize {
		hiddenLabels = make([]string, nn.HiddenSize)
		for i := 0; i < nn.HiddenSize; i++ {
			hiddenLabels[i] = fmt.Sprintf("H%d", i)
		}
	}

	if len(outputLabels) < nn.OutputSize {
		outputLabels = make([]string, nn.OutputSize)
		for i := 0; i < nn.OutputSize; i++ {
			outputLabels[i] = fmt.Sprintf("Out%d", i)
		}
	}

	// Print input to hidden weights
	fmt.Fprintf(v.writer, "\nInput to Hidden Weights:\n")
	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("-", 50))

	// Print header row with input labels
	fmt.Fprintf(v.writer, "%-10s", "")
	for j := 0; j < nn.InputSize; j++ {
		fmt.Fprintf(v.writer, "%-10s", inputLabels[j])
	}
	fmt.Fprintf(v.writer, "%-10s\n", "Bias")

	// Print weights
	for i := 0; i < nn.HiddenSize; i++ {
		fmt.Fprintf(v.writer, "%-10s", hiddenLabels[i])
		for j := 0; j < nn.InputSize; j++ {
			fmt.Fprintf(v.writer, "%-10.4f", nn.Weights1[i][j])
		}
		fmt.Fprintf(v.writer, "%-10.4f\n", nn.Bias1[i])
	}

	// Print hidden to output weights
	fmt.Fprintf(v.writer, "\nHidden to Output Weights:\n")
	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("-", 50))

	// Print header row with hidden labels
	fmt.Fprintf(v.writer, "%-10s", "")
	for j := 0; j < nn.HiddenSize; j++ {
		fmt.Fprintf(v.writer, "%-10s", hiddenLabels[j])
	}
	fmt.Fprintf(v.writer, "%-10s\n", "Bias")

	// Print weights
	for i := 0; i < nn.OutputSize; i++ {
		fmt.Fprintf(v.writer, "%-10s", outputLabels[i])
		for j := 0; j < nn.HiddenSize; j++ {
			fmt.Fprintf(v.writer, "%-10.4f", nn.Weights2[i][j])
		}
		fmt.Fprintf(v.writer, "%-10.4f\n", nn.Bias2[i])
	}
}

// VisualizePrediction visualizes a single prediction
func (v *Visualizer) VisualizePrediction(nn *Network, input []float64, output []float64, inputLabels, outputLabels []string) {
	// Generate default labels if not provided
	if len(inputLabels) < nn.InputSize {
		inputLabels = make([]string, nn.InputSize)
		for i := 0; i < nn.InputSize; i++ {
			inputLabels[i] = fmt.Sprintf("Input%d", i)
		}
	}

	if len(outputLabels) < nn.OutputSize {
		outputLabels = make([]string, nn.OutputSize)
		for i := 0; i < nn.OutputSize; i++ {
			outputLabels[i] = fmt.Sprintf("Output%d", i)
		}
	}

	fmt.Fprintf(v.writer, "\nPrediction:\n")
	fmt.Fprintf(v.writer, "%s\n", strings.Repeat("-", 50))

	// Print inputs
	fmt.Fprintf(v.writer, "Inputs:\n")
	for i := 0; i < nn.InputSize; i++ {
		fmt.Fprintf(v.writer, "  %s: %.4f\n", inputLabels[i], input[i])
	}

	// Print outputs
	fmt.Fprintf(v.writer, "\nOutputs:\n")
	bestIdx := 0
	bestVal := output[0]
	for i := 0; i < nn.OutputSize; i++ {
		fmt.Fprintf(v.writer, "  %s: %.4f", outputLabels[i], output[i])
		if output[i] > bestVal {
			bestVal = output[i]
			bestIdx = i
		}
		// Visualize the probability with a simple bar
		barLength := int(output[i] * 40.0)
		fmt.Fprintf(v.writer, " %s\n", strings.Repeat("█", barLength))
	}

	fmt.Fprintf(v.writer, "\nPrediction: %s (%.2f%%)\n", outputLabels[bestIdx], bestVal*100.0)
}

// VisualizeTrainingProgress visualizes training progress
func (v *Visualizer) VisualizeTrainingProgress(epoch int, totalEpochs int, loss float64) {
	progress := float64(epoch) / float64(totalEpochs)
	barWidth := 40
	barFilled := int(progress * float64(barWidth))

	fmt.Fprintf(v.writer, "\rEpoch %d/%d [%s%s] Loss: %.6f",
		epoch, totalEpochs,
		strings.Repeat("█", barFilled),
		strings.Repeat(" ", barWidth-barFilled),
		loss)
}

// Close closes the underlying writer if it's a file
func (v *Visualizer) Close() error {
	if closer, ok := v.writer.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// Helper function to find the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
