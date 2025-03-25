package visualizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewVisualizer(t *testing.T) {
	// Create temporary directory for test output
	tmpDir := t.TempDir()
	viz, err := NewVisualizer(tmpDir)
	if err != nil {
		t.Fatalf("NewVisualizer() error = %v", err)
	}
	defer viz.Close()

	// Check if output file was created
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read test directory: %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("Expected 1 file in output directory, got %d", len(entries))
	}
}

func TestVisualizeArchitecture(t *testing.T) {
	tmpDir := t.TempDir()
	viz, err := NewVisualizer(tmpDir)
	if err != nil {
		t.Fatalf("NewVisualizer() error = %v", err)
	}
	defer viz.Close()

	layerSizes := []int{9, 64, 3}
	layerNames := []string{"Input", "Hidden", "Output"}

	err = viz.VisualizeArchitecture(layerSizes, layerNames)
	if err != nil {
		t.Errorf("VisualizeArchitecture() error = %v", err)
	}

	// Read the output file
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read test directory: %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("Expected 1 file in output directory, got %d", len(entries))
	}

	content, err := os.ReadFile(filepath.Join(tmpDir, entries[0].Name()))
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	// Check if content contains expected layer information
	expected := []string{"Input: 9", "Hidden: 64", "Output: 3"}
	for _, line := range expected {
		if !strings.Contains(string(content), line) {
			t.Errorf("Output file missing expected line: %s", line)
		}
	}
}

func TestVisualizeWeights(t *testing.T) {
	tmpDir := t.TempDir()
	viz, err := NewVisualizer(tmpDir)
	if err != nil {
		t.Fatalf("NewVisualizer() error = %v", err)
	}
	defer viz.Close()

	weights := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	inputLabels := []string{"Input1", "Input2", "Input3"}
	outputLabels := []string{"Output1", "Output2"}

	err = viz.VisualizeWeights(weights, inputLabels, outputLabels)
	if err != nil {
		t.Errorf("VisualizeWeights() error = %v", err)
	}
}

func TestVisualizeActionProbs(t *testing.T) {
	tmpDir := t.TempDir()
	viz, err := NewVisualizer(tmpDir)
	if err != nil {
		t.Fatalf("NewVisualizer() error = %v", err)
	}
	defer viz.Close()

	probs := []float64{0.3, 0.4, 0.3}
	actionLabels := []string{"Action1", "Action2", "Action3"}

	err = viz.VisualizeActionProbs(probs, actionLabels)
	if err != nil {
		t.Errorf("VisualizeActionProbs() error = %v", err)
	}
}

func TestVisualizeTrainingProgress(t *testing.T) {
	tmpDir := t.TempDir()
	viz, err := NewVisualizer(tmpDir)
	if err != nil {
		t.Fatalf("NewVisualizer() error = %v", err)
	}
	defer viz.Close()

	rewards := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	windowSize := 3

	err = viz.VisualizeTrainingProgress(rewards, windowSize)
	if err != nil {
		t.Errorf("VisualizeTrainingProgress() error = %v", err)
	}
}
