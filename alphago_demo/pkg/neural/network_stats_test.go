package neural

import (
	"strings"
	"testing"
)

func TestCalculatePolicyNetworkStats(t *testing.T) {
	// Create a policy network with known dimensions
	hiddenSize := 64
	network := NewRPSPolicyNetwork(hiddenSize)

	// Calculate stats
	stats := CalculatePolicyNetworkStats(network)

	// Test that the dimensions are correct
	if stats.InputSize != 81 {
		t.Errorf("Expected input size 81, got %d", stats.InputSize)
	}

	if stats.HiddenSize != hiddenSize {
		t.Errorf("Expected hidden size %d, got %d", hiddenSize, stats.HiddenSize)
	}

	if stats.OutputSize != 9 {
		t.Errorf("Expected output size 9, got %d", stats.OutputSize)
	}

	// Test derived statistics
	expectedTotalNeurons := 81 + hiddenSize + 9
	if stats.TotalNeurons != expectedTotalNeurons {
		t.Errorf("Expected total neurons %d, got %d", expectedTotalNeurons, stats.TotalNeurons)
	}

	// Input->Hidden + Hidden->Output connections
	expectedConnections := (81 * hiddenSize) + (hiddenSize * 9)
	if stats.TotalConnections != expectedConnections {
		t.Errorf("Expected total connections %d, got %d", expectedConnections, stats.TotalConnections)
	}

	// Connections + biases (hidden + output)
	expectedParameters := expectedConnections + hiddenSize + 9
	if stats.TotalParameters != expectedParameters {
		t.Errorf("Expected total parameters %d, got %d", expectedParameters, stats.TotalParameters)
	}

	// Memory footprint should be positive
	if stats.MemoryFootprint <= 0 {
		t.Errorf("Expected positive memory footprint, got %f", stats.MemoryFootprint)
	}
}

func TestCalculateValueNetworkStats(t *testing.T) {
	// Create a value network with known dimensions
	hiddenSize := 32
	network := NewRPSValueNetwork(hiddenSize)

	// Calculate stats
	stats := CalculateValueNetworkStats(network)

	// Test that the dimensions are correct
	if stats.InputSize != 81 {
		t.Errorf("Expected input size 81, got %d", stats.InputSize)
	}

	if stats.HiddenSize != hiddenSize {
		t.Errorf("Expected hidden size %d, got %d", hiddenSize, stats.HiddenSize)
	}

	if stats.OutputSize != 1 {
		t.Errorf("Expected output size 1, got %d", stats.OutputSize)
	}

	// Test derived statistics
	expectedTotalNeurons := 81 + hiddenSize + 1
	if stats.TotalNeurons != expectedTotalNeurons {
		t.Errorf("Expected total neurons %d, got %d", expectedTotalNeurons, stats.TotalNeurons)
	}

	// Input->Hidden + Hidden->Output connections
	expectedConnections := (81 * hiddenSize) + (hiddenSize * 1)
	if stats.TotalConnections != expectedConnections {
		t.Errorf("Expected total connections %d, got %d", expectedConnections, stats.TotalConnections)
	}

	// Connections + biases (hidden + output)
	expectedParameters := expectedConnections + hiddenSize + 1
	if stats.TotalParameters != expectedParameters {
		t.Errorf("Expected total parameters %d, got %d", expectedParameters, stats.TotalParameters)
	}

	// Memory footprint should be positive
	if stats.MemoryFootprint <= 0 {
		t.Errorf("Expected positive memory footprint, got %f", stats.MemoryFootprint)
	}
}

func TestFormatNetworkStats(t *testing.T) {
	// Create test stats
	stats := NetworkStats{
		InputSize:        81,
		HiddenSize:       64,
		OutputSize:       9,
		TotalNeurons:     154,
		TotalConnections: 5184 + 576, // 81*64 + 64*9
		TotalParameters:  5184 + 576 + 64 + 9,
		MemoryFootprint:  46.64, // (5184 + 576 + 64 + 9) * 8 / 1024
	}

	// Format the stats
	formatted := FormatNetworkStats(stats)

	// Check that the format contains expected elements
	expectedStrings := []string{
		"Network Complexity:",
		"Architecture: 81-64-9",
		"Total neurons: 154",
		"Total connections: 5760",
		"Total parameters: 5833",
		"Memory footprint: 46.64 KB",
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(formatted, expected) {
			t.Errorf("Expected formatted string to contain '%s', but it didn't. Got: %s", expected, formatted)
		}
	}
}

func TestDisplayNetworkComplexity(t *testing.T) {
	// This test only verifies that the function doesn't panic
	// Since it mainly prints to console, we don't test actual output
	policyNetwork := NewRPSPolicyNetwork(64)
	valueNetwork := NewRPSValueNetwork(32)

	// This should not panic
	DisplayNetworkComplexity(policyNetwork, valueNetwork)
}
