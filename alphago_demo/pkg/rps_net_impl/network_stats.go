package neural

import "fmt"

// NetworkStats represents statistics about a neural network's complexity
type NetworkStats struct {
	InputSize        int     // Number of input neurons
	HiddenSize       int     // Number of hidden neurons
	OutputSize       int     // Number of output neurons
	TotalNeurons     int     // Total number of neurons
	TotalConnections int     // Total number of connections (weights)
	TotalParameters  int     // Total number of trainable parameters (weights + biases)
	MemoryFootprint  float64 // Estimated memory footprint in KB (assuming float64 = 8 bytes)
}

// CalculatePolicyNetworkStats calculates complexity metrics for a policy network
func CalculatePolicyNetworkStats(network *RPSPolicyNetwork) NetworkStats {
	inputSize := network.inputSize
	hiddenSize := network.hiddenSize
	outputSize := network.outputSize

	// Calculate number of connections
	inputToHidden := inputSize * hiddenSize
	hiddenToOutput := hiddenSize * outputSize
	totalConnections := inputToHidden + hiddenToOutput

	// Calculate total parameters (weights + biases)
	biases := hiddenSize + outputSize
	totalParameters := totalConnections + biases

	// Calculate estimated memory footprint (8 bytes per float64)
	memoryBytes := float64(totalParameters * 8)
	memoryKB := memoryBytes / 1024.0

	return NetworkStats{
		InputSize:        inputSize,
		HiddenSize:       hiddenSize,
		OutputSize:       outputSize,
		TotalNeurons:     inputSize + hiddenSize + outputSize,
		TotalConnections: totalConnections,
		TotalParameters:  totalParameters,
		MemoryFootprint:  memoryKB,
	}
}

// CalculateValueNetworkStats calculates complexity metrics for a value network
func CalculateValueNetworkStats(network *RPSValueNetwork) NetworkStats {
	inputSize := network.inputSize
	hiddenSize := network.hiddenSize
	outputSize := network.outputSize

	// Calculate number of connections
	inputToHidden := inputSize * hiddenSize
	hiddenToOutput := hiddenSize * outputSize
	totalConnections := inputToHidden + hiddenToOutput

	// Calculate total parameters (weights + biases)
	biases := hiddenSize + outputSize
	totalParameters := totalConnections + biases

	// Calculate estimated memory footprint (8 bytes per float64)
	memoryBytes := float64(totalParameters * 8)
	memoryKB := memoryBytes / 1024.0

	return NetworkStats{
		InputSize:        inputSize,
		HiddenSize:       hiddenSize,
		OutputSize:       outputSize,
		TotalNeurons:     inputSize + hiddenSize + outputSize,
		TotalConnections: totalConnections,
		TotalParameters:  totalParameters,
		MemoryFootprint:  memoryKB,
	}
}

// FormatNetworkStats returns a formatted string with network statistics
func FormatNetworkStats(stats NetworkStats) string {
	return fmt.Sprintf(
		"Network Complexity:\n"+
			"  Architecture: %d-%d-%d (input-hidden-output)\n"+
			"  Total neurons: %d\n"+
			"  Total connections: %d\n"+
			"  Total parameters: %d\n"+
			"  Memory footprint: %.2f KB",
		stats.InputSize, stats.HiddenSize, stats.OutputSize,
		stats.TotalNeurons,
		stats.TotalConnections,
		stats.TotalParameters,
		stats.MemoryFootprint,
	)
}

// DisplayNetworkComplexity prints network complexity information for policy and value networks
func DisplayNetworkComplexity(policyNetwork *RPSPolicyNetwork, valueNetwork *RPSValueNetwork) {
	policyStats := CalculatePolicyNetworkStats(policyNetwork)
	valueStats := CalculateValueNetworkStats(valueNetwork)

	fmt.Println("=== Neural Network Complexity ===")
	fmt.Println("Policy Network:")
	fmt.Println(FormatNetworkStats(policyStats))
	fmt.Println("\nValue Network:")
	fmt.Println(FormatNetworkStats(valueStats))

	// Calculate combined statistics
	totalParameters := policyStats.TotalParameters + valueStats.TotalParameters
	totalMemory := policyStats.MemoryFootprint + valueStats.MemoryFootprint
	fmt.Printf("\nTotal model parameters: %d (%.2f KB)\n", totalParameters, totalMemory)
	fmt.Println("===================================")
}
