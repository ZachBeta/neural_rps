//go:build gpu
// +build gpu

package gpu

import (
	"fmt"

	"github.com/zachbeta/neural_rps/pkg/common"
)

// init registers the GPU network factory functions
func init() {
	// This function will be called when the GPU package is imported
	// Register our factory functions for external use
}

// NewGPUNetwork is a factory function for creating GPU-accelerated neural networks
// It wraps the TensorFlowNetwork to allow easier usage from external packages
func NewGPUNetwork(inputSize, hiddenSize, outputSize int) (common.BatchedNeuralNetwork, error) {
	// For now, use the mock implementation
	// This will be replaced with actual TensorFlow implementation
	// once the dependency issues are resolved
	return NewMockGPUNetwork(inputSize, hiddenSize, outputSize)
}

// GetGPUModeName returns a string description of the current GPU mode
func GetGPUModeName() string {
	return fmt.Sprintf("GPU (Simulated)")
}
