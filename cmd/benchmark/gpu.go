//go:build gpu
// +build gpu

package main

import (
	"github.com/zachbeta/neural_rps/pkg/common"
	"github.com/zachbeta/neural_rps/pkg/neural/gpu"
)

func init() {
	// Register GPU factory when GPU support is enabled
	getGPUNetworkFactory = func() NetworkFactory {
		return func(inputSize, hiddenSize, outputSize int) (common.BatchedNeuralNetwork, error) {
			return gpu.NewGPUNetwork(inputSize, hiddenSize, outputSize)
		}
	}

	// Register GPU mode name
	getModeNameGPU = func() string {
		return gpu.GetGPUModeName()
	}
}
