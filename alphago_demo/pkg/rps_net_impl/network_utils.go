package neural

// CloneFloat64Slice makes a deep copy of a float64 slice
func CloneFloat64Slice(s []float64) []float64 {
	if s == nil {
		return nil
	}

	clone := make([]float64, len(s))
	copy(clone, s)
	return clone
}

// CloneFloat64Matrix makes a deep copy of a 2D float64 slice
func CloneFloat64Matrix(m [][]float64) [][]float64 {
	if m == nil {
		return nil
	}

	clone := make([][]float64, len(m))
	for i := range m {
		clone[i] = CloneFloat64Slice(m[i])
	}
	return clone
}

// Clone creates a deep copy of a value network
func (n *RPSValueNetwork) Clone() *RPSValueNetwork {
	clone := &RPSValueNetwork{
		inputSize:           n.inputSize,
		hiddenSize:          n.hiddenSize,
		outputSize:          n.outputSize,
		weightsInputHidden:  CloneFloat64Matrix(n.weightsInputHidden),
		biasesHidden:        CloneFloat64Slice(n.biasesHidden),
		weightsHiddenOutput: CloneFloat64Matrix(n.weightsHiddenOutput),
		biasesOutput:        CloneFloat64Slice(n.biasesOutput),
	}

	// Clone debug information if present
	if n.DebugEpochCount != nil {
		clone.DebugEpochCount = make([]int, len(n.DebugEpochCount))
		copy(clone.DebugEpochCount, n.DebugEpochCount)
	}

	return clone
}

// Clone creates a deep copy of a policy network
func (n *RPSPolicyNetwork) Clone() *RPSPolicyNetwork {
	clone := &RPSPolicyNetwork{
		inputSize:           n.inputSize,
		hiddenSize:          n.hiddenSize,
		outputSize:          n.outputSize,
		weightsInputHidden:  CloneFloat64Matrix(n.weightsInputHidden),
		biasesHidden:        CloneFloat64Slice(n.biasesHidden),
		weightsHiddenOutput: CloneFloat64Matrix(n.weightsHiddenOutput),
		biasesOutput:        CloneFloat64Slice(n.biasesOutput),
	}

	// Clone debug information if present
	if n.DebugEpochCount != nil {
		clone.DebugEpochCount = make([]int, len(n.DebugEpochCount))
		copy(clone.DebugEpochCount, n.DebugEpochCount)
	}

	return clone
}
