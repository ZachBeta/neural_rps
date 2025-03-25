package neural

import (
	"fmt"
	"math"
	"sync"
)

// TrainingOptions contains parameters for the training process
type TrainingOptions struct {
	LearningRate float64
	Epochs       int
	BatchSize    int
	Parallel     bool
}

// DefaultTrainingOptions returns the default training options
func DefaultTrainingOptions() TrainingOptions {
	return TrainingOptions{
		LearningRate: 0.01,
		Epochs:       1000,
		BatchSize:    32,
		Parallel:     true,
	}
}

// Train trains the neural network using the provided data
func (nn *Network) Train(inputs [][]float64, targets [][]float64, options TrainingOptions) error {
	if len(inputs) != len(targets) {
		return fmt.Errorf("inputs and targets should have the same length, got %d and %d", len(inputs), len(targets))
	}

	if len(inputs) == 0 {
		return fmt.Errorf("no training data provided")
	}

	// Ensure the input and output dimensions match the network architecture
	if len(inputs[0]) != nn.InputSize {
		return fmt.Errorf("input dimension mismatch: expected %d, got %d", nn.InputSize, len(inputs[0]))
	}

	if len(targets[0]) != nn.OutputSize {
		return fmt.Errorf("output dimension mismatch: expected %d, got %d", nn.OutputSize, len(targets[0]))
	}

	// Training loop
	for epoch := 0; epoch < options.Epochs; epoch++ {
		totalLoss := 0.0

		// Process data in batches
		for i := 0; i < len(inputs); i += options.BatchSize {
			end := i + options.BatchSize
			if end > len(inputs) {
				end = len(inputs)
			}

			// Extract batch
			batchInputs := inputs[i:end]
			batchTargets := targets[i:end]

			// Process batch (with or without parallelization)
			batchLoss := nn.processBatch(batchInputs, batchTargets, options)
			totalLoss += batchLoss
		}

		// Print progress periodically
		if epoch%100 == 0 || epoch == options.Epochs-1 {
			avgLoss := totalLoss / float64(len(inputs))
			fmt.Printf("Epoch %d/%d, Loss: %.6f\n", epoch+1, options.Epochs, avgLoss)
		}
	}

	return nil
}

// processBatch processes a batch of training examples
func (nn *Network) processBatch(inputs [][]float64, targets [][]float64, options TrainingOptions) float64 {
	totalLoss := 0.0

	// Create weight and bias gradients to accumulate over the batch
	dWeights1 := make([][]float64, nn.HiddenSize)
	dBias1 := make([]float64, nn.HiddenSize)
	dWeights2 := make([][]float64, nn.OutputSize)
	dBias2 := make([]float64, nn.OutputSize)

	// Initialize gradients
	for i := 0; i < nn.HiddenSize; i++ {
		dWeights1[i] = make([]float64, nn.InputSize)
	}

	for i := 0; i < nn.OutputSize; i++ {
		dWeights2[i] = make([]float64, nn.HiddenSize)
	}

	// Process all examples in the batch
	var wg sync.WaitGroup
	var mu sync.Mutex

	processSample := func(i int) {
		input := inputs[i]
		target := targets[i]

		// Forward pass
		hidden := make([]float64, nn.HiddenSize)
		for i := 0; i < nn.HiddenSize; i++ {
			sum := nn.Bias1[i]
			for j := 0; j < nn.InputSize; j++ {
				sum += nn.Weights1[i][j] * input[j]
			}
			hidden[i] = sum // Store pre-activation for derivative
		}

		// Apply ReLU
		hiddenActivated := make([]float64, nn.HiddenSize)
		for i := 0; i < nn.HiddenSize; i++ {
			hiddenActivated[i] = relu(hidden[i])
		}

		// Output layer
		output := make([]float64, nn.OutputSize)
		for i := 0; i < nn.OutputSize; i++ {
			sum := nn.Bias2[i]
			for j := 0; j < nn.HiddenSize; j++ {
				sum += nn.Weights2[i][j] * hiddenActivated[j]
			}
			output[i] = sum
		}

		// Apply softmax
		predictions := softmax(output)

		// Compute cross-entropy loss
		loss := 0.0
		for i := 0; i < nn.OutputSize; i++ {
			if target[i] > 0 {
				loss -= target[i] * math.Log(math.Max(predictions[i], 1e-7))
			}
		}

		// Backpropagation

		// Output layer gradients (dL/dz2 = predictions - targets for cross-entropy)
		outputError := make([]float64, nn.OutputSize)
		for i := 0; i < nn.OutputSize; i++ {
			outputError[i] = predictions[i] - target[i]
		}

		// Hidden layer gradients
		hiddenError := make([]float64, nn.HiddenSize)
		for i := 0; i < nn.HiddenSize; i++ {
			for j := 0; j < nn.OutputSize; j++ {
				hiddenError[i] += nn.Weights2[j][i] * outputError[j]
			}
			hiddenError[i] *= reluDerivative(hidden[i])
		}

		// Calculate weight and bias gradients
		localDWeights1 := make([][]float64, nn.HiddenSize)
		localDBias1 := make([]float64, nn.HiddenSize)
		localDWeights2 := make([][]float64, nn.OutputSize)
		localDBias2 := make([]float64, nn.OutputSize)

		// Initialize local gradients
		for i := 0; i < nn.HiddenSize; i++ {
			localDWeights1[i] = make([]float64, nn.InputSize)
			for j := 0; j < nn.InputSize; j++ {
				localDWeights1[i][j] = hiddenError[i] * input[j]
			}
			localDBias1[i] = hiddenError[i]
		}

		for i := 0; i < nn.OutputSize; i++ {
			localDWeights2[i] = make([]float64, nn.HiddenSize)
			for j := 0; j < nn.HiddenSize; j++ {
				localDWeights2[i][j] = outputError[i] * hiddenActivated[j]
			}
			localDBias2[i] = outputError[i]
		}

		// Accumulate gradients into the shared variables
		mu.Lock()
		totalLoss += loss

		// Accumulate dWeights1
		for i := 0; i < nn.HiddenSize; i++ {
			for j := 0; j < nn.InputSize; j++ {
				dWeights1[i][j] += localDWeights1[i][j]
			}
			dBias1[i] += localDBias1[i]
		}

		// Accumulate dWeights2
		for i := 0; i < nn.OutputSize; i++ {
			for j := 0; j < nn.HiddenSize; j++ {
				dWeights2[i][j] += localDWeights2[i][j]
			}
			dBias2[i] += localDBias2[i]
		}
		mu.Unlock()
	}

	// Process each example, either in parallel or sequentially
	if options.Parallel {
		for i := range inputs {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				processSample(idx)
			}(i)
		}
		wg.Wait()
	} else {
		for i := range inputs {
			processSample(i)
		}
	}

	// Apply gradients averaged over the batch
	batchSize := float64(len(inputs))
	for i := 0; i < nn.HiddenSize; i++ {
		for j := 0; j < nn.InputSize; j++ {
			nn.Weights1[i][j] -= options.LearningRate * dWeights1[i][j] / batchSize
		}
		nn.Bias1[i] -= options.LearningRate * dBias1[i] / batchSize
	}

	for i := 0; i < nn.OutputSize; i++ {
		for j := 0; j < nn.HiddenSize; j++ {
			nn.Weights2[i][j] -= options.LearningRate * dWeights2[i][j] / batchSize
		}
		nn.Bias2[i] -= options.LearningRate * dBias2[i] / batchSize
	}

	return totalLoss
}
