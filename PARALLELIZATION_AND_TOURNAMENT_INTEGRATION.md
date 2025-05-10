# Neural Network Parallelization and Tournament Integration

This document outlines our strategy for:
1. Parallelizing our neural network to efficiently utilize CPU/GPU resources
2. Integrating our neural network agent into the existing tournament system

## Parallelization Strategies

### CPU Parallelization Using Goroutines

Go's goroutines provide an excellent way to parallelize computation across CPU cores. We can implement the following:

```go
// Add this to our training.go file
func ParallelTrainModel(network *NeuralNetwork, trainInputs, trainTargets [][]float64, 
                        valInputs, valTargets [][]float64, epochs int, 
                        batchSize int, learningRate float64, numWorkers int) {
    
    // Determine number of workers based on available cores
    if numWorkers <= 0 {
        numWorkers = runtime.NumCPU() / 2 // Use 50% of available cores
    }
    
    // Data parallel approach - split the data among workers
    fmt.Printf("Training with %d worker goroutines\n", numWorkers)
    
    // Create worker pool and communication channels
    type batchResult struct {
        gradients map[string][][]float64
        biasGrads map[string][]float64
        loss      float64
    }
    
    jobs := make(chan int, numWorkers)
    results := make(chan batchResult, numWorkers)
    
    // Start worker goroutines
    for w := 0; w < numWorkers; w++ {
        go func(workerId int) {
            // Each worker gets its own copy of the network to avoid contention
            workerNet := deepCopyNetwork(network)
            
            for batchIdx := range jobs {
                // Calculate batch range for this worker
                startIdx := batchIdx * batchSize
                endIdx := startIdx + batchSize
                if endIdx > len(trainInputs) {
                    endIdx = len(trainInputs)
                }
                
                batchInputs := trainInputs[startIdx:endIdx]
                batchTargets := trainTargets[startIdx:endIdx]
                
                // Process mini-batch
                batchLoss := 0.0
                batchGrads := initializeGradients(workerNet)
                
                for i := range batchInputs {
                    // Forward pass
                    predictions := workerNet.Forward(batchInputs[i])
                    
                    // Calculate loss
                    loss := CrossEntropyLoss(predictions, batchTargets[i])
                    batchLoss += loss
                    
                    // Backpropagation - accumulate gradients in worker network
                    workerNet.Backpropagation(batchTargets[i])
                    
                    // Instead of updating weights immediately, accumulate gradients
                    accumulateGradients(batchGrads, workerNet)
                }
                
                // Send result back
                results <- batchResult{
                    gradients: batchGrads.weights,
                    biasGrads: batchGrads.biases,
                    loss: batchLoss / float64(len(batchInputs)),
                }
            }
        }(w)
    }
    
    for epoch := 0; epoch < epochs; epoch++ {
        // Shuffle training data
        shuffleData(trainInputs, trainTargets)
        
        // Submit batch jobs to workers
        numBatches := (len(trainInputs) + batchSize - 1) / batchSize
        
        // Send batch indexes to workers
        for b := 0; b < numBatches; b++ {
            jobs <- b
        }
        
        // Collect and process results
        totalLoss := 0.0
        mainGrads := initializeGradients(network)
        
        for b := 0; b < numBatches; b++ {
            // Get result from any worker
            result := <-results
            totalLoss += result.loss
            
            // Average gradients across workers
            averageGradients(mainGrads, result.gradients, result.biasGrads, float64(numWorkers))
        }
        
        // Update main network weights using accumulated gradients
        network.UpdateWeightsFromGradients(mainGrads, learningRate)
        
        // Evaluate on validation set
        valLoss, valAccuracy := Evaluate(network, valInputs, valTargets)
        
        // Report progress
        fmt.Printf("Epoch %d/%d: Train Loss=%.4f, Val Loss=%.4f, Val Acc=%.2f%%\n",
                  epoch+1, epochs, totalLoss/float64(numBatches), valLoss, valAccuracy*100)
    }
}

// Helper function to create a deep copy of a network
func deepCopyNetwork(orig *NeuralNetwork) *NeuralNetwork {
    // Create a new network with the same dimensions
    nn := NewNeuralNetwork(
        len(orig.InputLayer.Activations),
        len(orig.HiddenLayer.Activations),
        len(orig.OutputLayer.Activations),
    )
    
    // Copy weights and biases
    for i := range orig.HiddenLayer.Weights {
        for j := range orig.HiddenLayer.Weights[i] {
            nn.HiddenLayer.Weights[i][j] = orig.HiddenLayer.Weights[i][j]
        }
    }
    
    for i := range orig.OutputLayer.Weights {
        for j := range orig.OutputLayer.Weights[i] {
            nn.OutputLayer.Weights[i][j] = orig.OutputLayer.Weights[i][j]
        }
    }
    
    for i := range orig.HiddenLayer.Biases {
        nn.HiddenLayer.Biases[i] = orig.HiddenLayer.Biases[i]
    }
    
    for i := range orig.OutputLayer.Biases {
        nn.OutputLayer.Biases[i] = orig.OutputLayer.Biases[i]
    }
    
    return nn
}
```

### GPU Acceleration

For GPU acceleration, we would ideally use a library like CUDA or OpenCL. However, in Go, we'd need to use CGO to interface with these libraries. Here's a conceptual approach:

```go
// Example using CGO to access GPU acceleration (conceptual)
// This would require appropriate CUDA/OpenCL setup

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_ops.h" // Custom C wrapper for our operations
*/
import "C"
import "unsafe"

// GPUAcceleratedForward performs forward pass using GPU acceleration
func (nn *NeuralNetwork) GPUAcceleratedForward(batchInputs [][]float64) [][]float64 {
    // Convert Go slices to C arrays
    inputFlat := flattenBatch(batchInputs)
    inputPtr := unsafe.Pointer(&inputFlat[0])
    
    // Create output buffer
    batchSize := len(batchInputs)
    outputSize := len(nn.OutputLayer.Activations)
    outputFlat := make([]float64, batchSize*outputSize)
    outputPtr := unsafe.Pointer(&outputFlat[0])
    
    // Get pointers to weights and biases
    hiddenWeightsPtr := unsafe.Pointer(&flattenMatrix(nn.HiddenLayer.Weights)[0])
    hiddenBiasesPtr := unsafe.Pointer(&nn.HiddenLayer.Biases[0])
    outputWeightsPtr := unsafe.Pointer(&flattenMatrix(nn.OutputLayer.Weights)[0])
    outputBiasesPtr := unsafe.Pointer(&nn.OutputLayer.Biases[0])
    
    // Call GPU kernel for forward pass
    C.gpu_forward_pass(
        (*C.double)(inputPtr),
        (*C.double)(hiddenWeightsPtr),
        (*C.double)(hiddenBiasesPtr),
        (*C.double)(outputWeightsPtr),
        (*C.double)(outputBiasesPtr),
        (*C.double)(outputPtr),
        C.int(batchSize),
        C.int(len(nn.InputLayer.Activations)),
        C.int(len(nn.HiddenLayer.Activations)),
        C.int(outputSize),
    )
    
    // Reshape flat output back to batch
    return unflattenBatch(outputFlat, batchSize, outputSize)
}
```

### Batch Processing

Even without GPU acceleration, we can implement batch processing to improve performance:

```go
// Add to neural_network.go
func (nn *NeuralNetwork) ForwardBatch(inputs [][]float64) [][]float64 {
    batchSize := len(inputs)
    outputSize := len(nn.OutputLayer.Activations)
    outputs := make([][]float64, batchSize)
    
    // Pre-allocate output slices
    for i := range outputs {
        outputs[i] = make([]float64, outputSize)
    }
    
    // Process each input in the batch
    for i, input := range inputs {
        outputs[i] = nn.Forward(input)
    }
    
    return outputs
}
```

## Tournament Integration

To integrate our neural network into the tournament system, we need to create a proper agent wrapper:

### Neural Network Agent Interface

```go
// neural_agent.go
package agents

import (
    "github.com/ZachBeta/neural_rps/neural_from_scratch/neural"
    "github.com/ZachBeta/neural_rps/alphago_demo/pkg/game"
)

// NeuralNetworkAgent implements the Agent interface for tournament play
type NeuralNetworkAgent struct {
    name           string
    network        *neural.NeuralNetwork
    explorationTemp float64 // Temperature parameter for controlling exploration
}

// NewNeuralNetworkAgent creates a new agent using a pre-trained neural network
func NewNeuralNetworkAgent(name string, modelPath string, explorationTemp float64) (*NeuralNetworkAgent, error) {
    // Load the pre-trained model
    network, err := neural.LoadModel(modelPath)
    if err != nil {
        return nil, err
    }
    
    return &NeuralNetworkAgent{
        name:           name,
        network:        network,
        explorationTemp: explorationTemp,
    }, nil
}

// GetMove implements the Agent interface
func (a *NeuralNetworkAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
    // 1. Convert game state to neural network features
    features := gameStateToFeatures(state)
    
    // 2. Get move probabilities from neural network
    probabilities := a.network.Forward(features)
    
    // 3. Get valid moves from the game state
    validMoves := state.GetValidMoves()
    
    // 4. Select the best valid move
    var bestMove game.RPSMove
    var bestProb float64 = -1.0
    
    // If using exploration temperature, adjust probabilities
    if a.explorationTemp > 0 {
        probabilities = applyTemperature(probabilities, a.explorationTemp)
        // Optionally sample from distribution rather than taking max
        if rand.Float64() < 0.3 { // 30% chance to explore
            return sampleMove(validMoves, probabilities), nil
        }
    }
    
    // Find the valid move with highest probability
    for _, move := range validMoves {
        if probabilities[move.Position] > bestProb {
            bestProb = probabilities[move.Position]
            bestMove = move
        }
    }
    
    return bestMove, nil
}

// Name returns the agent's name
func (a *NeuralNetworkAgent) Name() string {
    return a.name
}

// Helper function to convert game state to features
func gameStateToFeatures(state *game.RPSGame) []float64 {
    // This function needs to match the feature encoding used during training
    
    // Create feature vector
    features := make([]float64, 81) // Adjust size as needed
    
    // Encode the board state
    for pos := 0; pos < 9; pos++ {
        card := state.GetCardAt(pos)
        offset := pos * 7
        
        if card == nil {
            features[offset] = 1.0 // Empty position
        } else if card.Owner == game.Player1 {
            switch card.Type {
            case game.Rock:
                features[offset+1] = 1.0
            case game.Paper:
                features[offset+2] = 1.0
            case game.Scissors:
                features[offset+3] = 1.0
            }
        } else { // Player2
            switch card.Type {
            case game.Rock:
                features[offset+4] = 1.0
            case game.Paper:
                features[offset+5] = 1.0
            case game.Scissors:
                features[offset+6] = 1.0
            }
        }
    }
    
    // Encode additional state information (hand cards, current player, etc.)
    // ...
    
    return features
}
```

### Tournament Integration

Now, we'll modify the tournament code to include our neural network agent:

```go
// In tournament.go or your main tournament file
func RunTournament() {
    // Create agents
    randomAgent := &agents.RandomAgent{Name: "Random"}
    minimaxAgent := agents.NewMinimaxAgent("Minimax-D3", 3, time.Second*5)
    
    // Create neural network agent
    neuralAgent, err := agents.NewNeuralNetworkAgent(
        "NeuralNet-128H", 
        "models/neural_net_128.model", 
        0.0, // No exploration in tournament
    )
    if err != nil {
        log.Fatalf("Failed to create neural agent: %v", err)
    }
    
    // List of all participants
    participants := []agents.Agent{
        randomAgent,
        minimaxAgent,
        neuralAgent,
    }
    
    // Run tournament (existing code)
    results := runRoundRobinTournament(participants, 100) // 100 games per matchup
    
    // Print results
    printTournamentResults(results)
}
```

### Performance Benchmarking

To evaluate our neural network's performance, we should add benchmarking code:

```go
func BenchmarkAgents() {
    // Load a set of test positions
    testPositions := loadTestPositions(100) // 100 diverse game positions
    
    // Create agents to benchmark
    agents := []agents.Agent{
        &agents.RandomAgent{Name: "Random"},
        agents.NewMinimaxAgent("Minimax-D1", 1, time.Second),
        agents.NewMinimaxAgent("Minimax-D3", 3, time.Second),
        agents.NewNeuralNetworkAgent("NeuralNet-128H", "models/neural_net_128.model", 0.0),
    }
    
    // Reference agent for measuring decision quality
    referenceAgent := agents.NewMinimaxAgent("Reference-D5", 5, time.Second*30)
    
    fmt.Println("=== Agent Performance Benchmark ===")
    fmt.Printf("Testing on %d positions\n\n", len(testPositions))
    
    for _, agent := range agents {
        fmt.Printf("Agent: %s\n", agent.Name())
        
        var totalTime time.Duration
        var correctMoves int
        var totalMoves int
        
        for _, pos := range testPositions {
            // Get reference best move
            refMove, _ := referenceAgent.GetMove(pos.Copy())
            
            // Time the agent's decision
            start := time.Now()
            agentMove, _ := agent.GetMove(pos.Copy())
            elapsed := time.Since(start)
            
            totalTime += elapsed
            totalMoves++
            
            // Check if move matches reference
            if agentMove.Position == refMove.Position {
                correctMoves++
            }
        }
        
        // Calculate metrics
        avgTime := totalTime.Seconds() / float64(totalMoves)
        accuracy := float64(correctMoves) / float64(totalMoves) * 100
        
        fmt.Printf("  Average decision time: %.6f seconds\n", avgTime)
        fmt.Printf("  Decision accuracy: %.2f%%\n", accuracy)
        fmt.Printf("  Decisions per second: %.1f\n", 1.0/avgTime)
        fmt.Println()
    }
}
```

## Training Pipeline with Parallelization

Here's how to tie everything together to train our neural network using parallelization:

```go
// In cmd/train_parallel/main.go
func main() {
    // Parse command line arguments
    hiddenSize := flag.Int("hidden", 128, "Size of hidden layer")
    epochs := flag.Int("epochs", 50, "Number of training epochs")
    batchSize := flag.Int("batch", 32, "Batch size")
    learningRate := flag.Float64("lr", 0.01, "Learning rate")
    threads := flag.Int("threads", runtime.NumCPU()/2, "Number of worker threads (default: 50% of CPU)")
    outputModel := flag.String("output", "neural_net.model", "Output model filename")
    flag.Parse()
    
    // Create neural network
    network := neural.NewNeuralNetwork(81, *hiddenSize, 9)
    
    // Load training data (from minimax-generated positions)
    fmt.Println("Loading training data...")
    trainInputs, trainTargets := loadTrainingData("data/training_data.json")
    valInputs, valTargets := loadValidationData("data/validation_data.json")
    
    fmt.Printf("Training data loaded: %d examples\n", len(trainInputs))
    fmt.Printf("Validation data loaded: %d examples\n", len(valInputs))
    
    // Train the network with parallelization
    fmt.Printf("Training with %d worker threads...\n", *threads)
    start := time.Now()
    
    neural.ParallelTrainModel(
        network,
        trainInputs,
        trainTargets,
        valInputs,
        valTargets,
        *epochs,
        *batchSize,
        *learningRate,
        *threads,
    )
    
    elapsed := time.Since(start)
    fmt.Printf("Training completed in %s\n", elapsed)
    
    // Save the trained model
    err := neural.SaveModel(network, *outputModel)
    if err != nil {
        log.Fatalf("Failed to save model: %v", err)
    }
    
    fmt.Printf("Model saved to %s\n", *outputModel)
    
    // Run a quick benchmark
    fmt.Println("\nRunning quick benchmark...")
    benchmarkModel(network)
}
```

## Conclusion

This implementation plan provides a clear path forward for:

1. Parallelizing our neural network training to utilize ~50% of CPU resources
2. Optionally adding GPU acceleration for further performance gains
3. Creating a neural network agent that can participate in tournaments
4. Measuring and benchmarking the performance of our neural network agent

The next steps would be to implement these components incrementally, testing each part to ensure it works correctly before moving on to the next. 