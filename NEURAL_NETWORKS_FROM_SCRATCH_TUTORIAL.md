# Building Neural Networks from Scratch: A Pedagogical Guide

## Introduction

This tutorial explores the implementation of neural networks from first principles, focusing on understanding rather than leveraging existing libraries. By building our supervised learning pipeline for the RPS card game from scratch, we'll gain deeper insights into how neural networks learn.

## Why Build from Scratch?

While frameworks like TensorFlow and PyTorch offer optimized implementations, implementing neural networks manually provides several educational benefits:

- **Transparency**: See exactly how data flows through the network
- **Understanding**: Grasp the mathematics behind backpropagation
- **Control**: Experiment with custom architectures and learning rules
- **Debugging**: Identify and fix issues at a fundamental level

## Core Components of Our Implementation

### 1. Neural Network Structure

```go
// Basic layer in our neural network
type Layer struct {
    Weights      [][]float64  // Weight matrix
    Biases       []float64    // Bias vector
    Activations  []float64    // Output values
    Inputs       []float64    // Input values (for backprop)
    Gradients    [][]float64  // Weight gradients
    BiasGrads    []float64    // Bias gradients
}

// Simple feed-forward neural network
type NeuralNetwork struct {
    InputLayer   *Layer
    HiddenLayer  *Layer
    OutputLayer  *Layer
}
```

### 2. Forward Pass (Prediction)

```go
// Perform forward pass through the network
func (nn *NeuralNetwork) Forward(input []float64) []float64 {
    // Store input for backpropagation
    nn.InputLayer.Inputs = input
    
    // Calculate hidden layer activations
    for i := range nn.HiddenLayer.Activations {
        sum := nn.HiddenLayer.Biases[i]
        for j, val := range input {
            sum += val * nn.HiddenLayer.Weights[j][i]
        }
        // Apply ReLU activation
        nn.HiddenLayer.Activations[i] = math.Max(0, sum)
    }
    
    // Calculate output layer activations
    for i := range nn.OutputLayer.Activations {
        sum := nn.OutputLayer.Biases[i]
        for j, val := range nn.HiddenLayer.Activations {
            sum += val * nn.OutputLayer.Weights[j][i]
        }
        nn.OutputLayer.Activations[i] = sum
    }
    
    // Apply softmax to output layer
    nn.OutputLayer.Activations = softmax(nn.OutputLayer.Activations)
    
    return nn.OutputLayer.Activations
}

// Softmax activation function for output layer
func softmax(values []float64) []float64 {
    result := make([]float64, len(values))
    
    // Find maximum to prevent numerical overflow
    max := values[0]
    for _, v := range values {
        if v > max {
            max = v
        }
    }
    
    // Calculate exp(x_i - max) for each value
    sum := 0.0
    for i, v := range values {
        exp := math.Exp(v - max)
        result[i] = exp
        sum += exp
    }
    
    // Normalize by dividing by sum
    for i := range result {
        result[i] /= sum
    }
    
    return result
}
```

### 3. Loss Function

```go
// Calculate cross-entropy loss between predictions and targets
func crossEntropyLoss(predictions, targets []float64) float64 {
    loss := 0.0
    for i, target := range targets {
        if target > 0 { // Only consider positive targets (usually one-hot encoded)
            // Add small epsilon to avoid log(0)
            loss -= target * math.Log(predictions[i] + 1e-10)
        }
    }
    return loss
}
```

### 4. Backpropagation

```go
// Calculate gradients through backpropagation
func (nn *NeuralNetwork) Backpropagation(targets []float64) {
    // Output layer gradients
    outputDeltas := make([]float64, len(nn.OutputLayer.Activations))
    for i, prediction := range nn.OutputLayer.Activations {
        // Gradient of softmax + cross-entropy is simply (prediction - target)
        outputDeltas[i] = prediction - targets[i]
    }
    
    // Update output layer gradients
    for i := range nn.OutputLayer.Weights {
        for j := range nn.OutputLayer.Weights[i] {
            // Gradient = input_activation * output_delta
            nn.OutputLayer.Gradients[i][j] = nn.HiddenLayer.Activations[i] * outputDeltas[j]
        }
    }
    
    // Update output layer bias gradients
    for i, delta := range outputDeltas {
        nn.OutputLayer.BiasGrads[i] = delta
    }
    
    // Hidden layer gradients
    hiddenDeltas := make([]float64, len(nn.HiddenLayer.Activations))
    for i := range hiddenDeltas {
        // Sum product of output weights and deltas
        for j, delta := range outputDeltas {
            hiddenDeltas[i] += nn.OutputLayer.Weights[i][j] * delta
        }
        
        // Apply derivative of ReLU
        if nn.HiddenLayer.Activations[i] > 0 {
            // ReLU derivative is 1 for x > 0, otherwise 0
            // No need to multiply by 0 when activation <= 0
        } else {
            hiddenDeltas[i] = 0
        }
    }
    
    // Update hidden layer gradients
    for i := range nn.HiddenLayer.Weights {
        for j := range nn.HiddenLayer.Weights[i] {
            nn.HiddenLayer.Gradients[i][j] = nn.InputLayer.Inputs[i] * hiddenDeltas[j]
        }
    }
    
    // Update hidden layer bias gradients
    for i, delta := range hiddenDeltas {
        nn.HiddenLayer.BiasGrads[i] = delta
    }
}
```

### 5. Weight Update (Gradient Descent)

```go
// Update weights using gradient descent
func (nn *NeuralNetwork) UpdateWeights(learningRate float64) {
    // Update output layer weights and biases
    for i := range nn.OutputLayer.Weights {
        for j := range nn.OutputLayer.Weights[i] {
            nn.OutputLayer.Weights[i][j] -= learningRate * nn.OutputLayer.Gradients[i][j]
        }
    }
    
    for i := range nn.OutputLayer.Biases {
        nn.OutputLayer.Biases[i] -= learningRate * nn.OutputLayer.BiasGrads[i]
    }
    
    // Update hidden layer weights and biases
    for i := range nn.HiddenLayer.Weights {
        for j := range nn.HiddenLayer.Weights[i] {
            nn.HiddenLayer.Weights[i][j] -= learningRate * nn.HiddenLayer.Gradients[i][j]
        }
    }
    
    for i := range nn.HiddenLayer.Biases {
        nn.HiddenLayer.Biases[i] -= learningRate * nn.HiddenLayer.BiasGrads[i]
    }
}
```

## Implementing the Training Loop

Now that we have our core neural network components, let's implement the training loop for our RPS game:

```go
// Train the network on a batch of examples
func trainBatch(network *NeuralNetwork, inputs, targets [][]float64, learningRate float64) float64 {
    totalLoss := 0.0
    
    for i := range inputs {
        // Forward pass
        predictions := network.Forward(inputs[i])
        
        // Calculate loss
        loss := crossEntropyLoss(predictions, targets[i])
        totalLoss += loss
        
        // Backpropagation
        network.Backpropagation(targets[i])
        
        // Update weights
        network.UpdateWeights(learningRate)
    }
    
    return totalLoss / float64(len(inputs))
}

// Train the network for multiple epochs
func train(network *NeuralNetwork, trainInputs, trainTargets, valInputs, valTargets [][]float64, 
           epochs int, batchSize int, learningRate float64) {
    
    for epoch := 0; epoch < epochs; epoch++ {
        // Shuffle training data
        shuffleData(trainInputs, trainTargets)
        
        // Train in batches
        totalLoss := 0.0
        numBatches := (len(trainInputs) + batchSize - 1) / batchSize
        
        for b := 0; b < numBatches; b++ {
            startIdx := b * batchSize
            endIdx := startIdx + batchSize
            if endIdx > len(trainInputs) {
                endIdx = len(trainInputs)
            }
            
            batchInputs := trainInputs[startIdx:endIdx]
            batchTargets := trainTargets[startIdx:endIdx]
            
            batchLoss := trainBatch(network, batchInputs, batchTargets, learningRate)
            totalLoss += batchLoss
        }
        
        avgLoss := totalLoss / float64(numBatches)
        
        // Evaluate on validation set
        valLoss, valAccuracy := evaluate(network, valInputs, valTargets)
        
        fmt.Printf("Epoch %d/%d: Train Loss=%.4f, Val Loss=%.4f, Val Acc=%.2f%%\n",
                 epoch+1, epochs, avgLoss, valLoss, valAccuracy*100)
        
        // Implement early stopping here
    }
}
```

## Feature Transformation for RPS Game States

A critical part of our implementation is transforming game states into suitable neural network inputs:

```go
// Convert a game state to neural network input features
func gameStateToFeatures(g *game.RPSGame) []float64 {
    features := make([]float64, 81) // 9×7 + 3 + 3 + 2 + 4 = 81 features
    
    // One-hot encode the board (9 positions × 7 states)
    for pos, card := range g.Board {
        offset := pos * 7
        
        if card.Owner == game.NoPlayer {
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
    
    // Encode hand cards (normalized counts)
    offset := 63
    rockCount, paperCount, scissorsCount := countCardTypes(g.Player1Hand)
    features[offset] = float64(rockCount) / 5.0
    features[offset+1] = float64(paperCount) / 5.0
    features[offset+2] = float64(scissorsCount) / 5.0
    
    // Repeat for Player2's hand
    offset = 66
    rockCount, paperCount, scissorsCount = countCardTypes(g.Player2Hand)
    features[offset] = float64(rockCount) / 5.0
    features[offset+1] = float64(paperCount) / 5.0
    features[offset+2] = float64(scissorsCount) / 5.0
    
    // Encode current player
    offset = 69
    if g.CurrentPlayer == game.Player1 {
        features[offset] = 1.0
    } else {
        features[offset+1] = 1.0
    }
    
    // Additional game state features (e.g., round number)
    offset = 71
    features[offset] = float64(g.Round) / 10.0
    
    return features
}
```

## Visualizing Learning Progress

To understand how our network is learning, we'll implement visualization tools:

```go
// Log weight histograms to understand distribution changes
func logWeightHistogram(network *NeuralNetwork, epoch int) {
    // Create histogram bins
    bins := make([]int, 10) // -1.0 to 1.0 in 0.2 increments
    
    // Count weights in hidden layer
    for _, row := range network.HiddenLayer.Weights {
        for _, weight := range row {
            bin := int((weight + 1.0) * 5)
            if bin < 0 {
                bin = 0
            } else if bin >= 10 {
                bin = 9
            }
            bins[bin]++
        }
    }
    
    // Print histogram
    fmt.Printf("Epoch %d Hidden Layer Weight Histogram:\n", epoch)
    for i, count := range bins {
        min := -1.0 + float64(i)*0.2
        max := min + 0.2
        fmt.Printf("%.1f to %.1f: %s\n", min, max, strings.Repeat("#", count/10))
    }
}

// Visualize move predictions for a sample position
func visualizePredictions(network *NeuralNetwork, gameState *game.RPSGame) {
    features := gameStateToFeatures(gameState)
    predictions := network.Forward(features)
    
    fmt.Println("Move Predictions:")
    fmt.Println("┌───┬───┬───┐")
    for row := 0; row < 3; row++ {
        fmt.Print("│")
        for col := 0; col < 3; col++ {
            pos := row*3 + col
            fmt.Printf("%3.0f%%│", predictions[pos]*100)
        }
        fmt.Println()
        if row < 2 {
            fmt.Println("├───┼───┼───┤")
        }
    }
    fmt.Println("└───┴───┴───┘")
}
```

## Analyzing and Debugging the Network

Implementing these debugging tools will help us understand our network:

```go
// Validate gradients numerically (for debugging)
func checkGradients(network *NeuralNetwork, input, target []float64) {
    epsilon := 1e-4
    
    // Store original predictions and loss
    originalPredictions := network.Forward(input)
    originalLoss := crossEntropyLoss(originalPredictions, target)
    
    // Check output layer weight gradients
    for i := range network.OutputLayer.Weights {
        for j := range network.OutputLayer.Weights[i] {
            // Save original weight
            originalWeight := network.OutputLayer.Weights[i][j]
            
            // Add epsilon
            network.OutputLayer.Weights[i][j] = originalWeight + epsilon
            predictions := network.Forward(input)
            lossPlus := crossEntropyLoss(predictions, target)
            
            // Subtract epsilon
            network.OutputLayer.Weights[i][j] = originalWeight - epsilon
            predictions = network.Forward(input)
            lossMinus := crossEntropyLoss(predictions, target)
            
            // Restore original weight
            network.OutputLayer.Weights[i][j] = originalWeight
            
            // Calculate numerical gradient
            numericalGradient := (lossPlus - lossMinus) / (2 * epsilon)
            
            // Compare with analytical gradient
            network.Forward(input)
            network.Backpropagation(target)
            analyticalGradient := network.OutputLayer.Gradients[i][j]
            
            diff := math.Abs(numericalGradient - analyticalGradient)
            if diff > 1e-5 {
                fmt.Printf("Gradient check failed at output[%d][%d]: numerical=%.6f, analytical=%.6f\n",
                           i, j, numericalGradient, analyticalGradient)
            }
        }
    }
    
    // Similar checks can be done for hidden layer gradients
}
```

## Running Experiments

Let's set up some experiments to understand the learning process:

```go
// Run different experiments to understand learning
func runExperiments() {
    // Experiment 1: Effect of hidden layer size
    hiddenSizes := []int{16, 32, 64, 128, 256}
    for _, size := range hiddenSizes {
        network := NewNeuralNetwork(81, size, 9)
        // Train and evaluate with this configuration
        // ...
        fmt.Printf("Hidden size %d: Final validation accuracy %.2f%%\n", 
                   size, finalAccuracy*100)
    }
    
    // Experiment 2: Effect of learning rate
    learningRates := []float64{0.1, 0.01, 0.001, 0.0001}
    for _, lr := range learningRates {
        network := NewNeuralNetwork(81, 128, 9)
        // Train and evaluate with this configuration
        // ...
        fmt.Printf("Learning rate %.4f: Final validation accuracy %.2f%%\n", 
                   lr, finalAccuracy*100)
    }
    
    // Experiment 3: Effect of training data size
    trainingSizes := []int{100, 500, 1000, 5000, 10000}
    for _, size := range trainingSizes {
        // Generate training data of this size
        // ...
        network := NewNeuralNetwork(81, 128, 9)
        // Train and evaluate with this configuration
        // ...
        fmt.Printf("Training examples %d: Final validation accuracy %.2f%%\n", 
                   size, finalAccuracy*100)
    }
}
```

## Comparing with Minimax

To assess the quality of our neural network, we'll compare its decisions with minimax:

```go
// Compare neural network decisions with minimax
func compareWithMinimax(network *NeuralNetwork, minimaxDepth int, numPositions int) {
    // Create a minimax agent
    minimaxAgent := agents.NewMinimaxAgent("Minimax", minimaxDepth, 5*time.Second, true)
    
    agreements := 0
    betterChoices := 0
    worseChoices := 0
    
    for i := 0; i < numPositions; i++ {
        // Generate a random game position
        game := generateRandomPosition()
        
        // Get minimax decision
        minimaxMove, _ := minimaxAgent.GetMove(game)
        
        // Get neural network decision
        features := gameStateToFeatures(game)
        predictions := network.Forward(features)
        
        // Find the highest probability move that is valid
        validMoves := game.GetValidMoves()
        var bestMove game.RPSMove
        bestProb := -1.0
        
        for _, move := range validMoves {
            prob := predictions[move.Position]
            if prob > bestProb {
                bestProb = prob
                bestMove = move
            }
        }
        
        // Compare decisions
        if bestMove.Position == minimaxMove.Position {
            agreements++
        } else {
            // Evaluate both moves to see which is better
            gameWithMinimax := game.Copy()
            minimaxMove.Player = game.CurrentPlayer
            gameWithMinimax.MakeMove(minimaxMove)
            minimaxScore := evaluatePosition(gameWithMinimax, game.CurrentPlayer)
            
            gameWithNN := game.Copy()
            bestMove.Player = game.CurrentPlayer
            gameWithNN.MakeMove(bestMove)
            nnScore := evaluatePosition(gameWithNN, game.CurrentPlayer)
            
            if nnScore > minimaxScore {
                betterChoices++
            } else {
                worseChoices++
            }
        }
    }
    
    fmt.Printf("Agreement with minimax: %.2f%%\n", float64(agreements)/float64(numPositions)*100)
    fmt.Printf("Neural network made better choices: %d times\n", betterChoices)
    fmt.Printf("Neural network made worse choices: %d times\n", worseChoices)
}
```

## Understanding Where Neural Networks Fail

Finally, let's analyze where our neural network makes mistakes:

```go
// Analyze failure cases
func analyzeMistakes(network *NeuralNetwork, minimaxAgent agents.Agent, numPositions int) {
    mistakesByPhase := map[string]int{
        "opening": 0,
        "midgame": 0,
        "endgame": 0,
    }
    
    mistakesByPattern := make(map[string]int)
    
    for i := 0; i < numPositions; i++ {
        // Generate a random game position
        game := generateRandomPosition()
        
        // Determine game phase
        phase := getGamePhase(game)
        
        // Get minimax and neural network moves
        minimaxMove, _ := minimaxAgent.GetMove(game)
        
        features := gameStateToFeatures(game)
        predictions := network.Forward(features)
        
        // Get best valid move from network
        bestMove := getBestValidMove(game, predictions)
        
        if bestMove.Position != minimaxMove.Position {
            // Record mistake by phase
            mistakesByPhase[phase]++
            
            // Analyze the pattern of the mistake
            pattern := analyzeMistakePattern(game, minimaxMove, bestMove)
            mistakesByPattern[pattern]++
            
            // Log detailed information for significant mistakes
            if isSignificantMistake(game, minimaxMove, bestMove) {
                fmt.Println("Significant mistake found:")
                printGameState(game)
                fmt.Printf("Minimax chose: %v\n", minimaxMove)
                fmt.Printf("Network chose: %v\n", bestMove)
                fmt.Println("Analysis:", explainMistake(game, minimaxMove, bestMove))
            }
        }
    }
    
    // Print summary statistics
    fmt.Println("Mistakes by game phase:")
    for phase, count := range mistakesByPhase {
        fmt.Printf("  %s: %d\n", phase, count)
    }
    
    fmt.Println("Top mistake patterns:")
    patterns := sortMapByValue(mistakesByPattern)
    for i, pattern := range patterns[:min(5, len(patterns))] {
        fmt.Printf("  %d. %s: %d occurrences\n", i+1, pattern, mistakesByPattern[pattern])
    }
}
```

## Conclusion: Learning by Building

By implementing neural networks from scratch, we gain several insights:

1. **Mathematics in Action**: See how the chain rule of calculus drives backpropagation
2. **Architecture Decisions**: Understand how choices like activation functions affect learning
3. **Performance Characteristics**: Observe how various hyperparameters influence training
4. **Debugging Skills**: Develop techniques for identifying and fixing neural network issues

This hands-on approach may not be the most efficient path to a production system, but it provides invaluable educational experience. The knowledge gained will serve you well when using higher-level frameworks, as you'll understand what's happening "under the hood."

## Next Steps

After mastering the basics of neural networks through manual implementation, consider:

1. **Experimenting** with different architectures (more layers, different sizes)
2. **Implementing** additional optimization techniques (momentum, Adam optimizer)
3. **Adding** regularization methods to prevent overfitting
4. **Comparing** your implementation with frameworks to identify optimizations
5. **Extending** to reinforcement learning using your neural network as a foundation

Building neural networks from scratch is a challenging but rewarding journey that deepens your understanding of machine learning fundamentals. The knowledge gained will help you make better decisions when using high-level frameworks for more complex projects. 