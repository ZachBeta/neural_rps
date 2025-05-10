# Supervised Learning Tutorial for Neural Network RPS

## Introduction to Supervised Learning for Game AI

This tutorial walks through building a complete supervised learning pipeline to train neural networks for the Rock-Paper-Scissors card game. We'll use the minimax algorithm as our "teacher" to generate optimal training data.

## Overview of the Pipeline

Our supervised learning approach consists of four stages:

1. **Data Generation**: Create training positions with minimax-determined optimal moves
2. **Preprocessing**: Convert game positions into neural network-friendly inputs
3. **Training**: Train neural networks with supervised learning on minimax data
4. **Evaluation**: Measure how well our neural networks perform against baselines

## Prerequisites

- Go programming language (1.16+)
- Basic understanding of neural networks
- Familiarity with game trees and minimax search

## Stage 1: Data Generation

We begin by generating high-quality training data using minimax search to find optimal moves.

```go
// Generate training data using minimax search
go run cmd/generate_training_data/main.go --positions 5000 --depth 5 --output training_data.json
```

The data generation process:
1. Creates random game positions
2. Uses minimax search to determine the optimal move
3. Stores the position and best move as a training example

Each training example contains:
- Board state (positions of cards)
- Player hands (available cards)
- Current player
- Best move determined by minimax
- Evaluation score

## Stage 2: Preprocessing

Next, we preprocess the generated data into a format suitable for neural networks:

```go
// Preprocess the data for neural network training
go run cmd/preprocess_data/main.go --input data/training_data.json --output-dir data
```

The preprocessing steps:
1. Split data into training (80%), validation (10%), and test (10%) sets
2. Convert game states into feature vectors:
   - One-hot encoding for board positions (empty, P1-R, P1-P, P1-S, P2-R, P2-P, P2-S)
   - Normalized card counts for player hands
   - Current player encoding
3. Create target vectors (one-hot encoding of best move)

## Stage 3: Training

With preprocessed data, we train our neural network:

```go
// Train the neural network on preprocessed data
go run cmd/train_supervised/main.go --hidden 128 --epochs 50 --output supervised
```

Training details:
- A neural network with 81 input features and 9 output nodes (representing moves)
- Cross-entropy loss to measure prediction error
- Stochastic gradient descent for optimization
- Early stopping to prevent overfitting

## Stage 4: Evaluation

Finally, we evaluate our trained model:

```go
// Evaluate the trained model against baselines
go run cmd/evaluate_model/main.go --model models/supervised_policy.model --games 100
```

Evaluation metrics:
1. **Win rate against random player**: Measures basic competence
2. **Win rate against minimax**: Measures strategic strength
3. **Move agreement with minimax**: Percentage of moves matching minimax choices
4. **Estimated ELO rating**: Approximate playing strength

## Understanding the Code

### Data Generation

The key function in data generation is creating examples from game positions:

```go
// Create a training example from a game state and minimax move
func createTrainingExample(g *game.RPSGame, move game.RPSMove, depth int) TrainingExample {
    // Convert game state to training example structure
    boardState := convertBoardState(g.Board)
    player1Hand := encodeHand(g.Player1Hand)
    player2Hand := encodeHand(g.Player2Hand)
    
    return TrainingExample{
        BoardState:    boardState,
        Player1Hand:   player1Hand,
        Player2Hand:   player2Hand,
        CurrentPlayer: playerToInt(g.CurrentPlayer),
        BestMove:      move.Position,
        GamePhase:     getGamePhase(g),
        SearchDepth:   depth,
    }
}
```

### Neural Network Architecture

Our model uses a simple fully-connected architecture:

```go
// Policy network for predicting moves
type RPSPolicyNetwork struct {
    inputLayer  *Layer
    hiddenLayer *Layer
    outputLayer *Layer
}

// Create a new policy network with specified hidden size
func NewRPSPolicyNetwork(hiddenSize int) *RPSPolicyNetwork {
    return &RPSPolicyNetwork{
        inputLayer:  NewLayer(81, hiddenSize),
        hiddenLayer: NewLayer(hiddenSize, 9),
        outputLayer: nil, // Output directly from hidden layer
    }
}
```

## Advanced Topics

Once you have the basic pipeline working, consider these improvements:

1. **Data augmentation**: Generate more training examples through symmetry and rotations
2. **Architecture tuning**: Experiment with different network sizes and structures
3. **Progressive learning**: Start with easy positions and gradually increase difficulty
4. **Reinforcement learning**: Use the supervised model as a starting point for self-play

## Common Issues and Solutions

- **Overfitting**: If the model performs well on training data but poorly on validation, reduce network size or add regularization
- **Poor generalization**: If the model doesn't play well against real opponents, generate more diverse training positions
- **Training instability**: If training loss fluctuates wildly, reduce the learning rate

## Conclusion

This supervised learning approach demonstrates how to create a neural network that can learn game strategy from expert (minimax) play. While this approach has limitations (it can only be as good as its teacher), it provides a strong foundation for more advanced techniques like reinforcement learning.

The complete code for this tutorial is available in the repository, with each stage implemented as separate commands for clarity and modularity. 