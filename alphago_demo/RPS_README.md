# AlphaGo-Style Rock-Paper-Scissors Card Game

This is an implementation of a Rock-Paper-Scissors card game with an AlphaGo-style AI. It demonstrates the core principles of AlphaGo/AlphaZero in a more complex game environment compared to Tic-Tac-Toe.

## Game Overview

The RPS card game combines traditional Rock-Paper-Scissors with strategic card placement:

- Players have hands of cards (Rock, Paper, or Scissors)
- Cards are placed on a 3×3 board
- When a card is placed next to an opponent's card, it can capture it according to RPS rules:
  - Rock beats Scissors
  - Paper beats Rock
  - Scissors beats Paper
- The game ends when both players run out of cards or the maximum number of rounds is reached
- The player with the most cards on the board wins

## Architecture

The project follows the same architecture as the Tic-Tac-Toe implementation for an "apples to apples" comparison:

- **pkg/game**: Game logic and board representation
- **pkg/neural**: Neural network implementations (policy and value networks)
- **pkg/mcts**: Monte Carlo Tree Search implementation
- **pkg/training**: Self-play and training logic

## Components

### Neural Networks

1. **Policy Network**: Predicts the probability distribution over possible positions to play
2. **Value Network**: Estimates the probability of winning from a given position

### Monte Carlo Tree Search (MCTS)

MCTS is used to improve the quality of moves by looking ahead and evaluating potential future positions. The neural networks guide the search process.

### Self-Play Training

The system trains by playing against itself, using the results to improve the neural networks:
1. Generate games through self-play using MCTS guided by current neural networks
2. Extract training examples from these games
3. Train neural networks on these examples
4. Repeat with improved networks

## How to Run

### Training Neural Networks

The primary focus of this project is training and comparing different neural network models. You can train the networks with different parameters using these commands:

```bash
# Train with default parameters (small training dataset)
make train-and-compete

# Train with large dataset comparison (100 vs 1000 games)
make train-large-comparison
```

The trained models will be saved to `alphago_demo/output/` directory.

### Model Comparison

After training multiple models with different parameters, the system automatically runs a tournament between them to evaluate performance. The tournament results show which training approach produces better agents.

For more extensive model comparison, you can use the dedicated comparison tool:

```bash
# Compare trained models with 100 games
make compare-models

# Or run directly with custom parameters
cd alphago_demo
./bin/compare_models --games 200 --model1-name "SmallModel" --model2-name "LargeModel" --verbose
```

This tool provides detailed tournament results and saves them to the `alphago_demo/results/` directory for later analysis.

### Running Tests

Run the comprehensive test suite:

```bash
cd alphago_demo
go test -v ./...
```

## Implementation Details

Key differences from the Tic-Tac-Toe implementation:

1. **More Complex Game Logic**: 
   - Cards with different types (Rock, Paper, Scissors)
   - Card captures following RPS rules
   - Limited resources (cards in hand)

2. **Richer State Representation**:
   - 81 input features (9 positions × 9 features per position)
   - Features encode card type, ownership, and current player

3. **Policy Focus**:
   - The policy network predicts which position to play (not which card)
   - Card selection is simplified in this implementation

## Future Improvements

Potential improvements include:
- Adding card selection intelligence to the policy network  
- Implementing a more sophisticated board representation
- Extending to larger boards or more complex card types
- Visualization of the MCTS search tree
- Optimizing for performance with parallel MCTS
- Human vs AI interface (currently out of scope)

## Technical Notes

- The neural networks are simplified 2-layer networks with ReLU and softmax/sigmoid activations
- Training uses gradient descent with a simple learning rate
- MCTS uses Upper Confidence Bounds (UCB) to balance exploration and exploitation
- All implementations are designed for educational purposes rather than maximum performance 

## Project Status

All core components of the RPS card game have been implemented and tested:

- ✅ Game logic (rules, board representation, card manipulation)
- ✅ Neural network implementation (policy and value networks)
- ✅ Monte Carlo Tree Search implementation
- ✅ Self-play training mechanism
- ✅ Model comparison through tournaments
- ✅ Test suite for all components

## Conclusion

This Rock-Paper-Scissors card game demonstrates the application of AlphaGo-style techniques to a card-based strategy game. The combination of neural networks and Monte Carlo Tree Search provides AI agents that can learn and improve through self-play.

The modular architecture allows for easy extension and modification of the game, such as adding new card types, changing board sizes, or implementing different neural network architectures. 