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
- **cmd/rps_card**: Main application entry point

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

### Building the Game
To build the RPS card game, run the build script:

```bash
cd alphago_demo
./build.sh
```

This will compile the game and create a binary in the `bin` directory.

### Running the Game
After building, you can run the game with:

```bash
cd alphago_demo
./bin/rps_card
```

The game provides two modes:
1. Play against AI - A human vs AI game mode
2. Watch AI vs AI demonstration - An automated demonstration of AI playing against itself

## Running Tests

Run the comprehensive test suite:

```bash
./run_tests.sh
```

## Game Modes

The application offers two modes:

1. **Play against AI**: Challenge the trained AI in a game of RPS
2. **AI vs AI demonstration**: Watch the AI play against itself

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

## Technical Notes

- The neural networks are simplified 2-layer networks with ReLU and softmax/sigmoid activations
- Training uses gradient descent with a simple learning rate
- MCTS uses Upper Confidence Bounds (UCB) to balance exploration and exploitation
- All implementations are designed for educational purposes rather than maximum performance 

## Project Status

All components of the RPS card game have been implemented and tested:

- ✅ Game logic (rules, board representation, card manipulation)
- ✅ Neural network implementation (policy and value networks)
- ✅ Monte Carlo Tree Search implementation
- ✅ Self-play training mechanism
- ✅ Game interface (console-based)
- ✅ Test suite for all components

The game is now ready to play! Build and run the game using the instructions above.

## Conclusion

This Rock-Paper-Scissors card game demonstrates the application of AlphaGo-style techniques to a card-based strategy game. The combination of neural networks and Monte Carlo Tree Search provides an AI opponent that can learn and improve through self-play.

The modular architecture allows for easy extension and modification of the game, such as adding new card types, changing board sizes, or implementing a graphical user interface.

Enjoy playing! 