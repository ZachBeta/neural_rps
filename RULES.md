# Neural Network Rock Paper Scissors

This file defines the rules and standards for the Neural RPS project, which includes:

* C++ RPS implementation (legacy and modern)
* Golang RPS implementation  
* AlphaGo-style implementation for RPS card game and Tic-Tac-Toe

## Project Overview
This project explores different implementations of neural networks for playing games:

1. Rock Paper Scissors - Using basic neural networks with reinforcement learning
2. RPS Card Game - Combining RPS with strategic card placement
3. Tic-Tac-Toe - Using AlphaGo-style techniques (MCTS + neural networks)

## Code Style Guidelines
### C++ Implementations
- Use modern C++ features (C++17 or later)
- Follow Google C++ Style Guide
- Use clear, descriptive variable and function names
- Include comments for complex logic
- Write unit tests for core functionality

### Golang Implementations
- Follow standard Go code conventions
- Use idiomatic Go patterns
- Organize code into appropriate packages
- Provide good documentation
- Write comprehensive tests

## Project Structure
```
.
├── cpp_implementation/        # Modern C++ implementation
├── legacy_cpp_implementation/ # Original C++ implementation
├── golang_implementation/     # Go implementation of RPS
├── alphago_demo/              # AlphaGo-style implementations
├── scripts/                   # Utility scripts
├── output/                    # Training output and model data
└── shared_output_format.md    # Output format specification
```

## Build System
- C++ implementations use CMake
- Go implementations use standard Go build tools
- A unified Makefile provides common commands for all implementations

## Neural Network Architectures

### RPS Implementations
- Input layer: 6-9 neurons (one-hot encoding of previous moves)
- Hidden layer: 8-16 neurons with activation (ReLU or Sigmoid)
- Output layer: 3 neurons with softmax activation (rock, paper, scissors probabilities)

### AlphaGo-Style Implementations
- Policy network and value network
- Input: Board state encoding
- Policy output: Move probabilities
- Value output: Win probability estimate
- Combined with Monte Carlo Tree Search (MCTS)

## Standardized Output Format
For comparing implementations, all versions must follow a consistent output format as defined in `shared_output_format.md`:

1. Header & Implementation Info
2. Network Architecture
3. Training Process
4. Model Predictions
5. Model Parameters (Optional)

## Game Definitions

### Rock Paper Scissors
- Traditional game rules (rock beats scissors, scissors beats paper, paper beats rock)
- Neural network plays against various strategies (random, pattern-based, mimic)
- Training uses reinforcement learning with rewards for wins

### RPS Card Game
- Cards are placed on a 3×3 board
- When adjacent, cards capture according to RPS rules:
  - Rock beats Scissors
  - Paper beats Rock
  - Scissors beats Paper
- Players have a limited hand of cards
- Cards are represented as R (Rock), P (Paper), and S (Scissors)
- UPPERCASE letters (R, P, S) represent Player 1's cards
- lowercase letters (r, p, s) represent Player 2's cards
- Game ends when:
  - Both players are out of cards, OR
  - One player has no valid moves, OR
  - Maximum rounds are reached
- **Winner Determination**: The player with the most cards on the board at the end wins
  - Only cards on the board count toward victory (not cards in hand)
  - In case of a tie, the game is a draw

#### Game Balance
- To address the first-mover advantage (similar to white in chess), a full game consists of two rounds
- Players switch positions after the first round (first player becomes second player)
- Final winner is determined by the combined score across both rounds
- If the score is tied after two rounds, an additional tiebreaker round may be played
- For tournament play, players should play an even number of games with alternating positions

### Tic-Tac-Toe
- Traditional 3×3 board
- Players alternate placing X and O
- First to get 3 in a row (horizontally, vertically, or diagonally) wins
- Used to demonstrate AlphaGo-style techniques 

## Testing Strategy

### Unit Tests
- Test core game functionality (moves, rule enforcement, etc.)
- Create controlled board states for testing specific scenarios
- Include edge cases (e.g., game end conditions, captures)
- Test winner determination logic with different board configurations
- Verify score counting is accurate

### System Tests
- Create standalone test programs that validate end-to-end functionality
- Simulate real gameplay situations
- Test both AI vs AI and human vs AI scenarios
- Verify integration with ML components (policy networks, MCTS)

### Example Test Cases
- Board with specific layouts (e.g., Player 1 has more cards)
- Games ending due to all cards being played
- Games ending due to no valid moves
- Edge cases for captures (various RPS combinations)

## Debugging Techniques

### Game State Visualization
- Print the board state after each move
- Use uppercase/lowercase letters to distinguish player ownership
- Show card counts for each player

### Debug Information
- Add debug prints showing score calculation
- Display card ownership information
- Show when captures happen and which cards are affected
- Include round/turn information

### Standalone Tests
- Create small, focused test programs like `test_winner_logic` to verify specific functionality
- Use controlled test scenarios to isolate bugs
- Add verbose debugging only in test versions

### Error Checking
When implementing game logic, verify these critical scenarios:
- Game end detection (all cards played, no valid moves, max rounds)
- Capture mechanics (ensure correct cards are captured)
- Winner determination (proper counting of cards)
- Move validation (ensures only valid moves are allowed)

## Implementation Guidelines

For any game implementation:
1. Start with clear rule definitions
2. Build a solid game state representation
3. Implement core mechanics (moves, captures)
4. Add end-game detection and winner determination
5. Create thorough tests before adding AI components
6. Add debug visualization features before testing with AI 