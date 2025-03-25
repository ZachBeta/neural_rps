# Neural Rock Paper Scissors

A C++ implementation of a neural network that learns to play Rock Paper Scissors against human players. The neural network learns from the player's moves and attempts to predict and counter their patterns.

## Features

- Feed-forward neural network with one hidden layer
- Real-time learning from player moves
- Interactive command-line interface
- Save and load model weights
- Unit tests using Google Test

## Prerequisites

- C++17 compatible compiler
- CMake 3.14 or higher
- Eigen3 library
- Google Test framework

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

4. Run the tests:
```bash
ctest
```

## Running the Game

After building, you can run the game from the build directory:
```bash
./src/neural_rps
```

## How to Play

1. Start the game
2. Enter your move using:
   - 'R' for Rock
   - 'P' for Paper
   - 'S' for Scissors
   - 'Q' to quit
3. The AI will respond with its move and the winner will be announced
4. The neural network learns from each game, improving its strategy over time

## Project Structure

- `include/` - Header files
- `src/` - Source files
- `tests/` - Unit tests
- `data/` - Directory for saved model weights

## Neural Network Architecture

- Input layer: 6 neurons (3 for player's last move, 3 for AI's last move)
- Hidden layer: 12 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (probabilities for Rock, Paper, Scissors)

## License

This project is open source and available under the MIT License.