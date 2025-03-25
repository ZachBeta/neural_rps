# Neural Rock Paper Scissors

A C++ implementation of a neural network that learns to play Rock Paper Scissors against human players. The neural network learns from the player's moves and attempts to predict and counter their patterns.

## Features

- Feed-forward neural network with one hidden layer
- Real-time learning from player moves
- Interactive command-line interface
- Save and load model weights
- Unit tests using Google Test
- Static code analysis with clang-tidy

## Prerequisites

### Required Tools
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.14 or higher
- Git

### Required Libraries
- Eigen3 library (linear algebra)
- Google Test framework (testing)
- LLVM/clang-tidy (static analysis, optional)

### Installing Dependencies

#### macOS (using Homebrew)
```bash
# Install build tools
brew install cmake

# Install required libraries
brew install eigen
brew install googletest

# Install clang-tidy (optional, for static analysis)
brew install llvm
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Ubuntu/Debian
```bash
# Install build tools
sudo apt update
sudo apt install build-essential cmake git

# Install required libraries
sudo apt install libeigen3-dev
sudo apt install libgtest-dev

# Install clang-tidy (optional, for static analysis)
sudo apt install clang-tidy
```

#### Windows (using vcpkg)
```powershell
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install

# Install required libraries
vcpkg install eigen3:x64-windows
vcpkg install gtest:x64-windows

# Install clang-tidy (optional, for static analysis)
vcpkg install llvm:x64-windows
```

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural_rps.git
cd neural_rps
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure with CMake:
```bash
# Basic configuration
cmake ..

# Or with vcpkg on Windows
cmake -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake ..
```

4. Build the project:
```bash
# On Unix-like systems
make

# On Windows with Visual Studio
cmake --build . --config Release
```

5. Run the tests:
```bash
ctest --output-on-failure
```

## Code Quality

This project uses clang-tidy for static code analysis. The configuration is in `.clang-tidy` at the root of the project.

To run clang-tidy manually on a specific file:
```bash
clang-tidy path/to/file.cpp
```

Common issues that clang-tidy checks for:
- Modern C++ practices
- Potential bugs and memory issues
- Style guide conformance
- Performance improvements

## Running the Game

After building, you can run the game from the build directory:
```bash
# On Unix-like systems
./src/neural_rps

# On Windows
.\src\Release\neural_rps.exe
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
  - `Card.hpp` - Card class definition
  - `Environment.hpp` - Game environment
  - `PPOAgent.hpp` - Reinforcement learning agent
- `src/` - Source files
  - `main.cpp` - Game entry point
- `tests/` - Unit tests
  - `test_card.cpp` - Card class tests
  - `test_environment.cpp` - Environment tests
  - `test_ppo_agent.cpp` - Agent tests
- `data/` - Directory for saved model weights
- `.clang-tidy` - Clang-tidy configuration
- `CMakeLists.txt` - CMake build configuration

## Neural Network Architecture

- Input layer: 6 neurons (3 for player's last move, 3 for AI's last move)
- Hidden layer: 12 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (probabilities for Rock, Paper, Scissors)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`ctest --output-on-failure`)
5. Run clang-tidy on modified files
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is open source and available under the MIT License.