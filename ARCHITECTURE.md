# Neural RPS Project Architecture

This document describes the high-level architecture of the Neural RPS project, focusing on the different implementations and their integration points.

## 1. Overview

The Neural RPS project consists of multiple implementations of neural network-based approaches to playing Rock Paper Scissors and related games. Each implementation uses different techniques, languages, and architectures:

- **Legacy C++ Implementation**: Original implementation using PPO
- **C++ Implementation**: Simplified demonstration version
- **Golang Implementation**: PPO-based implementation with improved architecture
- **AlphaGo Demo**: AlphaGo-style MCTS + neural networks for Tic-Tac-Toe and RPS Card Game

## 2. Package Structure

```
neural_rps/
├── legacy_cpp_implementation/   # Original C++ implementation
├── cpp_implementation/          # Simplified C++ demo
├── golang_implementation/       # Go implementation with PPO
├── alphago_demo/               # AlphaGo-style implementations
├── output/                     # Shared output directory
└── docs/                       # Project documentation
```

## 3. Key Components

### 3.1. Game Environments

Different game environments are implemented across the packages:

1. **Traditional RPS** (legacy_cpp_implementation, cpp_implementation, golang_implementation)
   - Simple game state: rock (0), paper (1), scissors (2)
   - Direct neural prediction of optimal move

2. **RPS Card Game** (alphago_demo, golang_implementation)
   - 3x3 board with RPS cards
   - Card capture mechanics based on RPS rules
   - Strategic placement and hand management

3. **Tic-Tac-Toe** (alphago_demo)
   - Classic 3x3 grid game
   - Used to demonstrate AlphaGo techniques

### 3.2. Neural Network Approaches

Multiple neural network approaches are implemented:

1. **PPO (Proximal Policy Optimization)**
   - Implemented in legacy_cpp_implementation, golang_implementation
   - Direct learning of policy from gameplay
   - Single network architecture

2. **AlphaGo-Style**
   - Implemented in alphago_demo
   - Dual network architecture: policy network + value network
   - Combined with Monte Carlo Tree Search
   - Self-play training methodology

### 3.3. Agent Types

The project implements several types of agents:

1. **Random Agent**: Makes random valid moves
2. **PPO Agent**: Uses PPO neural network to select moves
3. **AlphaGo Agent**: Uses MCTS + neural networks to select moves
4. **Human Agent**: Allows human players to participate (in some implementations)

## 4. Integration Points

The primary integration between packages occurs between `golang_implementation` and `alphago_demo`:

```
                 ┌────────────────────┐
                 │                    │
                 │   alphago_demo     │
                 │                    │
                 └─────────┬──────────┘
                           │
                           │ Exports
                           │
                           ▼
┌───────────────────────────────────────────┐
│                                           │
│  ┌─────────────────┐    ┌──────────────┐  │
│  │                 │    │              │  │
│  │  Neural Network │◄───┤ AlphaGoAgent │  │
│  │  Models         │    │ Adapter      │  │
│  │                 │    │              │  │
│  └─────────────────┘    └──────┬───────┘  │
│                                │          │
│                                ▼          │
│                        ┌───────────────┐  │
│                        │               │  │
│                        │  Tournament   │  │
│                        │  System       │  │
│                        │               │  │
│                        └───────────────┘  │
│                                           │
│         golang_implementation             │
└───────────────────────────────────────────┘
```

### 4.1. Model Loading

The `golang_implementation` can load neural network models trained by `alphago_demo`:

```go
// In golang_implementation/pkg/agent/alphago_agent.go
policyNet, valueNet, err := LoadAlphaGoNetworksFromFile(policyPath, valuePath)
```

### 4.2. Game State Conversion

The AlphaGoAgent adapter converts between game state formats:

```go
// In golang_implementation/pkg/agent/alphago_agent.go
func convertToAlphaGoGame(ourGame *game.RPSCardGame) *alphaGame.RPSGame {
    // ... conversion logic ...
}
```

### 4.3. Agent Interface

Both implementations implement a common agent interface pattern:

```go
// Common pattern across packages
type Agent interface {
    Name() string
    GetMove(gameState GameState) (Move, error)
}
```

## 5. Build and Execution Flow

### 5.1. AlphaGo Model Training

```
1. alphago_demo/cmd/train_models/main.go
   ├── Creates neural networks
   ├── Runs self-play to generate training data
   ├── Trains networks on the generated data
   └── Saves trained models to alphago_demo/output/
```

### 5.2. Tournament Execution

```
1. golang_implementation/cmd/tournament/main.go or alphago_vs_alphago/main.go
   ├── Loads trained models from alphago_demo/output/
   ├── Creates agent instances (PPO, AlphaGo, etc.)
   ├── Runs games between agents
   └── Records and analyzes results
```

## 6. Key Design Patterns

### 6.1. Adapter Pattern

The AlphaGoAgent serves as an adapter between the alphago_demo neural networks and the golang_implementation tournament system:

```go
// Adapter pattern in alphago_agent.go
type AlphaGoAgent struct {
    // alphago_demo components
    policyNetwork *neural.RPSPolicyNetwork
    valueNetwork  *neural.RPSValueNetwork
    mctsEngine    *mcts.RPSMCTS
    
    // Additional fields
    name        string
    simulations int
    exploration float64
}
```

### 6.2. Strategy Pattern

Different agent types implement the same interface but use different move selection strategies:

```go
// Strategy pattern - different implementations of GetMove
func (a *RandomAgent) GetMove(state *game.RPSCardGame) (game.RPSCardMove, error) {
    // Random selection strategy
}

func (a *PPOAgent) GetMove(state *game.RPSCardGame) (game.RPSCardMove, error) {
    // PPO network-based selection strategy
}

func (a *AlphaGoAgent) GetMove(state *game.RPSCardGame) (game.RPSCardMove, error) {
    // MCTS + neural network selection strategy
}
```

## 7. Cross-Package Dependencies

Here are the key cross-package dependencies:

1. **golang_implementation → alphago_demo**
   - Neural network model loading
   - MCTS algorithm usage
   - Game state conversion

2. **No dependencies in the reverse direction**
   - alphago_demo does not depend on golang_implementation
   - This maintains a clean dependency hierarchy

## 8. Extension Points

The architecture supports several extension points:

1. **New Agent Types**: Additional agent implementations can be added by implementing the Agent interface
2. **New Game Environments**: New games can be added following the GameState interface pattern
3. **Alternative Neural Network Architectures**: Different network designs can be implemented
4. **Additional Training Methods**: New training algorithms can be added

## 9. Future Architecture Considerations

For future development, these architectural changes should be considered:

1. **Shared Interface Package**: Move common interfaces to a shared package
2. **Plugin Architecture**: Allow dynamically loading different agent implementations
3. **Distributed Training**: Support for distributed training of neural networks
4. **Web Interface**: Add a web-based visualization and interaction layer

## 10. Conclusion

The Neural RPS project architecture demonstrates multiple approaches to implementing neural network-based game-playing agents. The integration between packages is handled through well-defined adapters and interfaces, allowing different implementation techniques to be compared within the same tournament framework. 