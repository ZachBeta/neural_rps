# Neural RPS Shared Interfaces

This document defines the shared interfaces that should be used consistently across packages in the Neural RPS project. Using these common interfaces ensures that components from different packages can work together seamlessly.

## 1. Agent Interface

The Agent interface represents any entity that can play a game by selecting moves.

```go
// Agent represents any entity that can select moves in a game
type Agent interface {
    // Name returns the agent's name
    Name() string
    
    // GetMove returns the best move according to the agent's strategy
    GetMove(gameState GameState) (Move, error)
}
```

### 1.1. Implementation Guidelines

When implementing this interface:

- `Name()` should return a descriptive, unique name for the agent
- `GetMove()` should return a valid move or an error if no valid moves exist
- Implementations should handle both their native game state types and perform conversions as needed
- If an agent needs access to its own internal state, it should maintain that state as fields

### 1.2. Adapter Pattern

When adapting agents between packages, use the adapter pattern:

```go
// Adapter for an AlphaGo agent
type AlphaGoAgentAdapter struct {
    // Original agent
    alphagoAgent *alphago.Agent
    
    // Additional fields as needed
    name string
}

// Implement the Agent interface
func (a *AlphaGoAgentAdapter) Name() string {
    return a.name
}

func (a *AlphaGoAgentAdapter) GetMove(gameState GameState) (Move, error) {
    // Convert gameState to alphago's format
    alphagoState := convertToAlphaGoState(gameState)
    
    // Get move from the original agent
    alphagoMove := a.alphagoAgent.GetMove(alphagoState)
    
    // Convert move back to our format
    return convertFromAlphaGoMove(alphagoMove), nil
}
```

## 2. Game State Interface

The GameState interface represents the current state of a game.

```go
// GameState represents the current state of a game
type GameState interface {
    // Copy returns a deep copy of this game state
    Copy() GameState
    
    // GetValidMoves returns all valid moves from this state
    GetValidMoves() []Move
    
    // MakeMove applies a move to this game state
    MakeMove(move Move) error
    
    // IsGameOver checks if the game is over
    IsGameOver() bool
    
    // GetWinner returns the winner of the game (if any)
    GetWinner() Player
}
```

### 2.1. Implementation Guidelines

When implementing this interface:

- `Copy()` should create a completely independent deep copy
- `GetValidMoves()` should return all legal moves from the current state
- `MakeMove()` should modify the state in-place and return an error for invalid moves
- `IsGameOver()` should return true when the game has reached a terminal state
- `GetWinner()` should return the winner or a "no winner" value for draws or ongoing games

### 2.2. Conversion Functions

When converting between different game state representations, use explicit conversion functions:

```go
// Convert from one game state format to another
func convertToAlphaGoState(gameState *RPSCardGame) *alphaGame.RPSGame {
    // Create a new state in the target format
    alphaGameState := alphaGame.NewRPSGame(...)
    
    // Copy data between formats
    // ...
    
    return alphaGameState
}
```

## 3. Move Interface

The Move interface represents an action a player can take in a game.

```go
// Move represents an action a player can take
type Move interface {
    // GetPlayer returns the player making the move
    GetPlayer() Player
    
    // IsValid checks if the move is valid in the given state
    IsValid(state GameState) bool
    
    // String returns a string representation
    String() string
}
```

### 3.1. Implementation Guidelines

For game-specific move types, consider using structs with the necessary fields:

```go
// RPSCardMove represents a move in the RPS card game
type RPSCardMove struct {
    CardIndex int    // Index of the card in hand
    Position  int    // Position on the board
    Player    Player // Player making the move
}
```

## 4. Neural Network Model Interfaces

### 4.1. Policy Network Interface

```go
// PolicyNetwork represents a neural network that predicts move probabilities
type PolicyNetwork interface {
    // Predict returns move probabilities given a game state
    Predict(gameState GameState) []float64
    
    // Train trains the network on a batch of examples
    Train(examples []TrainingExample, epochs int, learningRate float64) error
    
    // Save saves the network weights to a file
    SaveToFile(path string) error
    
    // Load loads the network weights from a file
    LoadFromFile(path string) error
}
```

### 4.2. Value Network Interface

```go
// ValueNetwork represents a neural network that evaluates game states
type ValueNetwork interface {
    // Predict returns a value estimation for a game state
    Predict(gameState GameState) float64
    
    // Train trains the network on a batch of examples
    Train(examples []TrainingExample, epochs int, learningRate float64) error
    
    // Save saves the network weights to a file
    SaveToFile(path string) error
    
    // Load loads the network weights from a file
    LoadFromFile(path string) error
}
```

## 5. Tournament Interface

```go
// Tournament represents a competition between agents
type Tournament interface {
    // AddAgent adds an agent to the tournament
    AddAgent(agent Agent)
    
    // Run runs the tournament for a specified number of games
    Run(numGames int) (Results, error)
    
    // GetResults returns the tournament results
    GetResults() Results
}

// Results represents tournament outcomes
type Results interface {
    // GetWinner returns the winner of the tournament
    GetWinner() Agent
    
    // GetWinRate returns the win rate for an agent
    GetWinRate(agent Agent) float64
    
    // GetStats returns detailed statistics
    GetStats() map[string]interface{}
}
```

## 6. Implementation Notes

### 6.1. Package Structure

These interfaces should be defined at appropriate locations in the project:

- Agent interfaces in `golang_implementation/pkg/agent`
- Game state interfaces in `golang_implementation/pkg/game`
- Tournament interfaces in `golang_implementation/pkg/tournament`

### 6.2. Type Conversions

When converting between types in different packages:

- Use explicit conversion functions rather than type assertions
- Document the mapping between different representations
- Handle error cases for incompatible states

### 6.3. Error Handling

All methods that can fail should return an error:

```go
// Preferred
func (a *Agent) GetMove(state GameState) (Move, error)

// Avoid
func (a *Agent) GetMove(state GameState) Move
```

## 7. Future Extensions

As the project evolves, consider these additional interfaces:

- Serialization interfaces for saving/loading game states
- Visualization interfaces for rendering game states
- Analysis interfaces for evaluating agent performance
- Configuration interfaces for parameter management

## 8. Conclusion

By following these interface guidelines, components from different packages in the Neural RPS project can interact seamlessly. This allows for comparing different agent types and neural network approaches within the same tournament framework.

For integration examples, see [integration_guide.md](integration_guide.md). 