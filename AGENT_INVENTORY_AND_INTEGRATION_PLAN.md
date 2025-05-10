# Agent Inventory and Integration Plan

This document catalogs the existing agent implementations in our codebase and outlines our plan for integrating our new neural network implementation.

## Current Agent Implementations

### Search-Based Agents

#### Minimax Agent
- **Location**: `alphago_demo/pkg/agents/minimax_agent.go`
- **Features**:
  - Configurable search depth
  - Adjustable time limit
  - Optional transposition table for caching
  - Implements the standard Agent interface
- **Usage**: Primary baseline for optimal play at various depths

#### MCTS (Monte Carlo Tree Search)
- **Implementation**: Not as a standalone agent, but as a component
- **Location**: `alphago_demo/pkg/mcts/search.go` and `node.go`
- **Features**:
  - Used within the AlphaGo-style agent
  - Configurable number of simulations
  - Tunable exploration constant

### Neural Network Agents

#### Basic Neural Agent
- **Location**: `alphago_demo/pkg/neural/agent.go`
- **Features**:
  - Uses policy network only
  - Selects moves based on highest probability
  - Simple implementation without search

#### AlphaGo-Style Agent
- **Location**: `golang_implementation/pkg/agent/alphago_agent.go`
- **Features**:
  - Combines neural networks with MCTS
  - Uses both policy and value networks
  - Includes state conversion between different game representations
  - More sophisticated decision making with tree search

#### From-Scratch Neural Network (Our Implementation)
- **Location**: `neural_from_scratch/`
- **Status**: Implemented but not yet integrated as an agent
- **Features**:
  - Manual implementation of feedforward neural network
  - Custom backpropagation
  - No dependencies on ML libraries
  - Educational tool for understanding neural networks

### Reinforcement Learning Agents

#### PPO Agent
- **Location**: `golang_implementation/pkg/agent/rps_ppo_agent.go`
- **Features**:
  - Policy gradient implementation
  - Can be trained through self-play
  - Has methods for saving/loading weights

### Other Agents

#### Random Agent
- **Implementation**: Not as a standalone class, but as functionality
- **Usage**: Referenced with `GetRandomMove()` in game implementations
- **Purpose**: Serves as a baseline for comparison

### Legacy C++ Implementations
- **Locations**: `cpp_implementation/` and `legacy_cpp_implementation/`
- **Contents**: Includes implementations of PPO and potentially other agents
- **Status**: Likely superseded by Go implementations

## Agent Interface

The standard Agent interface is defined in multiple locations but generally follows:

```go
type Agent interface {
    Name() string
    GetMove(state *game.RPSGame) (game.RPSMove, error)
}
```

## Integration Plan for ScratchNN

### 1. ScratchNN Agent Creation

We'll create a new agent implementation that uses our from-scratch neural network:

```go
// neural_from_scratch/scratch_agent.go
package neural

import (
    "github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// ScratchNNAgent implements the Agent interface using our from-scratch neural network
type ScratchNNAgent struct {
    name           string
    network        *NeuralNetwork
    explorationTemp float64
}

// NewScratchNNAgent creates a new agent using our from-scratch neural network
func NewScratchNNAgent(name string, modelPath string, explorationTemp float64) (*ScratchNNAgent, error) {
    // Load the pre-trained model
    network, err := LoadModel(modelPath)
    if err != nil {
        return nil, err
    }
    
    return &ScratchNNAgent{
        name:           name,
        network:        network,
        explorationTemp: explorationTemp,
    }, nil
}

// GetMove implements the Agent interface
func (a *ScratchNNAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
    // Convert game state to features
    features := gameStateToFeatures(state)
    
    // Get move probabilities from neural network
    probabilities := a.network.Forward(features)
    
    // Select best valid move
    validMoves := state.GetValidMoves()
    bestMove, _ := selectBestMove(validMoves, probabilities, a.explorationTemp)
    
    return bestMove, nil
}

// Name returns the agent's name
func (a *ScratchNNAgent) Name() string {
    return a.name
}
```

### 2. Parallelization Implementation

To parallelize our neural network, we'll follow the approach outlined in our previous document:

1. **CPU Parallelization**: Implement data-parallel training using goroutines
2. **Batch Processing**: Add batch processing capabilities for improved performance

### 3. Tournament Integration

We'll integrate our agent into the existing tournament system:

```go
// In tournament code
func RunTournament() {
    // Create other agents (existing code)
    randomAgent := &agents.RandomAgent{Name: "Random"}
    minimaxAgent := agents.NewMinimaxAgent("Minimax-D3", 3, time.Second*5, true)
    
    // Create our from-scratch neural agent
    scratchAgent, err := neural.NewScratchNNAgent(
        "ScratchNN-128H-Sup", 
        "models/scratch_nn_128h.model", 
        0.0,
    )
    if err != nil {
        log.Fatalf("Failed to create scratch neural agent: %v", err)
    }
    
    // List of all participants (including our new agent)
    participants := []agents.Agent{
        randomAgent,
        minimaxAgent,
        scratchAgent,
        // Other existing agents...
    }
    
    // Run tournament with all agents
    results := runRoundRobinTournament(participants)
    printTournamentResults(results)
}
```

## Naming Convention

Based on our analysis, we propose the following naming convention for agents:

```
[Framework]-[Architecture]-[Training]-[Size]
```

Examples:
- `Minimax-D3` (Minimax with depth 3)
- `ScratchNN-128H-Sup` (From scratch neural network, 128 hidden neurons, supervised learning)
- `PPO-64H-10K` (PPO agent, 64 hidden neurons, trained for 10K episodes)
- `AlphaRPS-128H-1K` (AlphaGo-style agent, 128 hidden neurons, 1000 MCTS simulations)

## Next Steps and Timeline

1. **ScratchNN Agent Creation** (Day 1-2)
   - Implement the agent wrapper
   - Create game state to feature conversion

2. **Batch Processing** (Day 2-3)
   - Add batch forward pass
   - Implement efficient mini-batch training

3. **CPU Parallelization** (Day 3-5)
   - Implement data-parallel training
   - Add gradient averaging

4. **Tournament Integration** (Day 5-6)
   - Set up tournament fixtures
   - Run comparison benchmarks

5. **Documentation and Analysis** (Day 6-7)
   - Document performance results
   - Compare with other agent types
   - Analyze strengths and weaknesses 