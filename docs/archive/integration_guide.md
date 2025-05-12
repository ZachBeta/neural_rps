# Neural RPS Integration Guide

This guide explains how to integrate components from different packages in the Neural RPS project, with a focus on using AlphaGo neural networks from the `alphago_demo` package in tournaments within the `golang_implementation` package.

## 1. Overview

The primary integration scenario in this project is using neural networks trained in the `alphago_demo` package with the tournament system in the `golang_implementation` package. This allows for comparing AlphaGo-style agents with PPO and other agent types.

## 2. Setting Up Integration

### 2.1. Ensure Dependencies are Installed

First, make sure you have all dependencies installed:

```bash
# Install all required dependencies
make install-deps
```

### 2.2. Train AlphaGo Models

Before integration, you need trained AlphaGo neural network models. Train them using:

```bash
# From the project root
make alphago-train
```

This will:
1. Build and run the training script
2. Generate self-play training data
3. Train policy and value networks
4. Save the trained models to `alphago_demo/output/`

## 3. Using AlphaGo Models in Tournaments

### 3.1. Integration via AlphaGoAgent Adapter

The `AlphaGoAgent` adapter in `golang_implementation/pkg/agent/alphago_agent.go` provides the bridge between packages:

```go
// Usage example:
import (
    "github.com/zachbeta/neural_rps/pkg/agent"
)

// Load neural networks from alphago_demo
policyNet, valueNet, err := agent.LoadAlphaGoNetworksFromFile(
    "../alphago_demo/output/rps_policy2.model",
    "../alphago_demo/output/rps_value2.model",
)
if err != nil {
    log.Fatalf("Failed to load networks: %v", err)
}

// Create AlphaGo agent using the loaded networks
alphagoAgent := agent.NewAlphaGoAgent(
    "AlphaGo-Agent",
    policyNet,
    valueNet,
    200,  // MCTS simulations
    1.0,  // Exploration constant
)

// Use the agent in tournaments, matches, etc.
```

### 3.2. Running a Tournament

Run a tournament between an AlphaGo agent and a PPO agent:

```bash
# From the project root
make golang-tournament
```

This will:
1. Build the tournament binary
2. Load AlphaGo neural networks from `alphago_demo/output/`
3. Create both AlphaGo and PPO agents
4. Run a tournament between them
5. Save results to `golang_implementation/results/`

### 3.3. Comparing Different AlphaGo Configurations

You can also compare different AlphaGo configurations:

```bash
# From the project root
make golang-vs-alphago
```

This will run a tournament between small and large AlphaGo models with different MCTS parameters.

## 4. Understanding Game State Conversion

The most important part of the integration is the game state conversion between formats. This is handled by the `convertToAlphaGoGame` function in the `AlphaGoAgent`:

```go
// Game state conversion
func convertToAlphaGoGame(ourGame *game.RPSCardGame) *alphaGame.RPSGame {
    // Create a new AlphaGo game state
    alphaGameState := alphaGame.NewRPSGame(
        ourGame.DeckSize,
        ourGame.HandSize,
        ourGame.MaxRounds,
    )

    // Copy board state
    for pos := 0; pos < 9; pos++ {
        // ... conversion logic ...
    }

    // Copy hands
    // ... conversion logic ...

    // Copy current player and round
    // ... conversion logic ...

    return alphaGameState
}
```

This function handles:
1. Creating a new AlphaGo game state
2. Copying the board configuration
3. Copying player hands
4. Copying current player and round information

## 5. Custom Integration Examples

### 5.1. Creating a Custom AlphaGo Agent

You can create a custom AlphaGo agent with different parameters:

```go
// Custom AlphaGo agent
customAgent := agent.NewAlphaGoAgent(
    "Custom-AlphaGo",
    policyNet,
    valueNet,
    500,  // More MCTS simulations for deeper search
    1.5,  // Higher exploration constant
)
```

### 5.2. Using Different Model Files

If you've trained multiple models, you can use different model files:

```go
// Load alternative models
policyNet, valueNet, err := agent.LoadAlphaGoNetworksFromFile(
    "../alphago_demo/output/rps_policy_alternative.model",
    "../alphago_demo/output/rps_value_alternative.model",
)
```

### 5.3. Integration in Your Own Code

To integrate these components in your own code:

```go
package main

import (
    "fmt"
    "log"

    "github.com/zachbeta/neural_rps/pkg/agent"
    "github.com/zachbeta/neural_rps/pkg/game"
)

func main() {
    // Load AlphaGo models
    policyNet, valueNet, err := agent.LoadAlphaGoNetworksFromFile(
        "../alphago_demo/output/rps_policy2.model",
        "../alphago_demo/output/rps_value2.model",
    )
    if err != nil {
        log.Fatalf("Failed to load models: %v", err)
    }

    // Create an AlphaGo agent
    alphagoAgent := agent.NewAlphaGoAgent(
        "AlphaGo",
        policyNet,
        valueNet,
        200,
        1.0,
    )

    // Create a PPO agent
    ppoAgent := agent.NewRPSPPOAgent(
        "PPO-Agent",
        128, // Hidden layer size
    )

    // Create a game instance
    gameInstance := game.NewRPSCardGame(21, 5, 10)

    // Get moves from both agents
    alphagoMove, _ := alphagoAgent.GetMove(gameInstance)
    ppoMove, _ := ppoAgent.GetMove(gameInstance)

    fmt.Printf("AlphaGo agent move: %v\n", alphagoMove)
    fmt.Printf("PPO agent move: %v\n", ppoMove)
}
```

## 6. Troubleshooting

### 6.1. Model Loading Errors

If you encounter errors loading AlphaGo models:

1. Ensure the model files exist in the expected location
2. Check file permissions
3. Verify the models were trained correctly
4. Ensure the path is relative to where the binary is run

Example fix:
```go
// Use absolute paths if relative paths are causing issues
policyPath := "/absolute/path/to/alphago_demo/output/rps_policy2.model"
valuePath := "/absolute/path/to/alphago_demo/output/rps_value2.model"
```

### 6.2. Game State Conversion Issues

If you encounter issues with game state conversion:

1. Enable debug output to see the conversion process
2. Verify both game state structures are compatible
3. Check for any initialization issues in the AlphaGo game state

### 6.3. Dependencies Between Packages

If you're modifying the code and encounter dependency issues:

1. Ensure `go.mod` files are up-to-date
2. Run `go mod tidy` in both packages
3. Check import paths for consistency

## 7. Advanced Topics

### 7.1. Custom Metrics and Logging

You can add custom metrics and logging to track the performance of your agents:

```go
// In your tournament code
type MatchResult struct {
    AlphaGoMoves []game.RPSCardMove
    PPOMoves     []game.RPSCardMove
    Winner       string
    Positions    map[int]int  // Position -> win count
}

// Track and analyze results
```

### 7.2. Extending the AlphaGoAgent

You can extend the `AlphaGoAgent` with additional functionality:

```go
// Extended AlphaGo agent with analytics
type AnalyticsAlphaGoAgent struct {
    *agent.AlphaGoAgent
    moveStats map[int]int  // Position -> selection count
}

func (a *AnalyticsAlphaGoAgent) GetMove(state *game.RPSCardGame) (game.RPSCardMove, error) {
    move, err := a.AlphaGoAgent.GetMove(state)
    if err == nil {
        a.moveStats[move.Position]++
    }
    return move, err
}
```

## 8. Conclusion

This integration guide demonstrates how to use components from different packages in the Neural RPS project. By leveraging the adapter pattern and clean interfaces, you can easily compare different agent types and neural network approaches within the same tournament framework.

For further details:
- See the architecture document: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Check the refactoring plan: [REFACTORING_PLAN.md](../REFACTORING_PLAN.md)
- Review package-specific documentation in each package's README.md 