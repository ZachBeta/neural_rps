# RPS Card Game Server

This is a game server implementation for allowing different agent implementations to compete against each other in the Rock Paper Scissors card game.

## Overview

The game server provides:

1. A standardized API for different agent implementations
2. Game state management following the RPS card game rules
3. Tournament functionality for automated competition
4. HTTP API for monitoring and interacting with games

## Getting Started

### Prerequisites

- Go 1.16 or later
- A neural network model for at least one agent implementation (optional)

### Installation

1. Build the game server:

```sh
make build-game-server
```

2. Build the agent adapters for C++ and Go implementations:

```sh
make build-agent-adapters
```

### Running a Tournament

To run a tournament between all available agents:

```sh
make run-tournament
```

This will:

1. Start the game server
2. Register all available agents
3. Run round-robin matches between all agents
4. Display tournament results

## Game Server API

The game server exposes the following HTTP API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents` | GET | List all registered agents |
| `/api/games` | GET | List all games |
| `/api/games` | POST | Create a new game between two agents |
| `/api/games/{id}` | GET | Get game status |
| `/api/games/{id}/run` | POST | Run a specific game |
| `/api/tournament/start` | POST | Start a new tournament |
| `/api/tournament/status` | GET | Get tournament status |

## Agent Interface

To implement a custom agent, it needs to satisfy the `Agent` interface:

```go
type Agent interface {
    GetMove(state *game.RPSGame) (game.RPSMove, error)
    Name() string
}
```

There are three main ways to implement an agent:

1. **Native Agent**: Directly implement the `Agent` interface in Go
2. **External Go Agent**: Create a Go executable that accepts game state and returns moves
3. **External C++ Agent**: Create a C++ executable that accepts game state and returns moves

## Game State Format

The game state is communicated to external agents in the following format:

```
Board:R.S.P...|Hand1:RPS|Hand2:RPS|Current:1
```

Where:
- `Board` is a 9-character string representing the 3x3 board
  - `R`, `P`, `S` are Player 1's Rock, Paper, Scissors cards
  - `r`, `p`, `s` are Player 2's cards
  - `.` represents an empty space
- `Hand1` is Player 1's hand (e.g., `RPS` means they have Rock, Paper, and Scissors)
- `Hand2` is Player 2's hand
- `Current` is the ID of the current player (1 or 2)

## Move Format

External agents should output moves in the following format:

```
CardIndex:Position
```

Where:
- `CardIndex` is the index of the card in the player's hand (0-based)
- `Position` is the position on the board (0-8, row-major order)

## Adding New Agents

To add a new agent to the tournament system:

1. Implement the `Agent` interface or create an executable that follows the state/move format
2. Register the agent in `run_tournament.go`
3. Rebuild and run the tournament

## Customizing Tournaments

You can customize tournament settings by modifying:

- `game_server.go`: Change deck size, hand size, max rounds
- `run_tournament.go`: Change which agents participate
- Pass command-line arguments to the game server

## License

This project is licensed under the same license as the parent project. 