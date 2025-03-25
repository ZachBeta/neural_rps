package main

import (
	"fmt"
	"math/rand"
	"os/exec"
	"strconv"
	"strings"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// Base Agent interface is defined in game_server.go
// type Agent interface {
//     GetMove(state *game.RPSGame) (game.RPSMove, error)
//     Name() string
// }

// RandomAgent implements a simple random move agent
type RandomAgent struct {
	name string
}

func NewRandomAgent(name string) *RandomAgent {
	return &RandomAgent{name: name}
}

func (a *RandomAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	validMoves := state.GetValidMoves()
	if len(validMoves) == 0 {
		return game.RPSMove{}, fmt.Errorf("no valid moves")
	}
	return validMoves[rand.Intn(len(validMoves))], nil
}

func (a *RandomAgent) Name() string {
	return a.name
}

// AlphaGoAgent wraps the AlphaGo-style MCTS + neural network agent
type AlphaGoAgent struct {
	name          string
	policyNetwork *neural.RPSPolicyNetwork
	valueNetwork  *neural.RPSValueNetwork
	mctsEngine    *mcts.RPSMCTS
}

func NewAlphaGoAgent(name string, policyNet *neural.RPSPolicyNetwork, valueNet *neural.RPSValueNetwork) *AlphaGoAgent {
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = 400 // Adjust as needed

	return &AlphaGoAgent{
		name:          name,
		policyNetwork: policyNet,
		valueNetwork:  valueNet,
		mctsEngine:    mcts.NewRPSMCTS(policyNet, valueNet, mctsParams),
	}
}

func (a *AlphaGoAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	// Use MCTS to find the best move
	a.mctsEngine.SetRootState(state)
	bestNode := a.mctsEngine.Search()

	if bestNode == nil || bestNode.Move == nil {
		// Fallback to random move if MCTS fails
		validMoves := state.GetValidMoves()
		if len(validMoves) == 0 {
			return game.RPSMove{}, fmt.Errorf("no valid moves")
		}
		return validMoves[rand.Intn(len(validMoves))], nil
	}

	return *bestNode.Move, nil
}

func (a *AlphaGoAgent) Name() string {
	return a.name
}

// GoExternalAgent adapts the Golang implementation via executable
type GoExternalAgent struct {
	name       string
	executable string
	args       []string
}

func NewGoExternalAgent(name, executable string, args []string) *GoExternalAgent {
	return &GoExternalAgent{
		name:       name,
		executable: executable,
		args:       args,
	}
}

func (a *GoExternalAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	// Convert game state to string representation for external process
	stateStr := serializeGameState(state)

	// Prepare command with args
	cmd := exec.Command(a.executable, append(a.args, "--state", stateStr)...)

	// Run command and capture output
	output, err := cmd.Output()
	if err != nil {
		return game.RPSMove{}, fmt.Errorf("failed to run Go agent: %v", err)
	}

	// Parse move from output
	return parseMove(string(output), state)
}

func (a *GoExternalAgent) Name() string {
	return a.name
}

// CPPExternalAgent adapts the C++ implementation via executable
type CPPExternalAgent struct {
	name       string
	executable string
	args       []string
}

func NewCPPExternalAgent(name, executable string, args []string) *CPPExternalAgent {
	return &CPPExternalAgent{
		name:       name,
		executable: executable,
		args:       args,
	}
}

func (a *CPPExternalAgent) GetMove(state *game.RPSGame) (game.RPSMove, error) {
	// Convert game state to string representation for external process
	stateStr := serializeGameState(state)

	// Prepare command with args
	cmd := exec.Command(a.executable, append(a.args, "--state", stateStr)...)

	// Run command and capture output
	output, err := cmd.Output()
	if err != nil {
		return game.RPSMove{}, fmt.Errorf("failed to run C++ agent: %v", err)
	}

	// Parse move from output
	return parseMove(string(output), state)
}

func (a *CPPExternalAgent) Name() string {
	return a.name
}

// Helper functions for serialization and deserialization

// serializeGameState converts a game state to a string representation
func serializeGameState(state *game.RPSGame) string {
	// Format:
	// Board: R.S.P...  (uppercase=player1, lowercase=player2, .=empty)
	// Hand1: RPS  (player 1 hand)
	// Hand2: RPS  (player 2 hand)
	// Current: 1 or 2 (current player)

	// Board representation
	boardStr := ""
	for i := 0; i < 9; i++ {
		card := state.Board[i]
		if card.Owner == game.NoPlayer {
			boardStr += "."
		} else {
			var symbol string
			switch card.Type {
			case game.Rock:
				symbol = "R"
			case game.Paper:
				symbol = "P"
			case game.Scissors:
				symbol = "S"
			}

			if card.Owner == game.Player1 {
				boardStr += symbol
			} else {
				boardStr += strings.ToLower(symbol)
			}
		}
	}

	// Hand1 representation
	hand1Str := ""
	for _, card := range state.Player1Hand {
		switch card.Type {
		case game.Rock:
			hand1Str += "R"
		case game.Paper:
			hand1Str += "P"
		case game.Scissors:
			hand1Str += "S"
		}
	}

	// Hand2 representation
	hand2Str := ""
	for _, card := range state.Player2Hand {
		switch card.Type {
		case game.Rock:
			hand2Str += "R"
		case game.Paper:
			hand2Str += "P"
		case game.Scissors:
			hand2Str += "S"
		}
	}

	// Current player
	currentStr := "1"
	if state.CurrentPlayer == game.Player2 {
		currentStr = "2"
	}

	return fmt.Sprintf("Board:%s|Hand1:%s|Hand2:%s|Current:%s", boardStr, hand1Str, hand2Str, currentStr)
}

// parseMove parses a move from a string representation
func parseMove(output string, state *game.RPSGame) (game.RPSMove, error) {
	// Expected format: "CardIndex:Position"
	parts := strings.Split(strings.TrimSpace(output), ":")
	if len(parts) != 2 {
		return game.RPSMove{}, fmt.Errorf("invalid move format: %s", output)
	}

	cardIndex, err := strconv.Atoi(parts[0])
	if err != nil {
		return game.RPSMove{}, fmt.Errorf("invalid card index: %s", parts[0])
	}

	position, err := strconv.Atoi(parts[1])
	if err != nil {
		return game.RPSMove{}, fmt.Errorf("invalid position: %s", parts[1])
	}

	// Validate the move
	validMoves := state.GetValidMoves()
	for _, move := range validMoves {
		if move.CardIndex == cardIndex && move.Position == position {
			return move, nil
		}
	}

	return game.RPSMove{}, fmt.Errorf("invalid move: card index %d, position %d", cardIndex, position)
}
