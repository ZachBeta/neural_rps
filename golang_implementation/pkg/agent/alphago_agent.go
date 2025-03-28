package agent

import (
	"fmt"

	alphaGame "github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
	"github.com/zachbeta/neural_rps/pkg/game"
)

// AlphaGoAgent is an adapter that allows using the AlphaGo-style neural networks
// from the alphago_demo package with the RPS card game environment defined in
// the golang_implementation package.
//
// This adapter provides the following integration features:
// 1. Loads and uses neural network models from the alphago_demo package
// 2. Converts game states between the formats used by the two packages
// 3. Uses the alphago_demo MCTS implementation to guide decision making
// 4. Implements the GetMove interface expected by the tournament system
//
// The integration allows for comparing the AlphaGo agent with other agent types
// (e.g., PPO) in the same environment using the tournament system.
type AlphaGoAgent struct {
	name          string
	policyNetwork *neural.RPSPolicyNetwork
	valueNetwork  *neural.RPSValueNetwork
	mctsEngine    *mcts.RPSMCTS
	simulations   int
	exploration   float64
}

// NewAlphaGoAgent creates a new AlphaGo-style agent using the provided policy and
// value networks from the alphago_demo package.
func NewAlphaGoAgent(name string, policyNet *neural.RPSPolicyNetwork, valueNet *neural.RPSValueNetwork,
	simulations int, explorationConst float64) *AlphaGoAgent {

	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = simulations
	mctsParams.ExplorationConst = explorationConst

	return &AlphaGoAgent{
		name:          name,
		policyNetwork: policyNet,
		valueNetwork:  valueNet,
		mctsEngine:    mcts.NewRPSMCTS(policyNet, valueNet, mctsParams),
		simulations:   simulations,
		exploration:   explorationConst,
	}
}

// Name returns the agent's name
func (a *AlphaGoAgent) Name() string {
	return a.name
}

// GetMove returns the best move according to the MCTS search.
// This method implements the GetMove interface expected by the tournament system.
func (a *AlphaGoAgent) GetMove(gameState *game.RPSCardGame) (game.RPSCardMove, error) {
	// Convert to AlphaGo game state
	alphaGameState := convertToAlphaGoGame(gameState)

	// Set root state for MCTS
	a.mctsEngine.SetRootState(alphaGameState)

	// Run MCTS search
	bestNode := a.mctsEngine.Search()

	// Extract move
	if bestNode == nil || bestNode.Move == nil {
		// Fallback to random move if MCTS fails
		return gameState.GetRandomMove()
	}

	// Convert back to our game move
	move := game.RPSCardMove{
		CardIndex: bestNode.Move.CardIndex,
		Position:  bestNode.Move.Position,
		Player:    game.Player(bestNode.Move.Player),
	}

	return move, nil
}

// convertToAlphaGoGame converts our game state to AlphaGo game state.
// This function maps between the RPSCardGame from the golang_implementation
// and the RPSGame from the alphago_demo package.
func convertToAlphaGoGame(ourGame *game.RPSCardGame) *alphaGame.RPSGame {
	alphaGameState := alphaGame.NewRPSGame(
		ourGame.DeckSize,
		ourGame.HandSize,
		ourGame.MaxRounds,
	)

	// Copy board state
	for pos := 0; pos < 9; pos++ {
		if ourGame.BoardOwner[pos] != game.NoPlayer {
			// Convert RPSCardType to int for the alphago format
			cardType := int(ourGame.Board[pos])

			// We need to set the card type and owner in the Board array
			alphaGameState.Board[pos].Type = alphaGame.RPSCardType(cardType)

			// Set the board owner by using the appropriate constant
			playerVal := 0
			if ourGame.BoardOwner[pos] == game.Player2 {
				playerVal = 1 // alphaGame.Player2
			}
			alphaGameState.SetBoardOwner(pos, playerVal)
		}
	}

	// Copy hands
	alphaPlayer1Hand := make([]int, len(ourGame.Player1Hand))
	for i, card := range ourGame.Player1Hand {
		alphaPlayer1Hand[i] = int(card)
	}
	alphaGameState.SetPlayer1Hand(alphaPlayer1Hand)

	alphaPlayer2Hand := make([]int, len(ourGame.Player2Hand))
	for i, card := range ourGame.Player2Hand {
		alphaPlayer2Hand[i] = int(card)
	}
	alphaGameState.SetPlayer2Hand(alphaPlayer2Hand)

	// Copy current player and round
	currentPlayer := 0
	if ourGame.CurrentPlayer == game.Player2 {
		currentPlayer = 1
	}
	alphaGameState.SetCurrentPlayer(currentPlayer)
	alphaGameState.SetRound(ourGame.Round)

	return alphaGameState
}

// LoadAlphaGoNetworksFromFile loads the policy and value networks from files.
// This function provides a convenient way to load the neural network models
// that were trained and saved by the alphago_demo package.
func LoadAlphaGoNetworksFromFile(policyPath, valuePath string) (*neural.RPSPolicyNetwork, *neural.RPSValueNetwork, error) {
	policyNet := neural.NewRPSPolicyNetwork(128) // Default hidden size, will be overwritten by loading
	err := policyNet.LoadFromFile(policyPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load policy network: %v", err)
	}

	valueNet := neural.NewRPSValueNetwork(128) // Default hidden size, will be overwritten by loading
	err = valueNet.LoadFromFile(valuePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load value network: %v", err)
	}

	return policyNet, valueNet, nil
}
