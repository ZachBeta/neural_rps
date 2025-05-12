package training

import (
	"testing"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	neural "github.com/zachbeta/neural_rps/alphago_demo/pkg/rps_net_impl"
)

func TestDefaultRPSSelfPlayParams(t *testing.T) {
	params := DefaultRPSSelfPlayParams()

	// Check that default parameters are reasonable
	if params.NumGames <= 0 {
		t.Errorf("Expected positive number of games, got %d", params.NumGames)
	}

	if params.DeckSize <= 0 {
		t.Errorf("Expected positive deck size, got %d", params.DeckSize)
	}

	if params.HandSize <= 0 {
		t.Errorf("Expected positive hand size, got %d", params.HandSize)
	}

	if params.MaxRounds <= 0 {
		t.Errorf("Expected positive max rounds, got %d", params.MaxRounds)
	}
}

func TestNewRPSSelfPlay(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create self-play with default parameters
	params := DefaultRPSSelfPlayParams()
	selfPlay := NewRPSSelfPlay(policyNetwork, valueNetwork, params)

	// Check that self-play was created correctly
	if selfPlay.policyNetwork != policyNetwork {
		t.Errorf("Expected policyNetwork to be the same as input")
	}

	if selfPlay.valueNetwork != valueNetwork {
		t.Errorf("Expected valueNetwork to be the same as input")
	}

	if selfPlay.params != params {
		t.Errorf("Expected params to be the same as input")
	}

	if selfPlay.examples == nil {
		t.Errorf("Expected examples to be initialized")
	}

	if len(selfPlay.examples) != 0 {
		t.Errorf("Expected examples to be empty initially")
	}
}

func TestRPSSelfPlayGenerateGames(t *testing.T) {
	// Create small policy and value networks for faster testing
	policyNetwork := neural.NewRPSPolicyNetwork(16)
	valueNetwork := neural.NewRPSValueNetwork(16)

	// Create self-play with small parameters for testing
	params := DefaultRPSSelfPlayParams()
	params.NumGames = 2
	params.DeckSize = 6
	params.HandSize = 2
	params.MaxRounds = 2
	params.MCTSParams.NumSimulations = 10

	selfPlay := NewRPSSelfPlay(policyNetwork, valueNetwork, params)

	// Generate games
	examples := selfPlay.GenerateGames(false)

	// Check that examples were generated
	if len(examples) == 0 {
		t.Errorf("Expected examples to be generated")
	}

	// Each game should generate at least one example
	if len(examples) < params.NumGames {
		t.Errorf("Expected at least %d examples, got %d", params.NumGames, len(examples))
	}

	// Check that examples have the correct format
	for i, example := range examples {
		// BoardState should have length 81
		if len(example.BoardState) != 81 {
			t.Errorf("Example %d: Expected BoardState to have length 81, got %d",
				i, len(example.BoardState))
		}

		// PolicyTarget should have length 9
		if len(example.PolicyTarget) != 9 {
			t.Errorf("Example %d: Expected PolicyTarget to have length 9, got %d",
				i, len(example.PolicyTarget))
		}

		// PolicyTarget should sum to approximately 1
		policySum := 0.0
		for _, p := range example.PolicyTarget {
			policySum += p
		}
		if policySum < 0.99 || policySum > 1.01 {
			t.Errorf("Example %d: Expected PolicyTarget to sum to approximately 1.0, got %f",
				i, policySum)
		}

		// ValueTarget should be in range [0, 1]
		if example.ValueTarget < 0.0 || example.ValueTarget > 1.0 {
			t.Errorf("Example %d: Expected ValueTarget to be in range [0, 1], got %f",
				i, example.ValueTarget)
		}
	}
}

func TestRPSSelfPlayExtractPolicy(t *testing.T) {
	// Create policy and value networks
	policyNetwork := neural.NewRPSPolicyNetwork(32)
	valueNetwork := neural.NewRPSValueNetwork(32)

	// Create self-play
	params := DefaultRPSSelfPlayParams()
	selfPlay := NewRPSSelfPlay(policyNetwork, valueNetwork, params)

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Test with nil node (should return uniform policy)
	var node *mcts.RPSMCTSNode = nil
	policy := selfPlay.extractPolicy(node)

	// Check policy length
	if len(policy) != 9 {
		t.Errorf("Expected policy to have length 9, got %d", len(policy))
	}

	// Check that policy sums to approximately 1
	policySum := 0.0
	for _, p := range policy {
		policySum += p
	}
	if policySum < 0.99 || policySum > 1.01 {
		t.Errorf("Expected policy to sum to approximately 1.0, got %f", policySum)
	}

	// Check that policy is uniform
	expectedProb := 1.0 / 9.0
	for i, p := range policy {
		if p != expectedProb {
			t.Errorf("Expected policy[%d] to be %f, got %f", i, expectedProb, p)
		}
	}

	// Now test with a node with children
	// Create a root node
	root := mcts.NewRPSMCTSNode(gameState, nil, nil, nil)

	// Create a few children with different visit counts
	for i := 0; i < 3; i++ {
		move := game.RPSMove{CardIndex: 0, Position: i, Player: game.Player1}
		childState := gameState.Copy()
		childState.MakeMove(move)

		child := mcts.NewRPSMCTSNode(childState, &move, root, nil)
		child.Visits = (i + 1) * 10 // Position 0: 10 visits, Position 1: 20 visits, Position 2: 30 visits

		root.Children = append(root.Children, child)
	}

	// Extract policy
	policy = selfPlay.extractPolicy(root)

	// Check policy length
	if len(policy) != 9 {
		t.Errorf("Expected policy to have length 9, got %d", len(policy))
	}

	// Check that policy sums to approximately 1
	policySum = 0.0
	for _, p := range policy {
		policySum += p
	}
	if policySum < 0.99 || policySum > 1.01 {
		t.Errorf("Expected policy to sum to approximately 1.0, got %f", policySum)
	}

	// Check that visited positions have non-zero probabilities
	// Total visits: 10 + 20 + 30 = 60
	// Position 0: 10/60 = 1/6
	// Position 1: 20/60 = 1/3
	// Position 2: 30/60 = 1/2
	expectedProbs := map[int]float64{
		0: 1.0 / 6.0,
		1: 1.0 / 3.0,
		2: 1.0 / 2.0,
	}

	for pos, expectedProb := range expectedProbs {
		if policy[pos] < expectedProb-0.01 || policy[pos] > expectedProb+0.01 {
			t.Errorf("Expected policy[%d] to be approximately %f, got %f",
				pos, expectedProb, policy[pos])
		}
	}

	// Check that unvisited positions have zero probabilities
	for i := 3; i < 9; i++ {
		if policy[i] != 0.0 {
			t.Errorf("Expected policy[%d] to be 0.0, got %f", i, policy[i])
		}
	}
}

func TestRPSSelfPlayTrainNetworks(t *testing.T) {
	// Create small policy and value networks for faster testing
	policyNetwork := neural.NewRPSPolicyNetwork(16)
	valueNetwork := neural.NewRPSValueNetwork(16)

	// Create self-play with small parameters for testing
	params := DefaultRPSSelfPlayParams()
	params.NumGames = 1
	params.DeckSize = 6
	params.HandSize = 2
	params.MaxRounds = 2
	params.MCTSParams.NumSimulations = 5

	selfPlay := NewRPSSelfPlay(policyNetwork, valueNetwork, params)

	// Generate games to get some examples
	examples := selfPlay.GenerateGames(false)

	// Make sure we have examples
	if len(examples) == 0 {
		t.Errorf("Failed to generate examples for training test")
		return
	}

	// Train networks
	numEpochs := 3
	batchSize := 2
	learningRate := 0.01

	// Should not panic
	selfPlay.TrainNetworks(numEpochs, batchSize, learningRate, false)

	// Test with no examples (should just print a message and return)
	selfPlay.examples = []RPSTrainingExample{}

	// Should not panic
	selfPlay.TrainNetworks(numEpochs, batchSize, learningRate, false)
}

func TestRPSSelfPlayFullPipeline(t *testing.T) {
	// Create small policy and value networks for faster testing
	policyNetwork := neural.NewRPSPolicyNetwork(16)
	valueNetwork := neural.NewRPSValueNetwork(16)

	// Create self-play with small parameters for testing
	params := DefaultRPSSelfPlayParams()
	params.NumGames = 1
	params.DeckSize = 6
	params.HandSize = 2
	params.MaxRounds = 2
	params.MCTSParams.NumSimulations = 5

	selfPlay := NewRPSSelfPlay(policyNetwork, valueNetwork, params)

	// Run the full pipeline: generate games and train networks
	examples := selfPlay.GenerateGames(false)

	// Make sure we have examples
	if len(examples) == 0 {
		t.Errorf("Failed to generate examples for training test")
		return
	}

	// Train networks
	numEpochs := 2
	batchSize := 2
	learningRate := 0.01

	// Should not panic
	selfPlay.TrainNetworks(numEpochs, batchSize, learningRate, false)

	// Create MCTS to test the trained networks
	mctsParams := mcts.DefaultRPSMCTSParams()
	mctsParams.NumSimulations = 5
	mctsEngine := mcts.NewRPSMCTS(policyNetwork, valueNetwork, mctsParams)

	// Create a game state
	gameState := game.NewRPSGame(15, 5, 10)

	// Set the root state
	mctsEngine.SetRootState(gameState)

	// Get the best move
	bestMove := mctsEngine.GetBestMove()

	// Check that a move was returned
	if bestMove == nil {
		t.Errorf("Failed to get a move from trained networks")
		return
	}

	// The move should be valid
	if bestMove.Player != gameState.CurrentPlayer {
		t.Errorf("Expected move to be for the current player %v, got %v",
			gameState.CurrentPlayer, bestMove.Player)
	}
}
