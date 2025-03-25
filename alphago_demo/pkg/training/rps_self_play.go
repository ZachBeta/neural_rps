package training

import (
	"fmt"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// RPSTrainingExample represents a single training example from self-play
type RPSTrainingExample struct {
	BoardState   []float64
	PolicyTarget []float64
	ValueTarget  float64
}

// RPSSelfPlayParams contains parameters for self-play
type RPSSelfPlayParams struct {
	NumGames   int
	DeckSize   int
	HandSize   int
	MaxRounds  int
	MCTSParams mcts.RPSMCTSParams
}

// DefaultRPSSelfPlayParams returns default self-play parameters
func DefaultRPSSelfPlayParams() RPSSelfPlayParams {
	return RPSSelfPlayParams{
		NumGames:   100,
		DeckSize:   21, // 7 of each card type
		HandSize:   5,
		MaxRounds:  10,
		MCTSParams: mcts.DefaultRPSMCTSParams(),
	}
}

// RPSSelfPlay handles self-play for generating training data
type RPSSelfPlay struct {
	params        RPSSelfPlayParams
	policyNetwork *neural.RPSPolicyNetwork
	valueNetwork  *neural.RPSValueNetwork
	examples      []RPSTrainingExample
}

// NewRPSSelfPlay creates a new self-play instance
func NewRPSSelfPlay(policyNetwork *neural.RPSPolicyNetwork, valueNetwork *neural.RPSValueNetwork, params RPSSelfPlayParams) *RPSSelfPlay {
	return &RPSSelfPlay{
		params:        params,
		policyNetwork: policyNetwork,
		valueNetwork:  valueNetwork,
		examples:      make([]RPSTrainingExample, 0),
	}
}

// GenerateGames generates games through self-play
func (sp *RPSSelfPlay) GenerateGames(verbose bool) []RPSTrainingExample {
	sp.examples = make([]RPSTrainingExample, 0)

	for i := 0; i < sp.params.NumGames; i++ {
		if verbose && i%10 == 0 {
			fmt.Printf("Playing game %d/%d\n", i+1, sp.params.NumGames)
		}

		gameExamples := sp.playGame(verbose && i == 0)
		sp.examples = append(sp.examples, gameExamples...)
	}

	return sp.examples
}

// playGame plays a single game and returns training examples
func (sp *RPSSelfPlay) playGame(verbose bool) []RPSTrainingExample {
	gameInstance := game.NewRPSGame(sp.params.DeckSize, sp.params.HandSize, sp.params.MaxRounds)
	moveHistory := make([]game.RPSMove, 0)
	stateHistory := make([]*game.RPSGame, 0)
	policyHistory := make([][]float64, 0)

	// Create MCTS instance
	mctsEngine := mcts.NewRPSMCTS(sp.policyNetwork, sp.valueNetwork, sp.params.MCTSParams)

	// Play until game is over
	for !gameInstance.IsGameOver() {
		// Store current state
		stateHistory = append(stateHistory, gameInstance.Copy())

		// Set root state for MCTS
		mctsEngine.SetRootState(gameInstance)

		// Search for best move
		bestNode := mctsEngine.Search()

		// Extract policy from MCTS visit counts
		policy := sp.extractPolicy(bestNode)
		policyHistory = append(policyHistory, policy)

		// Make the move
		if bestNode != nil && bestNode.Move != nil {
			moveHistory = append(moveHistory, *bestNode.Move)
			gameInstance.MakeMove(*bestNode.Move)

			if verbose {
				fmt.Println(gameInstance.String())
			}
		} else {
			// Fallback to random move if MCTS fails
			randomMove, err := gameInstance.GetRandomMove()
			if err == nil {
				moveHistory = append(moveHistory, randomMove)
				gameInstance.MakeMove(randomMove)

				if verbose {
					fmt.Println(gameInstance.String())
				}
			} else {
				// Break if no moves possible
				break
			}
		}
	}

	// Determine game result
	var value float64
	winner := gameInstance.GetWinner()

	if winner == game.NoPlayer {
		value = 0.5 // Draw
	} else if winner == game.Player1 {
		value = 1.0 // Player1 wins
	} else {
		value = 0.0 // Player2 wins
	}

	// Create training examples
	examples := make([]RPSTrainingExample, 0, len(stateHistory))

	for i, state := range stateHistory {
		// Flip value based on player perspective
		var targetValue float64
		if state.CurrentPlayer == game.Player1 {
			targetValue = value
		} else {
			targetValue = 1.0 - value
		}

		example := RPSTrainingExample{
			BoardState:   state.GetBoardAsFeatures(),
			PolicyTarget: policyHistory[i],
			ValueTarget:  targetValue,
		}

		examples = append(examples, example)
	}

	return examples
}

// extractPolicy extracts a policy distribution from MCTS visit counts
func (sp *RPSSelfPlay) extractPolicy(node *mcts.RPSMCTSNode) []float64 {
	policy := make([]float64, 9) // 9 positions flattened (3x3 board)

	if node == nil || len(node.Children) == 0 {
		// Uniform random policy if no children
		for i := range policy {
			policy[i] = 1.0 / float64(9)
		}
		return policy
	}

	// Group children by position to handle multiple cards that can be played at the same position
	movesByPosition := make(map[int]int) // position -> total visits
	totalVisits := 0

	for _, child := range node.Children {
		if child.Move != nil {
			position := child.Move.Position
			movesByPosition[position] += child.Visits
			totalVisits += child.Visits
		}
	}

	// If no visits, use uniform random policy
	if totalVisits == 0 {
		for i := range policy {
			policy[i] = 1.0 / float64(9)
		}
		return policy
	}

	// Set policy based on visit proportions by position
	for pos, visits := range movesByPosition {
		policy[pos] = float64(visits) / float64(totalVisits)
	}

	return policy
}

// TrainNetworks trains the policy and value networks
func (sp *RPSSelfPlay) TrainNetworks(numEpochs int, batchSize int, learningRate float64, verbose bool) {
	if len(sp.examples) == 0 {
		fmt.Println("No training examples available. Generate games first.")
		return
	}

	// Shuffle examples
	rand.Shuffle(len(sp.examples), func(i, j int) {
		sp.examples[i], sp.examples[j] = sp.examples[j], sp.examples[i]
	})

	// Prepare training data
	inputFeatures := make([][]float64, len(sp.examples))
	policyTargets := make([][]float64, len(sp.examples))
	valueTargets := make([]float64, len(sp.examples))

	for i, example := range sp.examples {
		inputFeatures[i] = example.BoardState
		policyTargets[i] = example.PolicyTarget
		valueTargets[i] = example.ValueTarget
	}

	// Train for the specified number of epochs
	for epoch := 0; epoch < numEpochs; epoch++ {
		// Process in batches
		for batchStart := 0; batchStart < len(inputFeatures); batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > len(inputFeatures) {
				batchEnd = len(inputFeatures)
			}

			batchInputs := inputFeatures[batchStart:batchEnd]
			batchPolicyTargets := policyTargets[batchStart:batchEnd]
			batchValueTargets := valueTargets[batchStart:batchEnd]

			// Train policy network
			policyLoss := sp.policyNetwork.Train(batchInputs, batchPolicyTargets, learningRate)

			// Train value network
			valueLoss := sp.valueNetwork.Train(batchInputs, batchValueTargets, learningRate)

			if verbose && (epoch == 0 || epoch == numEpochs-1 || epoch%(numEpochs/10) == 0) && batchStart == 0 {
				fmt.Printf("Epoch %d/%d: Policy Loss = %.4f, Value Loss = %.4f\n",
					epoch+1, numEpochs, policyLoss, valueLoss)
			}
		}
	}
}
