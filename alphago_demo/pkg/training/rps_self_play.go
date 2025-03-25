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

	// Calculate update frequency based on number of games
	// For small number of games (< 100), show updates every 10 games
	// For large number of games, show updates every 1% of progress
	updateFrequency := 10
	if sp.params.NumGames >= 100 {
		// At least every 1% of games, but maximum every 10 games for very large values
		onePercent := sp.params.NumGames / 100
		if onePercent < 10 {
			updateFrequency = onePercent
			if updateFrequency < 1 {
				updateFrequency = 1
			}
		}
	}

	for i := 0; i < sp.params.NumGames; i++ {
		// Show progress more frequently
		if verbose && (i%updateFrequency == 0 || i == sp.params.NumGames-1) {
			fmt.Printf("Playing game %d/%d (%.1f%%)\n",
				i+1, sp.params.NumGames, float64(i+1)/float64(sp.params.NumGames)*100)
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

// TrainNetworks trains the policy and value networks on the generated examples
func (sp *RPSSelfPlay) TrainNetworks(numEpochs int, batchSize int, learningRate float64, verbose bool) ([]float64, []float64) {
	// Check if we have examples
	if len(sp.examples) == 0 {
		if verbose {
			fmt.Println("No training examples to learn from!")
		}
		return nil, nil
	}

	// Shuffle examples for better learning
	rand.Shuffle(len(sp.examples), func(i, j int) {
		sp.examples[i], sp.examples[j] = sp.examples[j], sp.examples[i]
	})

	// Track losses for each epoch
	policyLosses := make([]float64, numEpochs)
	valueLosses := make([]float64, numEpochs)

	// Initialize or clear debug epoch counters
	sp.policyNetwork.DebugEpochCount = []int{0}
	sp.valueNetwork.DebugEpochCount = []int{0}

	// Train networks
	for epoch := 0; epoch < numEpochs; epoch++ {
		// Update epoch counter for debugging
		sp.policyNetwork.DebugEpochCount[0] = epoch
		sp.valueNetwork.DebugEpochCount[0] = epoch

		policyLoss := 0.0
		valueLoss := 0.0

		// Calculate previous losses for improvement reporting
		var prevPolicyLoss, prevValueLoss float64
		if epoch > 0 {
			prevPolicyLoss = policyLosses[epoch-1]
			prevValueLoss = valueLosses[epoch-1]
		}

		// Process in batches
		for b := 0; b < len(sp.examples); b += batchSize {
			end := b + batchSize
			if end > len(sp.examples) {
				end = len(sp.examples)
			}

			batch := sp.examples[b:end]

			// Create batch inputs and targets
			states := make([][]float64, len(batch))
			policyTargets := make([][]float64, len(batch))
			valueTargets := make([]float64, len(batch))

			for i, example := range batch {
				states[i] = example.BoardState
				policyTargets[i] = example.PolicyTarget
				valueTargets[i] = example.ValueTarget
			}

			// Train policy network with lower learning rate for larger networks
			actualLR := learningRate
			if sp.policyNetwork.GetHiddenSize() > 100 {
				// Reduce learning rate for larger networks to prevent instability
				actualLR = learningRate * 0.5
			}

			policyLossBatch := sp.policyNetwork.Train(states, policyTargets, actualLR)
			policyLoss += policyLossBatch

			// Train value network with same adjusted learning rate
			valueLossBatch := sp.valueNetwork.Train(states, valueTargets, actualLR)
			valueLoss += valueLossBatch
		}

		// Calculate average loss
		batchCount := (len(sp.examples) + batchSize - 1) / batchSize
		if batchCount > 0 {
			policyLoss /= float64(batchCount)
			valueLoss /= float64(batchCount)
		}

		// Store the losses
		policyLosses[epoch] = policyLoss
		valueLosses[epoch] = valueLoss

		// Calculate improvement percentages
		policyImprovement := 0.0
		valueImprovement := 0.0
		if epoch > 0 && prevPolicyLoss > 0 {
			policyImprovement = (prevPolicyLoss - policyLoss) / prevPolicyLoss * 100
		}
		if epoch > 0 && prevValueLoss > 0 {
			valueImprovement = (prevValueLoss - valueLoss) / prevValueLoss * 100
		}

		if verbose {
			improveStr := ""
			if epoch > 0 {
				improveStr = fmt.Sprintf(" (Policy: %+.1f%%, Value: %+.1f%%)",
					policyImprovement, valueImprovement)
			}

			fmt.Printf("Epoch %d/%d - Policy Loss: %.4f, Value Loss: %.4f%s\n",
				epoch+1, numEpochs, policyLoss, valueLoss, improveStr)

			// Add extra warnings if we see unexpected patterns in the losses
			if policyLoss < 0.0001 || valueLoss < 0.0001 {
				fmt.Printf("WARNING: Very low loss detected, possible underfitting or training collapse\n")
			}
			if epoch > 0 && (policyLoss > prevPolicyLoss*2 || valueLoss > prevValueLoss*2) {
				fmt.Printf("WARNING: Loss increased significantly, possible training instability\n")
			}
		}
	}

	return policyLosses, valueLosses
}
