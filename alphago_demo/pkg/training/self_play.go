package training

import (
	"fmt"
	"math/rand"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/mcts"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

// AGTrainingExample represents a single training example from self-play
type AGTrainingExample struct {
	BoardState   []float64
	PolicyTarget []float64
	ValueTarget  float64
}

// AGSelfPlayParams contains parameters for self-play
type AGSelfPlayParams struct {
	NumGames   int
	MCTSParams mcts.AGMCTSParams
}

// DefaultAGSelfPlayParams returns default self-play parameters
func DefaultAGSelfPlayParams() AGSelfPlayParams {
	return AGSelfPlayParams{
		NumGames:   100,
		MCTSParams: mcts.DefaultAGMCTSParams(),
	}
}

// AGSelfPlay handles self-play for generating training data
type AGSelfPlay struct {
	params        AGSelfPlayParams
	policyNetwork *neural.AGPolicyNetwork
	valueNetwork  *neural.AGValueNetwork
	examples      []AGTrainingExample
}

// NewAGSelfPlay creates a new self-play instance
func NewAGSelfPlay(policyNetwork *neural.AGPolicyNetwork, valueNetwork *neural.AGValueNetwork, params AGSelfPlayParams) *AGSelfPlay {
	return &AGSelfPlay{
		params:        params,
		policyNetwork: policyNetwork,
		valueNetwork:  valueNetwork,
		examples:      make([]AGTrainingExample, 0),
	}
}

// GenerateGames generates games through self-play
func (sp *AGSelfPlay) GenerateGames(verbose bool) []AGTrainingExample {
	sp.examples = make([]AGTrainingExample, 0)

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
func (sp *AGSelfPlay) playGame(verbose bool) []AGTrainingExample {
	gameInstance := game.NewAGGame()
	moveHistory := make([]game.AGMove, 0)
	stateHistory := make([]*game.AGGame, 0)
	policyHistory := make([][]float64, 0)

	// Create MCTS instance
	mctsEngine := mcts.NewAGMCTS(sp.policyNetwork, sp.valueNetwork, sp.params.MCTSParams)

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

	if winner == game.Empty {
		value = 0.5 // Draw
	} else if winner == game.PlayerX {
		value = 1.0 // X wins
	} else {
		value = 0.0 // O wins
	}

	// Create training examples
	examples := make([]AGTrainingExample, 0, len(stateHistory))

	for i, state := range stateHistory {
		// Flip value based on player perspective
		var targetValue float64
		if state.CurrentPlayer == game.PlayerX {
			targetValue = value
		} else {
			targetValue = 1.0 - value
		}

		example := AGTrainingExample{
			BoardState:   state.GetBoardAsFeatures(),
			PolicyTarget: policyHistory[i],
			ValueTarget:  targetValue,
		}

		examples = append(examples, example)
	}

	return examples
}

// extractPolicy extracts a policy distribution from MCTS visit counts
func (sp *AGSelfPlay) extractPolicy(node *mcts.AGMCTSNode) []float64 {
	policy := make([]float64, 9) // 3x3 board flattened

	if node == nil || len(node.Children) == 0 {
		// Uniform random policy if no children
		for i := range policy {
			policy[i] = 1.0 / float64(9)
		}
		return policy
	}

	// Sum up total visits for normalization
	totalVisits := 0
	for _, child := range node.Children {
		totalVisits += child.Visits
	}

	// If no visits, use uniform random policy
	if totalVisits == 0 {
		for i := range policy {
			policy[i] = 1.0 / float64(9)
		}
		return policy
	}

	// Set policy based on visit proportions
	for _, child := range node.Children {
		if child.Move != nil {
			index := child.Move.Row*3 + child.Move.Col
			policy[index] = float64(child.Visits) / float64(totalVisits)
		}
	}

	return policy
}

// TrainNetworks trains the policy and value networks
func (sp *AGSelfPlay) TrainNetworks(numEpochs int, batchSize int, learningRate float64, verbose bool) {
	if len(sp.examples) == 0 {
		fmt.Println("No training examples available. Generate games first.")
		return
	}

	// Shuffle examples
	rand.Shuffle(len(sp.examples), func(i, j int) {
		sp.examples[i], sp.examples[j] = sp.examples[j], sp.examples[i]
	})

	// Train networks
	for epoch := 0; epoch < numEpochs; epoch++ {
		policyLoss := 0.0
		valueLoss := 0.0

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

			// Train policy network
			policyLossBatch := sp.policyNetwork.Train(states, policyTargets, learningRate)
			policyLoss += policyLossBatch

			// Train value network
			valueLossBatch := sp.valueNetwork.Train(states, valueTargets, learningRate)
			valueLoss += valueLossBatch
		}

		// Calculate average loss
		batchCount := (len(sp.examples) + batchSize - 1) / batchSize
		if batchCount > 0 {
			policyLoss /= float64(batchCount)
			valueLoss /= float64(batchCount)
		}

		if verbose || epoch%10 == 0 {
			fmt.Printf("Epoch %d/%d - Policy Loss: %.4f, Value Loss: %.4f\n",
				epoch+1, numEpochs, policyLoss, valueLoss)
		}
	}
}
