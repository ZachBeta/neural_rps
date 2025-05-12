package training

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

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
	NumGames      int
	DeckSize      int
	HandSize      int
	MaxRounds     int
	MCTSParams    mcts.RPSMCTSParams
	ForceParallel bool // Force parallel execution regardless of game count
	NumThreads    int  // Specific number of threads to use (0 = auto)
}

// DefaultRPSSelfPlayParams returns default self-play parameters
func DefaultRPSSelfPlayParams() RPSSelfPlayParams {
	return RPSSelfPlayParams{
		NumGames:      100,
		DeckSize:      21, // 7 of each card type
		HandSize:      5,
		MaxRounds:     10,
		MCTSParams:    mcts.DefaultRPSMCTSParams(),
		ForceParallel: false,
		NumThreads:    0, // Auto-select thread count
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

	// Use serial or parallel generation based on game count and available cores
	if (sp.params.NumGames < 5 || runtime.NumCPU() <= 2) && !sp.params.ForceParallel {
		// Use original serial implementation for small jobs or limited cores
		return sp.generateGamesSerial(verbose)
	} else {
		// Use parallel implementation for larger jobs with multiple cores
		// or when explicitly requested with ForceParallel
		return sp.generateGamesParallel(verbose)
	}
}

// generateGamesSerial generates games serially (original implementation)
func (sp *RPSSelfPlay) generateGamesSerial(verbose bool) []RPSTrainingExample {
	startTime := time.Now()
	totalExamples := 0

	for i := 0; i < sp.params.NumGames; i++ {
		if verbose || (i+1)%10 == 0 || i == 0 {
			fmt.Printf("Playing game %d/%d (%.1f%%)\n", i+1, sp.params.NumGames,
				float64(i+1)/float64(sp.params.NumGames)*100)
		}

		gameExamples := sp.playGame(verbose && i == 0)
		sp.examples = append(sp.examples, gameExamples...)
		totalExamples += len(gameExamples)

		// Report progress for long runs
		if (i+1)%20 == 0 && i+1 < sp.params.NumGames {
			elapsed := time.Since(startTime)
			gamesPerSecond := float64(i+1) / elapsed.Seconds()
			estimatedTotal := time.Duration(float64(sp.params.NumGames) / gamesPerSecond * float64(time.Second))
			estimatedRemaining := estimatedTotal - elapsed

			fmt.Printf("  Progress: %d/%d games, %.2f games/sec, ~%s remaining\n",
				i+1, sp.params.NumGames, gamesPerSecond, estimatedRemaining.Round(time.Second))
		}
	}

	// Calculate statistics
	elapsed := time.Since(startTime)
	examplesPerGame := float64(totalExamples) / float64(sp.params.NumGames)
	gamesPerSecond := float64(sp.params.NumGames) / elapsed.Seconds()

	if verbose {
		fmt.Printf("Generated %d training examples in %s (%.1f examples/game, %.2f games/sec)\n",
			totalExamples, elapsed, examplesPerGame, gamesPerSecond)
	}

	return sp.examples
}

// generateGamesParallel generates games in parallel using multiple goroutines
func (sp *RPSSelfPlay) generateGamesParallel(verbose bool) []RPSTrainingExample {
	startTime := time.Now()

	// Determine number of workers based on CPU count
	// Use N-1 workers to avoid saturating the system
	numWorkers := runtime.NumCPU() - 1
	if numWorkers < 1 {
		numWorkers = 1
	}

	// Use explicit thread count if specified
	if sp.params.NumThreads > 0 {
		numWorkers = sp.params.NumThreads
	}

	// Create a buffered channel for game examples
	gamesChan := make(chan []RPSTrainingExample, sp.params.NumGames)

	// For progress tracking
	progressChan := make(chan int, sp.params.NumGames)

	var wg sync.WaitGroup

	// Start a goroutine to track and report progress
	if verbose {
		go func() {
			completed := 0
			ticker := time.NewTicker(5 * time.Second)
			defer ticker.Stop()

			for {
				select {
				case _, ok := <-progressChan:
					if !ok {
						return // Channel closed, exit goroutine
					}
					completed++

					// Report every 10% or when requested
					if completed%10 == 0 || completed == sp.params.NumGames {
						elapsed := time.Since(startTime)
						gamesPerSecond := float64(completed) / elapsed.Seconds()
						estimatedTotal := time.Duration(float64(sp.params.NumGames) / gamesPerSecond * float64(time.Second))
						estimatedRemaining := estimatedTotal - elapsed

						fmt.Printf("  Progress: %d/%d games (%.1f%%), %.2f games/sec, ~%s remaining\n",
							completed, sp.params.NumGames,
							float64(completed)/float64(sp.params.NumGames)*100,
							gamesPerSecond, estimatedRemaining.Round(time.Second))
					}

				case <-ticker.C:
					// Regular progress update every 5 seconds
					if completed > 0 && completed < sp.params.NumGames {
						elapsed := time.Since(startTime)
						gamesPerSecond := float64(completed) / elapsed.Seconds()
						estimatedTotal := time.Duration(float64(sp.params.NumGames) / gamesPerSecond * float64(time.Second))
						estimatedRemaining := estimatedTotal - elapsed

						fmt.Printf("  Progress: %d/%d games (%.1f%%), %.2f games/sec, ~%s remaining\n",
							completed, sp.params.NumGames,
							float64(completed)/float64(sp.params.NumGames)*100,
							gamesPerSecond, estimatedRemaining.Round(time.Second))
					}
				}
			}
		}()
	}

	fmt.Printf("Starting parallel self-play with %d workers for %d games...\n",
		numWorkers, sp.params.NumGames)

	// Create and start worker goroutines
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// Calculate games per worker
			gamesPerWorker := sp.params.NumGames / numWorkers
			startGame := workerID * gamesPerWorker
			endGame := startGame + gamesPerWorker

			// Last worker takes any remainder
			if workerID == numWorkers-1 {
				endGame = sp.params.NumGames
			}

			// Each worker needs its own copy of networks
			localPolicyNet := sp.policyNetwork.Clone()
			localValueNet := sp.valueNetwork.Clone()

			// Each worker generates its assigned games
			for j := startGame; j < endGame; j++ {
				examples := sp.playGameWithNetworks(localPolicyNet, localValueNet, verbose && j == 0)
				gamesChan <- examples
				if verbose {
					progressChan <- 1
				}
			}
		}(i)
	}

	// Close channels once all workers are done
	go func() {
		wg.Wait()
		close(gamesChan)
		if verbose {
			close(progressChan)
		}
	}()

	// Collect all game examples
	allExamples := make([]RPSTrainingExample, 0)
	totalExamples := 0

	for examples := range gamesChan {
		allExamples = append(allExamples, examples...)
		totalExamples += len(examples)
	}

	// Calculate and report statistics
	elapsed := time.Since(startTime)
	examplesPerGame := float64(totalExamples) / float64(sp.params.NumGames)
	gamesPerSecond := float64(sp.params.NumGames) / elapsed.Seconds()

	fmt.Printf("Generated %d training examples in %s (%.1f examples/game, %.2f games/sec)\n",
		totalExamples, elapsed, examplesPerGame, gamesPerSecond)

	sp.examples = allExamples
	return allExamples
}

// playGameWithNetworks plays a single game using the provided networks
// This allows worker goroutines to use their own network copies
func (sp *RPSSelfPlay) playGameWithNetworks(
	policyNetwork *neural.RPSPolicyNetwork,
	valueNetwork *neural.RPSValueNetwork,
	verbose bool) []RPSTrainingExample {

	gameInstance := game.NewRPSGame(sp.params.DeckSize, sp.params.HandSize, sp.params.MaxRounds)
	moveHistory := make([]game.RPSMove, 0)
	stateHistory := make([]*game.RPSGame, 0)
	policyHistory := make([][]float64, 0)

	// Create MCTS instance with the worker's network copies
	mctsParams := sp.params.MCTSParams
	mctsEngine := mcts.NewRPSMCTS(policyNetwork, valueNetwork, mctsParams)

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

// Original playGame implementation remains unchanged
func (sp *RPSSelfPlay) playGame(verbose bool) []RPSTrainingExample {
	return sp.playGameWithNetworks(sp.policyNetwork, sp.valueNetwork, verbose)
}

// extractPolicy extracts a policy distribution from MCTS visit counts
func (sp *RPSSelfPlay) extractPolicy(node *mcts.RPSMCTSNode) []float64 {
	// Initialize policy target with zeros (9 possible positions)
	policyTarget := make([]float64, 9)

	// Check if node and children are valid
	if node == nil || len(node.Children) == 0 {
		// No children, return uniform distribution or zeros if no valid moves
		if node != nil && node.GameState != nil {
			validMoves := node.GameState.GetValidMoves()
			if len(validMoves) > 0 {
				prob := 1.0 / float64(len(validMoves))
				for _, move := range validMoves {
					if move.Position >= 0 && move.Position < 9 {
						policyTarget[move.Position] = prob
					}
				}
			}
		}
		return policyTarget
	}

	// Use visit counts from children to form the policy target
	movesByPosition := make([]int64, 9) // Changed to int64 to match atomic.Int64.Load()
	totalVisits := int64(0)             // Changed to int64

	for _, child := range node.Children {
		if child.Move != nil && child.Move.Position >= 0 && child.Move.Position < 9 {
			position := child.Move.Position
			childVisitsLoaded := child.Visits.Load()
			movesByPosition[position] += childVisitsLoaded
			totalVisits += childVisitsLoaded
		}
	}

	if totalVisits > 0 {
		for i := 0; i < 9; i++ {
			policyTarget[i] = float64(movesByPosition[i]) / float64(totalVisits)
		}
	} else {
		// Fallback to uniform if no visits (should be rare if search ran)
		if node.GameState != nil {
			validMoves := node.GameState.GetValidMoves()
			if len(validMoves) > 0 {
				prob := 1.0 / float64(len(validMoves))
				for _, move := range validMoves {
					if move.Position >= 0 && move.Position < 9 {
						policyTarget[move.Position] = prob
					}
				}
			}
		}
	}

	return policyTarget
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
