package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"
)

// TrainingExample matches the struct from generate_training_data
type TrainingExample struct {
	BoardState    []int   `json:"board_state"`    // Flattened board (9 positions)
	Player1Hand   []int   `json:"player1_hand"`   // Card types in P1's hand
	Player2Hand   []int   `json:"player2_hand"`   // Card types in P2's hand
	CurrentPlayer int     `json:"current_player"` // 1 or 2
	BestMove      int     `json:"best_move"`      // 0-8 position index
	Evaluation    float64 `json:"evaluation"`     // Minimax evaluation
	GamePhase     string  `json:"game_phase"`     // "opening", "midgame", "endgame"
	SearchDepth   int     `json:"search_depth"`   // Depth used for this position
}

func main() {
	// Parse command line flags
	inputFile := flag.String("input", "data/training_data.json", "Input file with raw training data")
	outputDir := flag.String("output-dir", "data", "Directory to save processed data")
	trainSplit := flag.Float64("train-split", 0.8, "Proportion of data for training (0.0-1.0)")
	valSplit := flag.Float64("val-split", 0.1, "Proportion of data for validation (0.0-1.0)")
	flag.Parse()

	// Seed random number generator for consistent shuffle
	rand.Seed(time.Now().UnixNano())

	// Create output directory if it doesn't exist
	err := os.MkdirAll(*outputDir, 0755)
	if err != nil {
		panic(fmt.Sprintf("Failed to create output directory: %v", err))
	}

	// Load training data
	file, err := os.Open(*inputFile)
	if err != nil {
		panic(fmt.Sprintf("Failed to open training data: %v", err))
	}
	defer file.Close()

	var examples []TrainingExample
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&examples); err != nil {
		panic(fmt.Sprintf("Failed to decode training data: %v", err))
	}

	fmt.Printf("Loaded %d training examples\n", len(examples))

	// Shuffle data for random split
	rand.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})

	// Split into training, validation, and test sets
	numExamples := len(examples)
	numTraining := int(float64(numExamples) * *trainSplit)
	numValidation := int(float64(numExamples) * *valSplit)

	trainingData := examples[:numTraining]
	validationData := examples[numTraining : numTraining+numValidation]
	testData := examples[numTraining+numValidation:]

	fmt.Printf("Split into %d training, %d validation, %d test examples\n",
		len(trainingData), len(validationData), len(testData))

	// Convert to network inputs and outputs
	trainInputs, trainTargets := convertToNetworkFormat(trainingData)
	valInputs, valTargets := convertToNetworkFormat(validationData)
	testInputs, testTargets := convertToNetworkFormat(testData)

	// Save the processed data
	saveSets(*outputDir, "training", trainInputs, trainTargets)
	saveSets(*outputDir, "validation", valInputs, valTargets)
	saveSets(*outputDir, "test", testInputs, testTargets)

	fmt.Println("Preprocessing complete.")
}

// convertToNetworkFormat converts training examples to neural network inputs/outputs
func convertToNetworkFormat(examples []TrainingExample) ([][]float64, [][]float64) {
	inputs := make([][]float64, len(examples))
	targets := make([][]float64, len(examples))

	for i, example := range examples {
		// Create input vector (81 features)
		input := make([]float64, 81)

		// Encode board state (9 positions x 7 possible states)
		for pos, value := range example.BoardState {
			// One-hot encode each position (empty, P1-R, P1-P, P1-S, P2-R, P2-P, P2-S)
			offset := pos * 7
			if value == 0 {
				input[offset] = 1.0 // Empty
			} else {
				input[offset+value] = 1.0
			}
		}

		// Encode player hands (3 features each for counts)
		for j, count := range example.Player1Hand {
			input[63+j] = float64(count) / 5.0 // Normalize by max hand size
		}

		for j, count := range example.Player2Hand {
			input[66+j] = float64(count) / 5.0
		}

		// Encode current player (2 features)
		if example.CurrentPlayer == 1 {
			input[69] = 1.0
		} else {
			input[70] = 1.0
		}

		// Remaining indices 71-80 reserved for possible future features

		// Create target vector (one-hot encoded move)
		target := make([]float64, 9)
		target[example.BestMove] = 1.0

		inputs[i] = input
		targets[i] = target
	}

	return inputs, targets
}

// saveSets saves inputs and targets to files
func saveSets(dir, prefix string, inputs [][]float64, targets [][]float64) {
	// Save inputs
	inputFile, err := os.Create(fmt.Sprintf("%s/%s_inputs.json", dir, prefix))
	if err != nil {
		panic(fmt.Sprintf("Failed to create input file: %v", err))
	}
	defer inputFile.Close()

	encoder := json.NewEncoder(inputFile)
	if err := encoder.Encode(inputs); err != nil {
		panic(fmt.Sprintf("Failed to write inputs: %v", err))
	}

	// Save targets
	targetFile, err := os.Create(fmt.Sprintf("%s/%s_targets.json", dir, prefix))
	if err != nil {
		panic(fmt.Sprintf("Failed to create target file: %v", err))
	}
	defer targetFile.Close()

	encoder = json.NewEncoder(targetFile)
	if err := encoder.Encode(targets); err != nil {
		panic(fmt.Sprintf("Failed to write targets: %v", err))
	}

	fmt.Printf("Saved %s data (%d examples)\n", prefix, len(inputs))
}
