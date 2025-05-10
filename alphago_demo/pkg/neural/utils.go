package neural

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

// Helper functions for activation
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func softmax(values []float64) []float64 {
	// Find the maximum value to prevent overflow
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}

	// Apply exp and sum
	expSum := 0.0
	output := make([]float64, len(values))
	for i, v := range values {
		exp := math.Exp(v - max)
		output[i] = exp
		expSum += exp
	}

	// Normalize
	for i := range output {
		output[i] /= expSum
	}

	return output
}

// JSON serialization helpers
func saveToJSON(filename string, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, jsonData, 0644)
}

func loadFromJSON(filename string, data interface{}) error {
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	return json.Unmarshal(jsonData, data)
}

// Helper functions for loading weights from JSON data
func loadWeightsMatrix(data interface{}, target *[][]float64) error {
	if data == nil {
		return errors.New("weights data is nil")
	}

	// Try to cast the data to a slice of slices
	matrix, ok := data.([]interface{})
	if !ok {
		return errors.New("invalid matrix format")
	}

	// Ensure target matrix is initialized
	if *target == nil || len(*target) == 0 {
		return errors.New("target matrix not initialized")
	}

	// Load matrix values
	for i, row := range matrix {
		if i >= len(*target) {
			break
		}

		rowData, ok := row.([]interface{})
		if !ok {
			continue
		}

		for j, val := range rowData {
			if j >= len((*target)[i]) {
				break
			}

			if floatVal, ok := val.(float64); ok {
				(*target)[i][j] = floatVal
			}
		}
	}

	return nil
}

func loadWeightsVector(data interface{}, target *[]float64) error {
	if data == nil {
		return errors.New("vector data is nil")
	}

	// Try to cast the data to a slice
	vector, ok := data.([]interface{})
	if !ok {
		return errors.New("invalid vector format")
	}

	// Ensure target vector is initialized
	if *target == nil {
		return errors.New("target vector not initialized")
	}

	// Load vector values
	for i, val := range vector {
		if i >= len(*target) {
			break
		}

		if floatVal, ok := val.(float64); ok {
			(*target)[i] = floatVal
		}
	}

	return nil
}

// clipGradient restricts a gradient value to a specified range to prevent explosion
func clipGradient(gradient float64, threshold float64) float64 {
	if gradient > threshold {
		return threshold
	}
	if gradient < -threshold {
		return -threshold
	}
	return gradient
}

// CheckForNaN returns true if the value is NaN or Infinity
func CheckForNaN(value float64) bool {
	return math.IsNaN(value) || math.IsInf(value, 0)
}

// PolicyNetwork is an interface that can be implemented by different policy network types
type PolicyNetwork interface {
	Predict(features []float64) []float64
}

// LoadPolicyNetwork loads a policy network from a file
func LoadPolicyNetwork(filename string) (*RPSPolicyNetwork, error) {
	network := &RPSPolicyNetwork{}
	err := network.LoadFromFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to load policy network: %v", err)
	}
	return network, nil
}
