package agent

import (
	"testing"
)

func TestNewPPOAgent(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	if agent.stateSize != 9 {
		t.Errorf("NewPPOAgent() stateSize = %v, want 9", agent.stateSize)
	}
	if agent.actionSize != 3 {
		t.Errorf("NewPPOAgent() actionSize = %v, want 3", agent.actionSize)
	}
}

func TestGetPolicyProbs(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	probs := agent.GetPolicyProbs(state)

	// Check probabilities sum to 1
	sum := 0.0
	for _, prob := range probs {
		sum += prob
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("GetPolicyProbs() probabilities sum = %v, want ~1.0", sum)
	}

	// Check all probabilities are non-negative
	for i, prob := range probs {
		if prob < 0 {
			t.Errorf("GetPolicyProbs() probability[%d] = %v, want >= 0", i, prob)
		}
	}
}

func TestSampleAction(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	validActions := []int{0, 1, 2}

	// Test sampling with all actions valid
	action := agent.SampleAction(state, validActions)
	if action < 0 || action >= 3 {
		t.Errorf("SampleAction() returned invalid action %v", action)
	}

	// Test sampling with restricted valid actions
	validActions = []int{1}
	action = agent.SampleAction(state, validActions)
	if action != 1 {
		t.Errorf("SampleAction() with restricted actions returned %v, want 1", action)
	}
}

func TestGetValue(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	value := agent.GetValue(state)

	// Value should be between 0 and 2 (3 possible actions - 1)
	if value < 0 || value > 2 {
		t.Errorf("GetValue() returned value %v, want between 0 and 2", value)
	}
}

func TestUpdate(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	states := [][]float64{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}
	actions := []int{0}
	rewards := []float64{1.0}
	values := []float64{0.5}

	// Test that update doesn't panic
	agent.Update(states, actions, rewards, values)
}
