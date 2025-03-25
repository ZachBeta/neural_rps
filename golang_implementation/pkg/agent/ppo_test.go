package agent

import (
	"math"
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

func TestPolicyConvergence(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	state[0] = 1.0 // Set one input to 1, rest to 0

	// Store initial policy
	initialProbs := agent.GetPolicyProbs(state)

	// Train on consistent positive rewards for action 0
	for i := 0; i < 100; i++ {
		states := [][]float64{state}
		actions := []int{0}
		rewards := []float64{1.0}
		values := []float64{0.5}
		agent.Update(states, actions, rewards, values)
	}

	// Get final policy
	finalProbs := agent.GetPolicyProbs(state)

	// Check that policy converged towards action 0
	if finalProbs[0] < initialProbs[0] {
		t.Errorf("Policy did not converge towards action 0: initial=%f, final=%f",
			initialProbs[0], finalProbs[0])
	}
}

func TestValueEstimation(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	state[0] = 1.0

	// Store initial value estimate
	initialValue := agent.GetValue(state)

	// Train on consistent positive rewards
	for i := 0; i < 100; i++ {
		states := [][]float64{state}
		actions := []int{0}
		rewards := []float64{1.0}
		values := []float64{1.0}
		agent.Update(states, actions, rewards, values)
	}

	// Get final value estimate
	finalValue := agent.GetValue(state)

	// Check that value estimate increased
	if finalValue < initialValue {
		t.Errorf("Value estimate did not increase: initial=%f, final=%f",
			initialValue, finalValue)
	}
}

func TestPolicyEntropy(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	state[0] = 1.0

	// Calculate initial policy entropy
	initialProbs := agent.GetPolicyProbs(state)
	initialEntropy := calculateEntropy(initialProbs)

	// Train on mixed rewards
	for i := 0; i < 100; i++ {
		states := [][]float64{state}
		actions := []int{i % 3} // Cycle through actions
		rewards := []float64{0.5}
		values := []float64{0.5}
		agent.Update(states, actions, rewards, values)
	}

	// Calculate final policy entropy
	finalProbs := agent.GetPolicyProbs(state)
	finalEntropy := calculateEntropy(finalProbs)

	// Check that entropy decreased (policy became more deterministic)
	if finalEntropy > initialEntropy {
		t.Errorf("Policy entropy did not decrease: initial=%f, final=%f",
			initialEntropy, finalEntropy)
	}
}

func calculateEntropy(probs []float64) float64 {
	entropy := 0.0
	for _, p := range probs {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

func TestGradientScaling(t *testing.T) {
	agent := NewPPOAgent(9, 3)
	state := make([]float64, 9)
	state[0] = 1.0

	// Store initial policy
	initialProbs := agent.GetPolicyProbs(state)

	// Test different reward scales
	rewardScales := []float64{0.1, 1.0, 10.0}
	for _, scale := range rewardScales {
		// Reset agent
		agent = NewPPOAgent(9, 3)

		// Train with scaled rewards
		for i := 0; i < 100; i++ {
			states := [][]float64{state}
			actions := []int{0}
			rewards := []float64{scale}
			values := []float64{scale}
			agent.Update(states, actions, rewards, values)
		}

		// Get final policy
		finalProbs := agent.GetPolicyProbs(state)
		policyChange := math.Abs(finalProbs[0] - initialProbs[0])

		// Check that policy changes scale with reward magnitude
		if policyChange < scale*0.1 {
			t.Errorf("Policy changes do not scale with reward magnitude %f: change=%f",
				scale, policyChange)
		}
	}
}
