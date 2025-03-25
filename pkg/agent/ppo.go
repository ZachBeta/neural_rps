package agent

import (
	"math"
	"math/rand"

	"github.com/zachbeta/neural_rps/pkg/neural"
)

// PPOAgent represents a Proximal Policy Optimization agent
type PPOAgent struct {
	network      *neural.Network
	stateSize    int
	actionSize   int
	gamma        float64
	epsilon      float64
	clipEpsilon  float64
	valueCoeff   float64
	entropyCoeff float64
}

// NewPPOAgent creates a new PPO agent
func NewPPOAgent(stateSize, actionSize int) *PPOAgent {
	return &PPOAgent{
		network:      neural.NewNetwork(stateSize, 64, actionSize),
		stateSize:    stateSize,
		actionSize:   actionSize,
		gamma:        0.99,
		epsilon:      0.2,
		clipEpsilon:  0.2,
		valueCoeff:   0.5,
		entropyCoeff: 0.01,
	}
}

// GetPolicyProbs returns the action probabilities for the current state
func (a *PPOAgent) GetPolicyProbs(state []float64) []float64 {
	return a.network.Forward(state)
}

// SampleAction samples an action from the policy
func (a *PPOAgent) SampleAction(state []float64, validActions []int) int {
	probs := a.GetPolicyProbs(state)

	// Mask invalid actions
	for i := range probs {
		isValid := false
		for _, validAction := range validActions {
			if i == validAction {
				isValid = true
				break
			}
		}
		if !isValid {
			probs[i] = 0
		}
	}

	// Normalize probabilities
	sum := 0.0
	for _, prob := range probs {
		sum += prob
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}

	// Sample action
	r := rand.Float64()
	cumsum := 0.0
	for i, prob := range probs {
		cumsum += prob
		if r <= cumsum {
			return i
		}
	}

	return validActions[0] // Fallback to first valid action
}

// GetValue returns the value estimate for the current state
func (a *PPOAgent) GetValue(state []float64) float64 {
	probs := a.GetPolicyProbs(state)
	value := 0.0
	for i, prob := range probs {
		value += float64(i) * prob
	}
	return value
}

// Update updates the agent's policy using PPO
func (a *PPOAgent) Update(states [][]float64, actions []int, rewards []float64, values []float64) {
	if len(states) == 0 {
		return
	}

	// Calculate advantages
	advantages := make([]float64, len(rewards))
	for i := len(rewards) - 1; i >= 0; i-- {
		if i == len(rewards)-1 {
			advantages[i] = rewards[i] - values[i]
		} else {
			advantages[i] = rewards[i] + a.gamma*values[i+1] - values[i]
		}
	}

	// Normalize advantages
	mean := 0.0
	for _, adv := range advantages {
		mean += adv
	}
	mean /= float64(len(advantages))

	std := 0.0
	for _, adv := range advantages {
		diff := adv - mean
		std += diff * diff
	}
	std = math.Sqrt(std / float64(len(advantages)))

	if std > 0 {
		for i := range advantages {
			advantages[i] = (advantages[i] - mean) / std
		}
	}

	// Update policy
	for i := 0; i < len(states); i++ {
		oldProbs := a.GetPolicyProbs(states[i])
		oldProb := oldProbs[actions[i]]

		// Calculate new probabilities
		newProbs := a.GetPolicyProbs(states[i])
		newProb := newProbs[actions[i]]

		// Calculate ratio
		ratio := newProb / oldProb

		// Calculate clipped surrogate objective
		surr1 := ratio * advantages[i]
		surr2 := math.Max(math.Min(ratio, 1+a.epsilon), 1-a.epsilon) * advantages[i]
		surr := math.Min(surr1, surr2)

		// Calculate value loss
		valueLoss := math.Pow(rewards[i]-a.GetValue(states[i]), 2)

		// Calculate entropy bonus
		entropy := 0.0
		for _, prob := range newProbs {
			if prob > 0 {
				entropy -= prob * math.Log(prob)
			}
		}

		// Total loss
		loss := -surr + a.valueCoeff*valueLoss - a.entropyCoeff*entropy

		// Update network (simplified - in practice, you'd use proper gradient descent)
		// This is just a placeholder for the actual update logic
		// In a real implementation, you'd use backpropagation and gradient descent
		// The loss value would be used to compute gradients and update weights
		_ = loss // TODO: Implement proper gradient descent using the loss value
	}
}

// SaveWeights saves the agent's network weights
func (a *PPOAgent) SaveWeights(filename string) error {
	return a.network.SaveWeights(filename)
}

// LoadWeights loads the agent's network weights
func (a *PPOAgent) LoadWeights(filename string) error {
	return a.network.LoadWeights(filename)
}
