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
	lambda       float64
	learningRate float64
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
		lambda:       0.95,
		learningRate: 0.001,
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

// Update updates the policy and value networks using PPO
func (a *PPOAgent) Update(states [][]float64, actions []int, rewards []float64, values []float64) {
	if len(states) == 0 {
		return
	}

	// Compute advantages and returns
	advantages := make([]float64, len(rewards))
	returns := make([]float64, len(rewards))
	lastValue := 0.0
	for i := len(rewards) - 1; i >= 0; i-- {
		delta := rewards[i] + a.gamma*lastValue - values[i]
		advantages[i] = delta + a.gamma*a.lambda*lastValue
		returns[i] = rewards[i] + a.gamma*lastValue
		lastValue = values[i]
	}

	// Normalize advantages
	meanAdv := 0.0
	stdAdv := 0.0
	for _, adv := range advantages {
		meanAdv += adv
	}
	meanAdv /= float64(len(advantages))
	for _, adv := range advantages {
		diff := adv - meanAdv
		stdAdv += diff * diff
	}
	stdAdv = math.Sqrt(stdAdv/float64(len(advantages)) + 1e-8)
	for i := range advantages {
		advantages[i] = (advantages[i] - meanAdv) / stdAdv
	}

	// Update policy network
	for i, state := range states {
		oldProbs := a.GetPolicyProbs(state)
		oldProb := oldProbs[actions[i]]

		// Compute policy gradients
		gradients := make([]float64, a.actionSize)
		for j := range gradients {
			if j == actions[i] {
				ratio := math.Exp(math.Log(oldProbs[j]) - math.Log(oldProb))
				clippedRatio := math.Max(math.Min(ratio, 1.0+a.clipEpsilon), 1.0-a.clipEpsilon)
				gradients[j] = math.Min(ratio*advantages[i], clippedRatio*advantages[i])
			}
		}

		// Scale learning rate based on advantage magnitude but cap it
		effectiveLR := a.learningRate * math.Min(math.Sqrt(math.Abs(advantages[i])), 1.0)
		a.network.Backward(state, oldProbs, gradients, effectiveLR)

		// Update value network with advantages as targets
		value := a.GetValue(state)
		valueGradients := []float64{advantages[i] - value}
		valueOutput := []float64{value}
		a.network.Backward(state, valueOutput, valueGradients, a.learningRate)
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
