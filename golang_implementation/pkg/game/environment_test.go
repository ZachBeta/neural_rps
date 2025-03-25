package game

import (
	"testing"
)

func TestCardCreation(t *testing.T) {
	tests := []struct {
		cardType CardType
		want     string
	}{
		{Warrior, "Warrior"},
		{Mage, "Mage"},
		{Archer, "Archer"},
	}

	for _, tt := range tests {
		card := NewCard(tt.cardType)
		if card.Name() != tt.want {
			t.Errorf("NewCard(%v).Name() = %v, want %v", tt.cardType, card.Name(), tt.want)
		}
	}
}

func TestEnvironmentReset(t *testing.T) {
	env := NewEnvironment()
	env.Reset()

	// Check hand sizes
	if len(env.hand) != 3 {
		t.Errorf("Reset() hand size = %v, want 3", len(env.hand))
	}
	if len(env.opponentHand) != 3 {
		t.Errorf("Reset() opponent hand size = %v, want 3", len(env.opponentHand))
	}

	// Check valid actions
	if len(env.validActions) != 3 {
		t.Errorf("Reset() valid actions size = %v, want 3", len(env.validActions))
	}
}

func TestEnvironmentStep(t *testing.T) {
	env := NewEnvironment()
	env.Reset()

	// Test valid action
	reward, done := env.Step(env.hand[0].Type)
	if done {
		t.Error("Step() with valid action should not be done")
	}

	// Test invalid action
	reward, done = env.Step(CardType(999)) // Invalid card type
	if !done {
		t.Error("Step() with invalid action should be done")
	}
	if reward != -1.0 {
		t.Errorf("Step() with invalid action reward = %v, want -1.0", reward)
	}
}

func TestCalculateReward(t *testing.T) {
	env := NewEnvironment()
	tests := []struct {
		action         CardType
		opponentAction CardType
		want           float64
	}{
		{Warrior, Mage, -1.0},   // Warrior loses to Mage
		{Warrior, Archer, 1.0},  // Warrior beats Archer
		{Warrior, Warrior, 0.0}, // Draw
		{Mage, Archer, -1.0},    // Mage loses to Archer
		{Mage, Warrior, 1.0},    // Mage beats Warrior
		{Mage, Mage, 0.0},       // Draw
		{Archer, Warrior, -1.0}, // Archer loses to Warrior
		{Archer, Mage, 1.0},     // Archer beats Mage
		{Archer, Archer, 0.0},   // Draw
	}

	for _, tt := range tests {
		reward := env.calculateReward(tt.action, tt.opponentAction)
		if reward != tt.want {
			t.Errorf("calculateReward(%v, %v) = %v, want %v",
				tt.action, tt.opponentAction, reward, tt.want)
		}
	}
}

func TestGetState(t *testing.T) {
	env := NewEnvironment()
	env.Reset()

	state := env.GetState()
	if len(state) != 9 {
		t.Errorf("GetState() returned state of length %v, want 9", len(state))
	}

	// Check that state is properly normalized (sum of one-hot encodings)
	sum := 0.0
	for _, v := range state {
		sum += v
	}
	if sum != 3.0 { // 1 for last played + 1 for current hand + 1 for opponent hand
		t.Errorf("GetState() returned state with sum %v, want 3.0", sum)
	}
}
