package game

import (
	"math/rand"
)

// CardType represents the type of card (Warrior, Mage, or Archer)
type CardType int

const (
	Warrior CardType = iota
	Mage
	Archer
)

// Card represents a card in the game
type Card struct {
	Type CardType
}

// NewCard creates a new card of the specified type
func NewCard(cardType CardType) Card {
	return Card{Type: cardType}
}

// Name returns the string representation of the card type
func (c Card) Name() string {
	switch c.Type {
	case Warrior:
		return "Warrior"
	case Mage:
		return "Mage"
	case Archer:
		return "Archer"
	default:
		return "Unknown"
	}
}

// Environment represents the game environment
type Environment struct {
	lastPlayed   Card
	hand         []Card
	opponentHand []Card
	validActions []CardType
}

// NewEnvironment creates a new game environment
func NewEnvironment() *Environment {
	return &Environment{
		hand:         make([]Card, 3),
		opponentHand: make([]Card, 3),
		validActions: make([]CardType, 0, 3),
	}
}

// Reset resets the environment to its initial state
func (e *Environment) Reset() {
	// Initialize hands with random cards
	e.hand = make([]Card, 3)
	e.opponentHand = make([]Card, 3)

	for i := 0; i < 3; i++ {
		e.hand[i] = NewCard(CardType(rand.Intn(3)))
		e.opponentHand[i] = NewCard(CardType(rand.Intn(3)))
	}
	e.lastPlayed = Card{}
	e.updateValidActions()
}

// GetState returns the current state of the environment as a vector
func (e *Environment) GetState() []float64 {
	state := make([]float64, 9) // 3 for last played + 3 for hand + 3 for opponent hand

	// Last played card (one-hot encoding)
	if e.lastPlayed.Type == Warrior {
		state[0] = 1
	} else if e.lastPlayed.Type == Mage {
		state[1] = 1
	} else if e.lastPlayed.Type == Archer {
		state[2] = 1
	}

	// Current hand (one-hot encoding)
	for i := 0; i < 3; i++ {
		if i < len(e.hand) {
			switch e.hand[i].Type {
			case Warrior:
				state[3+i] = 1
			case Mage:
				state[4+i] = 1
			case Archer:
				state[5+i] = 1
			}
		}
	}

	return state
}

// Step takes an action and returns the reward and whether the episode is done
func (e *Environment) Step(action CardType) (float64, bool) {
	if !e.IsValidAction(action) {
		return -1.0, true
	}

	// Play the card
	e.lastPlayed = NewCard(action)

	// Remove the played card from hand
	for i, card := range e.hand {
		if card.Type == action {
			// Shift remaining cards left
			copy(e.hand[i:], e.hand[i+1:])
			e.hand = e.hand[:len(e.hand)-1]
			break
		}
	}

	// Opponent plays a random card
	opponentAction := e.opponentHand[rand.Intn(len(e.opponentHand))]

	// Calculate reward
	reward := e.calculateReward(action, opponentAction.Type)

	// Remove opponent's played card
	for i, card := range e.opponentHand {
		if card.Type == opponentAction.Type {
			copy(e.opponentHand[i:], e.opponentHand[i+1:])
			e.opponentHand = e.opponentHand[:len(e.opponentHand)-1]
			break
		}
	}

	// Update valid actions for next turn
	e.updateValidActions()

	// Check if game is done
	done := len(e.hand) == 0 || len(e.opponentHand) == 0

	return reward, done
}

// IsValidAction checks if the given action is valid
func (e *Environment) IsValidAction(action CardType) bool {
	for _, validAction := range e.validActions {
		if validAction == action {
			return true
		}
	}
	return false
}

// updateValidActions updates the list of valid actions based on the current hand
func (e *Environment) updateValidActions() {
	e.validActions = e.validActions[:0]
	for _, card := range e.hand {
		e.validActions = append(e.validActions, card.Type)
	}
}

// calculateReward calculates the reward for a given action against the opponent's action
func (e *Environment) calculateReward(action, opponentAction CardType) float64 {
	// Rock Paper Scissors rules
	switch action {
	case Warrior:
		switch opponentAction {
		case Mage:
			return -1.0
		case Archer:
			return 1.0
		default:
			return 0.0
		}
	case Mage:
		switch opponentAction {
		case Archer:
			return -1.0
		case Warrior:
			return 1.0
		default:
			return 0.0
		}
	case Archer:
		switch opponentAction {
		case Warrior:
			return -1.0
		case Mage:
			return 1.0
		default:
			return 0.0
		}
	default:
		return 0.0
	}
}
