package analysis

import (
	"fmt"
	"strings"
	"sync"

	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// PositionResult stores the result of a minimax search for a given position
type PositionResult struct {
	BestMove      game.RPSMove
	Value         float64
	Depth         int
	NodesExplored int
}

// SimpleTranspositionTable caches position evaluations in memory
type SimpleTranspositionTable struct {
	entries map[string]PositionResult
	mu      sync.RWMutex
	hits    int
	misses  int
}

// NewSimpleTranspositionTable creates a new transposition table
func NewSimpleTranspositionTable() *SimpleTranspositionTable {
	return &SimpleTranspositionTable{
		entries: make(map[string]PositionResult),
	}
}

// Get retrieves a cached position result
func (t *SimpleTranspositionTable) Get(position *game.RPSGame) (PositionResult, bool) {
	key := positionToKey(position)

	t.mu.RLock()
	result, found := t.entries[key]
	t.mu.RUnlock()

	if found {
		t.mu.Lock()
		t.hits++
		t.mu.Unlock()
	} else {
		t.mu.Lock()
		t.misses++
		t.mu.Unlock()
	}

	return result, found
}

// Put stores a position result in the cache
func (t *SimpleTranspositionTable) Put(position *game.RPSGame, result PositionResult) {
	key := positionToKey(position)

	t.mu.Lock()
	t.entries[key] = result
	t.mu.Unlock()
}

// GetStats returns cache statistics
func (t *SimpleTranspositionTable) GetStats() (int, int, float64) {
	t.mu.RLock()
	hits := t.hits
	misses := t.misses
	t.mu.RUnlock()

	// Calculate hit rate
	total := hits + misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100.0
	}

	return hits, misses, hitRate
}

// Size returns the number of entries in the cache
func (t *SimpleTranspositionTable) Size() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.entries)
}

// Clear empties the cache
func (t *SimpleTranspositionTable) Clear() {
	t.mu.Lock()
	t.entries = make(map[string]PositionResult)
	t.hits = 0
	t.misses = 0
	t.mu.Unlock()
}

// positionToKey generates a string key from a position
func positionToKey(position *game.RPSGame) string {
	// Simple representation of board state as a string
	var sb strings.Builder

	// Encode board
	for i := 0; i < 9; i++ {
		card := position.Board[i]
		if card.Owner == game.NoPlayer {
			sb.WriteString(".")
		} else {
			var symbol string
			switch card.Type {
			case game.Rock:
				symbol = "R"
			case game.Paper:
				symbol = "P"
			case game.Scissors:
				symbol = "S"
			}

			if card.Owner == game.Player1 {
				sb.WriteString(symbol)
			} else {
				sb.WriteString(strings.ToLower(symbol))
			}
		}
	}

	// Encode current player
	sb.WriteString(fmt.Sprintf("|%d", position.CurrentPlayer))

	// Encode hand sizes (don't need exact cards, just counts)
	sb.WriteString(fmt.Sprintf("|%d|%d",
		len(position.Player1Hand), len(position.Player2Hand)))

	return sb.String()
}
