# Minimax Position Caching Plan

This document outlines our incremental approach to implementing position caching for the minimax analyzer. Like tacking in windsurfing, we'll make a series of small adjustments based on performance data rather than overengineering upfront.

## Problem Statement

When analyzing RPS card game positions:

- The same positions are often evaluated multiple times
- Deeper searches (depth > 4) can become computationally expensive
- Training neural networks could benefit from background position analysis
- Persistently storing analysis results could speed up future work

## Implementation Stages

### Stage 1: Simple Transposition Table

A lightweight in-memory cache with minimal complexity:

```go
// SimpleTranspositionTable caches position evaluations in memory
type SimpleTranspositionTable struct {
    entries map[string]PositionResult
    mu      sync.RWMutex
    hits    int
    misses  int
}

type PositionResult struct {
    BestMove      game.RPSMove
    Value         float64
    Depth         int
    NodesExplored int
}

func NewSimpleTranspositionTable() *SimpleTranspositionTable {
    return &SimpleTranspositionTable{
        entries: make(map[string]PositionResult),
    }
}

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

func (t *SimpleTranspositionTable) Put(position *game.RPSGame, result PositionResult) {
    key := positionToKey(position)
    
    t.mu.Lock()
    t.entries[key] = result
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
```

#### Integration with Minimax

```go
func (m *MinimaxEngine) FindBestMove(state *game.RPSGame) (game.RPSMove, float64) {
    // If we have a transposition table, check it first
    if m.TranspositionTable != nil {
        if result, found := m.TranspositionTable.Get(state); found {
            // Only use cached result if it was searched at sufficient depth
            if result.Depth >= m.MaxDepth {
                return result.BestMove, result.Value
            }
        }
    }
    
    // Perform regular search
    m.NodesEvaluated = 0
    m.StartTime = time.Now()
    
    // Initialize alpha-beta bounds
    alpha := math.Inf(-1)
    beta := math.Inf(1)
    
    // Determine if current player is maximizing
    maximizingPlayer := state.CurrentPlayer == game.Player1
    
    // Call minimax search
    value, move := m.minimax(state, m.MaxDepth, alpha, beta, maximizingPlayer)
    
    // Cache the result
    if m.TranspositionTable != nil {
        m.TranspositionTable.Put(state, PositionResult{
            BestMove:      move,
            Value:         value,
            Depth:         m.MaxDepth,
            NodesExplored: m.NodesEvaluated,
        })
    }
    
    return move, value
}
```

### Stage 2: LRU Cache (If Needed)

If memory becomes a constraint, implement an LRU eviction policy:

```go
// LRUCache implements a thread-safe LRU cache with limited size
type LRUCache struct {
    capacity int
    mu       sync.RWMutex
    items    map[string]*list.Element
    queue    *list.List
    hits     int
    misses   int
}

// Only implement if benchmarks show we're approaching memory limits
```

### Stage 3: Background Worker (If Needed)

For integrating with training, a background analysis worker:

```go
// GameTreeWorker analyzes positions in the background
type GameTreeWorker struct {
    positionQueue  chan *game.RPSGame
    resultQueue    chan PositionResult
    minimaxEngine  *analysis.MinimaxEngine
    workerCount    int
}

// Only implement if training integration benefits are demonstrated
```

### Stage 4: Persistent Storage (If Needed)

For long-term storage of analysis results:

```go
// DiskCache persists positions to disk
type DiskCache struct {
    cacheDir string
    memory   map[string]string // Maps position hash to filename
}

// Only implement if reuse between runs proves valuable
```

## Decision Metrics

We'll track these metrics to guide our implementation decisions:

### Performance Metrics

1. **Cache Hit Rate**: Percentage of positions found in cache
   - Below 10%: Cache may not be worth the overhead
   - 10-30%: Simple cache is sufficient
   - Above 30%: Consider more advanced caching

2. **Memory Usage**: Size of the cache in MB
   - Below 100MB: Simple cache is fine
   - 100-500MB: Consider LRU eviction
   - Above 500MB: Implement memory constraints

3. **Search Speed**: Positions evaluated per second
   - Current baseline: TBD positions/second
   - Speed improvement goal: 30%+ improvement to justify complexity

### Implementation Triggers

| Metric | Threshold | Next Stage |
|--------|-----------|------------|
| Cache Hit Rate | > 30% | Continue using/improving Stage 1 |
| Memory Usage | > 250MB | Implement Stage 2 (LRU) |
| Training Integration | Need background analysis | Implement Stage 3 (Worker) |
| Multiple Analysis Runs | Need persistent results | Implement Stage 4 (Disk) |

## Measurement Approach

After implementing Stage 1:

1. Run benchmark tests with and without caching
2. Measure and report:
   - Positions analyzed 
   - Cache hits/misses
   - Memory usage
   - Search speed improvement
   - Search depth impact on metrics

Based on these measurements, we'll decide whether to proceed to later stages. 