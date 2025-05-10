# Minimax Analyzer Implementation Plan

## Directory Structure

```
alphago_demo/
├── cmd/
│   ├── analyze_model/        # New command for analyzing models
│   │   └── main.go           # Entry point for model analysis 
├── pkg/
│   ├── analysis/             # New package for analysis tools
│   │   ├── minimax.go        # Minimax implementation
│   │   ├── evaluation.go     # Position evaluation functions
│   │   ├── benchmark.go      # Benchmark position management
│   │   └── report.go         # Analysis reporting utilities
│   ├── game/                 # Existing game implementation
│   ├── mcts/                 # Existing MCTS implementation
│   └── neural/               # Existing neural network implementation
└── testdata/
    └── benchmark_positions/  # Collection of test positions
        ├── early_game.json   # Early game scenarios
        ├── midgame.json      # Midgame tactical positions  
        └── endgame.json      # Endgame positions
```

## Immediate Implementation Tasks

### 1. Core Minimax Implementation (pkg/analysis/minimax.go)

```go
package analysis

import (
	"math"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// MinimaxEngine implements an alpha-beta search for RPS card game
type MinimaxEngine struct {
	MaxDepth      int
	NodesEvaluated int
	MaxTime       time.Duration
	StartTime     time.Time
	EvaluationFn  func(*game.RPSGame) float64
}

// NewMinimaxEngine creates a new minimax search engine
func NewMinimaxEngine(maxDepth int, evalFn func(*game.RPSGame) float64) *MinimaxEngine {
	return &MinimaxEngine{
		MaxDepth:     maxDepth,
		EvaluationFn: evalFn,
	}
}

// FindBestMove returns the best move for the current player
func (m *MinimaxEngine) FindBestMove(state *game.RPSGame) (game.RPSMove, float64) {
	m.NodesEvaluated = 0
	m.StartTime = time.Now()
	
	// Initialize alpha-beta bounds
	alpha := math.Inf(-1)
	beta := math.Inf(1)
	
	// Determine if current player is maximizing
	maximizingPlayer := state.CurrentPlayer == game.Player1
	
	// Call minimax search
	value, move := m.minimax(state, m.MaxDepth, alpha, beta, maximizingPlayer)
	
	return move, value
}

// minimax performs alpha-beta pruned minimax search
func (m *MinimaxEngine) minimax(state *game.RPSGame, depth int, alpha, beta float64, maximizingPlayer bool) (float64, game.RPSMove) {
	// Implementation of the minimax algorithm with alpha-beta pruning
	// As outlined in the MINIMAX_ANALYZER_PLAN.md document
}
```

### 2. Position Evaluation (pkg/analysis/evaluation.go)

```go
package analysis

import (
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// StandardEvaluator provides a comprehensive evaluation function
func StandardEvaluator(state *game.RPSGame) float64 {
	if state.IsGameOver() {
		winner := state.GetWinner()
		if winner == game.Player1 {
			return 1000.0 // Large positive value for Player1 win
		} else if winner == game.Player2 {
			return -1000.0 // Large negative value for Player2 win
		}
		return 0.0 // Draw
	}
	
	// Combine multiple evaluation factors
	return materialScore(state) + positionalScore(state) + relationshipScore(state)
}

// materialScore evaluates the material advantage (difference in number of cards)
func materialScore(state *game.RPSGame) float64 {
	p1Cards := state.CountPlayerCards(game.Player1)
	p2Cards := state.CountPlayerCards(game.Player2)
	return float64(p1Cards - p2Cards) * 10.0 // Weight material difference
}

// positionalScore evaluates board control and positioning
func positionalScore(state *game.RPSGame) float64 {
	// Evaluate control of key positions (center, corners)
	// Implementation to follow
	return 0.0
}

// relationshipScore evaluates the RPS relationships between adjacent cards
func relationshipScore(state *game.RPSGame) float64 {
	// Evaluate how cards interact with adjacent cards
	// Implementation to follow
	return 0.0
}
```

### 3. Benchmark Position Structure (pkg/analysis/benchmark.go)

```go
package analysis

import (
	"encoding/json"
	"io/ioutil"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// BenchmarkPosition represents a test position with known best moves
type BenchmarkPosition struct {
	Name             string         `json:"name"`
	Description      string         `json:"description"`
	Difficulty       int            `json:"difficulty"` // 1-5 scale
	GameState        *game.RPSGame  `json:"game_state"`
	OptimalMoves     []game.RPSMove `json:"optimal_moves"`
	Category         string         `json:"category"` // "early", "midgame", "endgame"
	MinimaxDepth     int            `json:"minimax_depth"` // Recommended search depth
	StrategicTheme   string         `json:"theme"`
}

// BenchmarkSuite is a collection of benchmark positions
type BenchmarkSuite struct {
	Positions []*BenchmarkPosition `json:"positions"`
	Name      string               `json:"name"`
}

// LoadBenchmarkSuite loads positions from a file
func LoadBenchmarkSuite(path string) (*BenchmarkSuite, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	
	var suite BenchmarkSuite
	err = json.Unmarshal(data, &suite)
	if err != nil {
		return nil, err
	}
	
	return &suite, nil
}
```

### 4. Analysis Report Structure (pkg/analysis/report.go)

```go
package analysis

import (
	"fmt"
	"time"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/game"
)

// PositionAnalysis contains analysis results for a single position
type PositionAnalysis struct {
	Position          *BenchmarkPosition
	AgentMove         game.RPSMove
	OptimalMove       game.RPSMove
	IsOptimal         bool
	MoveQuality       float64 // 0-1 scale
	DecisionTime      time.Duration
	NodesEvaluated    int
	AgentEvaluation   float64
	MinimaxEvaluation float64
}

// AgentAnalysisReport contains a complete analysis of an agent
type AgentAnalysisReport struct {
	AgentName           string
	PositionAnalyses    []*PositionAnalysis
	OptimalMoveRate     float64
	AverageMoveQuality  float64
	AverageDecisionTime time.Duration
	TotalNodesEvaluated int
	BlunderRate         float64
}

// NewAgentAnalysisReport creates a new analysis report
func NewAgentAnalysisReport(agentName string) *AgentAnalysisReport {
	return &AgentAnalysisReport{
		AgentName:        agentName,
		PositionAnalyses: make([]*PositionAnalysis, 0),
	}
}

// AddPositionAnalysis adds a position analysis to the report
func (r *AgentAnalysisReport) AddPositionAnalysis(analysis *PositionAnalysis) {
	r.PositionAnalyses = append(r.PositionAnalyses, analysis)
}

// CalculateMetrics computes aggregate metrics for the report
func (r *AgentAnalysisReport) CalculateMetrics() {
	// Calculate metrics from individual position analyses
}

// GenerateTextReport creates a text-based report
func (r *AgentAnalysisReport) GenerateTextReport() string {
	// Generate formatted text report
	return ""
}

// SaveToFile saves the report to a file
func (r *AgentAnalysisReport) SaveToFile(path string) error {
	// Save report as JSON or markdown
	return nil
}
```

### 5. Command Line Interface (cmd/analyze_model/main.go)

```go
package main

import (
	"flag"
	"fmt"
	"os"
	
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/analysis"
	"github.com/zachbeta/neural_rps/alphago_demo/pkg/neural"
)

func main() {
	// Parse command line arguments
	modelPath := flag.String("model", "", "Path to model file (policy network)")
	valueModelPath := flag.String("value-model", "", "Path to value network model file")
	benchmarkPath := flag.String("positions", "testdata/benchmark_positions/standard.json", "Path to benchmark positions")
	depth := flag.Int("depth", 5, "Minimax search depth")
	outputPath := flag.String("output", "", "Output file for analysis report")
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	
	flag.Parse()
	
	if *modelPath == "" {
		fmt.Println("Error: Model path is required")
		flag.Usage()
		os.Exit(1)
	}
	
	// Load benchmark positions
	fmt.Println("Loading benchmark positions...")
	benchmarks, err := analysis.LoadBenchmarkSuite(*benchmarkPath)
	if err != nil {
		fmt.Printf("Error loading benchmark positions: %v\n", err)
		os.Exit(1)
	}
	
	// Load model
	fmt.Printf("Loading model from %s...\n", *modelPath)
	agent, err := loadAgent(*modelPath, *valueModelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}
	
	// Initialize minimax engine
	minimaxEngine := analysis.NewMinimaxEngine(*depth, analysis.StandardEvaluator)
	
	// Analyze agent on benchmark positions
	fmt.Printf("Analyzing %s against %d benchmark positions...\n", agent.Name(), len(benchmarks.Positions))
	report := analysis.NewAgentAnalysisReport(agent.Name())
	
	// Run analysis
	for _, position := range benchmarks.Positions {
		// Perform analysis on each position
		// Add results to report
	}
	
	// Calculate metrics
	report.CalculateMetrics()
	
	// Print report summary
	fmt.Println(report.GenerateTextReport())
	
	// Save report if output path provided
	if *outputPath != "" {
		err = report.SaveToFile(*outputPath)
		if err != nil {
			fmt.Printf("Error saving report: %v\n", err)
		} else {
			fmt.Printf("Analysis report saved to %s\n", *outputPath)
		}
	}
}

// loadAgent loads a model from file
func loadAgent(policyPath, valuePath string) (Agent, error) {
	// Implementation to load various agent types
	return nil, nil
}
```

## Next Steps Timeline

### Week 1: Core Implementation

1. **Day 1**: Create the directory structure and initial files
   - Set up `pkg/analysis` package
   - Implement basic test scaffolding
   
2. **Day 2**: Implement basic minimax algorithm
   - Core search algorithm with alpha-beta pruning
   - Simple position evaluation function
   - Testing with simple positions

3. **Day 3**: Develop position evaluation components
   - Material evaluation
   - Position evaluation
   - Card relationship evaluation
   - Test and tune weights

### Week 2: Position Benchmarks & Analysis

4. **Day 4**: Create benchmark position framework
   - Define position serialization format
   - Create sample positions for early/mid/endgame
   - Validate positions with minimax

5. **Day 5**: Build analysis reporting system
   - Implement analysis metrics
   - Create text and JSON report formats
   - Add visualization helpers

6. **Day 6**: Create command line interface
   - Build standalone analysis command
   - Implement model loading system
   - Add documentation and examples

### Week 3: Integration & Extension

7. **Day 7**: Integration with training system
   - Add hooks to evaluate models during training
   - Create automated benchmark reports
   - Add periodic model quality checks

8. **Day 8**: Code cleanup and optimization
   - Profile and optimize search algorithm
   - Add caching and move ordering
   - Documentation and testing improvements

## Expected Results

1. A robust minimax implementation that can analyze RPS card game positions
2. A suite of benchmark positions representing different game phases
3. Tools to evaluate neural network models against optimal play
4. Insights into the strengths and weaknesses of our training methods

## Future Extensions

1. Endgame tablebase for positions with few cards remaining
2. Integration with MCTS for hybrid search approaches
3. Automated position generation for more comprehensive testing
4. Interactive visualization of decision trees 