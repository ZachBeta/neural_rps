# Graceful Exit & Checkpoint System Implementation

## Overview

This document outlines the implementation plan for adding robust graceful exit handling and checkpointing to our neural network training systems. This will allow long-running training processes to be safely interrupted and resumed later without losing progress.

## Core Requirements

1. Handle interruption signals (Ctrl+C, SIGTERM) gracefully
2. Save training state at regular intervals and upon interruption
3. Provide detailed progress reporting during training
4. Enable resumption of training from the last saved state
5. Maintain backward compatibility with existing training workflows

## Implementation Details

### Signal Handling

```go
import (
    "os"
    "os/signal"
    "syscall"
    "time"
    "fmt"
)

// Setup signal handling in main training functions
func setupGracefulExit(checkpointFunc func()) {
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)

    go func() {
        <-c
        fmt.Println("\n\nGraceful shutdown initiated...")
        checkpointFunc()
        fmt.Println("Training state saved. Exiting.")
        os.Exit(0)
    }()
}
```

### Progress Tracking

```go
// Common progress struct for all training methods
type TrainingProgress struct {
    StartTime           time.Time
    ElapsedTime         time.Duration
    CurrentPhase        string // "self-play", "training", "evaluation", etc.
    CompletedGames      int
    TotalGames          int
    CompletedEpochs     int    // For AlphaGo
    CurrentGeneration   int    // For NEAT
    CurrentEvaluation   int    // Number of evaluations in current generation
    TotalEvaluations    int    // Total evaluations needed in current generation
    LastCheckpoint      time.Time
    CheckpointInterval  time.Duration
    EstimatedCompletion time.Time
    CPUUtilization      float64 // Percentage
    MemoryUsage         uint64  // Bytes
    CurrentLoss         float64 // For supervised learning
    BestFitness         float64 // For NEAT
}

// Print progress update to console
func (p *TrainingProgress) PrintUpdate() {
    // Format and print current progress
    // Include time estimates, completion percentage, etc.
}
```

### AlphaGo-Style Training Checkpoints

```go
type AlphaGoCheckpoint struct {
    Progress       TrainingProgress
    PolicyNetwork  *neural.RPSPolicyNetwork
    ValueNetwork   *neural.RPSValueNetwork
    TrainingExamples []training.Example
    CurrentEpoch   int
    Hyperparameters struct {
        LearningRate float64
        BatchSize    int
        // Other hyperparameters
    }
}

func SaveAlphaGoCheckpoint(checkpoint AlphaGoCheckpoint, path string) error {
    // Serialize and save checkpoint to file
}

func LoadAlphaGoCheckpoint(path string) (AlphaGoCheckpoint, error) {
    // Load and deserialize checkpoint from file
}
```

### NEAT Training Checkpoints

```go
type NEATCheckpoint struct {
    Progress      TrainingProgress
    Population    *neat.Population
    Species       map[int][]int
    BestGenome    *neat.Genome
    CurrentConfig neat.Config
    InnovCounter  int
}

func SaveNEATCheckpoint(checkpoint NEATCheckpoint, path string) error {
    // Serialize and save checkpoint to file
}

func LoadNEATCheckpoint(path string) (NEATCheckpoint, error) {
    // Load and deserialize checkpoint from file
}
```

## Integration with Existing Code

### For AlphaGo Training Loop

```go
func trainAlphaGoAgent(agent Agent, selfPlayGames, mctsSimulations int, outputDir string) {
    // Initial setup
    
    // Setup checkpointing
    checkpointInterval := 10 * time.Minute
    lastCheckpoint := time.Now()
    checkpointPath := fmt.Sprintf("%s/checkpoint_%s.json", outputDir, agent.Name)
    
    // Check for existing checkpoint
    checkpoint, err := LoadAlphaGoCheckpoint(checkpointPath)
    if err == nil {
        // Resume from checkpoint
        // ...
    }
    
    // Setup graceful exit
    setupGracefulExit(func() {
        SaveAlphaGoCheckpoint(checkpoint, checkpointPath)
    })
    
    // Main training loop with progress updates and periodic checkpointing
    for epoch := startEpoch; epoch < totalEpochs; epoch++ {
        // Training code
        
        // Update progress
        progress.CurrentEpoch = epoch
        progress.ElapsedTime = time.Since(progress.StartTime)
        progress.PrintUpdate()
        
        // Periodic checkpointing
        if time.Since(lastCheckpoint) > checkpointInterval {
            SaveAlphaGoCheckpoint(checkpoint, checkpointPath)
            lastCheckpoint = time.Now()
        }
    }
    
    // Final saving of trained model
}
```

### For NEAT Training Loop

```go
func (p *Population) Evolve(cfg Config, threads int) *Genome {
    // Initial setup
    
    // Setup checkpointing
    checkpointInterval := 10 * time.Minute
    lastCheckpoint := time.Now()
    checkpointPath := fmt.Sprintf("output/neat_checkpoint_gen%03d.json", p.Progress.CurrentGeneration)
    
    // Check for existing checkpoint
    checkpoint, err := LoadNEATCheckpoint(checkpointPath)
    if err == nil {
        // Resume from checkpoint
        // ...
    }
    
    // Setup graceful exit
    setupGracefulExit(func() {
        SaveNEATCheckpoint(checkpoint, checkpointPath)
    })
    
    // Main evolution loop with progress updates and periodic checkpointing
    for gen := startGen; gen < cfg.Generations; gen++ {
        // Generation code
        
        // Update progress
        p.Progress.CurrentGeneration = gen
        p.Progress.ElapsedTime = time.Since(p.Progress.StartTime)
        p.Progress.PrintUpdate()
        
        // Save the best genome for this generation
        bestGenome := p.GetBestGenome()
        bestPolicy, bestValue := bestGenome.ToNetworks()
        bestPolicy.SaveToFile(fmt.Sprintf("output/neat_gen%02d_policy.model", gen))
        bestValue.SaveToFile(fmt.Sprintf("output/neat_gen%02d_value.model", gen))
        
        // Periodic checkpointing
        if time.Since(lastCheckpoint) > checkpointInterval {
            SaveNEATCheckpoint(checkpoint, checkpointPath)
            lastCheckpoint = time.Now()
        }
    }
    
    return p.GetBestGenome()
}
```

## Command Line Interface Enhancements

```
Usage: go run cmd/train_models/main.go [options]

Options:
  --checkpoint-interval duration   Time between automatic checkpoints (default 10m)
  --resume                         Resume from latest checkpoint if available
  --progress-update-interval int   Progress update frequency in seconds (default 30)
  --save-metrics                   Save training metrics to CSV file
```

## Enhanced Training Report

At the end of training (or upon exit), generate a comprehensive report:

```
===== TRAINING REPORT: NEAT Model =====
Start time: 2025-05-10 14:32:17
End time:   2025-05-10 18:45:23
Total duration: 4h 13m 6s

Performance:
- Games played: 15,432
- Games per second: 1.02
- Evaluations: 327,500
- CPU utilization: 92.3%
- Memory usage (peak): 1.2 GB

Model Evolution:
- Initial hidden nodes: 64
- Final hidden nodes: 87
- Connection count: 4,312
- Total parameters: 22,487

Final Performance:
- Best fitness: 0.783
- Tournament win rate: 68.2%

Checkpoint files:
- neat_checkpoint_gen001.json
- neat_checkpoint_gen010.json
- neat_checkpoint_gen020.json
- neat_checkpoint_final.json

Model saved to: output/neat_final_policy.model, output/neat_final_value.model
```

## Expected Benefits

1. **Reliability**: No more lost training progress due to interruptions
2. **Visibility**: Clear progress reporting shows training status at a glance
3. **Flexibility**: Ability to pause/resume long training runs
4. **Resource efficiency**: Can split training across multiple sessions
5. **Better debugging**: More detailed metrics to identify issues

## Implementation Timeline

1. **Phase 1**: Core signal handling and basic checkpoint structure (1 day)
2. **Phase 2**: Progress tracking and reporting (1 day)
3. **Phase 3**: AlphaGo training integration (1 day)
4. **Phase 4**: NEAT training integration (1 day)
5. **Phase 5**: Testing and refinement (1-2 days) 