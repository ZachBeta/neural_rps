# Neural Network Training and Comparison for RPS Card Game

This document describes our approach to training and comparing neural networks for the Rock-Paper-Scissors card game.

## Training Approach

We train the neural networks using self-play reinforcement learning similar to AlphaGo/AlphaZero:

1. **Self-Play Generation**: The AI plays games against itself, using Monte Carlo Tree Search (MCTS) guided by the current policy and value networks to generate high-quality moves.

2. **Training Data Collection**: From these self-play games, we collect:
   - Board positions
   - Action probabilities (from MCTS visit counts)
   - Game outcomes

3. **Neural Network Training**: Using this data, we train:
   - Policy Network: Predicts the probability distribution over possible moves
   - Value Network: Predicts the expected outcome from a given position

4. **Iteration**: The improved networks are then used for the next round of self-play, creating a cycle of continuous improvement.

## Network Complexity Metrics

The training and comparison tools now display detailed network complexity metrics:

### Architecture Details
- **Network Structure**: Input, hidden, and output layer sizes for both policy and value networks
- **Total Neurons**: Combined count of input, hidden, and output neurons
- **Total Connections**: Number of connections (weights) between neurons
- **Total Parameters**: Number of trainable parameters (weights + biases)
- **Memory Footprint**: Estimated memory usage in kilobytes

### Performance Metrics
- **Training Speed**: Examples processed per second, batches per second
- **Loss Trends**: Initial vs best loss values, showing improvement percentages
- **Training Data Distribution**: Percentage of win/loss/draw examples

This information helps us understand how model complexity affects performance and provides insight into training efficiency.

## Model Comparisons

We compare different models to understand what factors lead to better performance:

### Training Configurations

We currently test several variables:

1. **Number of Self-Play Games**:
   - Small dataset (100 games)
   - Large dataset (1000 games)

2. **Network Size**:
   - Small networks (64 hidden units)
   - Large networks (128 hidden units)

3. **Training Epochs**:
   - Short training (5 epochs)
   - Long training (10 epochs)

### Evaluation Methodology

Models are evaluated through direct competition:
- Tournament format with 30-100 games
- Alternating first player to ensure fairness
- Reporting win percentages and statistical significance

## Current Findings

In our latest experiments:

- **Model 1**: 100 self-play games, 64 hidden units, 5 epochs
- **Model 2**: 1000 self-play games, 128 hidden units, 10 epochs

Results showed a win rate of approximately:
- Model 1: ~40-45%
- Model 2: ~55-60%

This suggests that more training data, larger network size, and more training epochs lead to improved performance.

## How to Run Comparisons

You can train and compare models using these commands:

```bash
# Train with default parameters (quick)
make train-and-compete

# Train with large dataset comparison (100 vs 1000 games - takes longer)
make train-large-comparison

# Compare pre-trained models in extensive tournament
make compare-models
```

Results are saved in the `alphago_demo/results/` directory for analysis.

## Future Experiments

Potential areas for future investigation:

1. **Neural Network Architecture**: Testing deeper networks or different activation functions
2. **MCTS Parameters**: Adjusting simulation count, exploration constants
3. **Policy Enhancements**: Incorporating action selection into policy prediction
4. **Transfer Learning**: Applying knowledge from simpler games to more complex ones
5. **Parallel Training**: Using multiple cores/machines to speed up training

## Visualization

Tournament results can be visualized using the model comparison tool's output files. These contain detailed performance metrics that can be graphed to understand learning curves and relative performance. 