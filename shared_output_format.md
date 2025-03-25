# Neural RPS - Shared Output Format

To enable direct comparison between implementations, all versions should follow this consistent output format.

## Format Sections

Each implementation's output should contain these sections in this order:

1. **Header & Implementation Info**
2. **Network Architecture**
3. **Training Process**
4. **Model Predictions**
5. **Model Parameters (Optional)**

## Section Details

### 1. Header & Implementation Info
```
==================================================
Neural Rock Paper Scissors - [IMPLEMENTATION_LANGUAGE] Implementation
==================================================
Version: [VERSION_NUMBER]
Implementation Type: [IMPLEMENTATION_TYPE]
```

### 2. Network Architecture
```
==================================================
Network Architecture
==================================================
Input Layer: [NUM_NEURONS] neurons ([INPUT_DESCRIPTION])
Hidden Layer: [NUM_NEURONS] neurons ([ACTIVATION_FUNCTION])
Output Layer: [NUM_NEURONS] neurons ([ACTIVATION_FUNCTION])

[OPTIONAL_VISUALIZATION]
```

### 3. Training Process
```
==================================================
Training Process
==================================================
Training Episodes: [NUM_EPISODES]
Final Average Reward: [REWARD_VALUE]
Training Time: [TIME_IN_SECONDS]s

[OPTIONAL_TRAINING_CURVE_OR_METRICS]
```

### 4. Model Predictions
```
==================================================
Model Predictions
==================================================
Input: Opponent played Rock
Output: [ROCK_PROB]% Rock, [PAPER_PROB]% Paper, [SCISSORS_PROB]% Scissors
Prediction: [PREDICTION]

Input: Opponent played Paper
Output: [ROCK_PROB]% Rock, [PAPER_PROB]% Paper, [SCISSORS_PROB]% Scissors
Prediction: [PREDICTION]

Input: Opponent played Scissors
Output: [ROCK_PROB]% Rock, [PAPER_PROB]% Paper, [SCISSORS_PROB]% Scissors
Prediction: [PREDICTION]
```

### 5. Model Parameters (Optional)
```
==================================================
Model Parameters (Optional)
==================================================
[WEIGHTS_AND_BIASES_SUMMARY]
```

## Implementation Notes

1. The output should be written to a file named `[IMPLEMENTATION_NAME]_output.txt` in the project root directory.
2. The optional sections can be omitted if the implementation doesn't support them.
3. The visualization and training curve sections should use ASCII art for consistency.
4. The prediction section must include predictions for all three cases (opponent played Rock, Paper, Scissors).
5. Probability values should be formatted to 2 decimal places (e.g., 52.34%). 