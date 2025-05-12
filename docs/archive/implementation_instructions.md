# Implementation Instructions for Shared Output Format

This document provides guidance on how to modify each implementation to follow the shared output format defined in `shared_output_format.md`.

## General Steps for All Implementations

1. Create a function to format and print the standardized output
2. Ensure each implementation can extract the required metrics
3. Create functions to format model predictions in consistent format
4. Update the main loop to call these functions

## C++ Implementation

### Files to modify:
- `cpp_implementation/src/main.cpp` - For the demo version
- `cpp_implementation/src/NeuralNetwork.cpp` - For model details
- `cpp_implementation/include/NeuralNetwork.h` - For output utility functions

### Code Changes:

1. Add output utility functions to `NeuralNetwork.h`:

```cpp
// Add to NeuralNetwork.h
void printStandardizedOutput(const std::string& filename);
void printModelArchitecture(std::ostream& out);
void printTrainingProcess(std::ostream& out, int episodes, float finalReward, float trainingTime);
void printModelPredictions(std::ostream& out);
void printModelParameters(std::ostream& out);
```

2. Implement these functions in `NeuralNetwork.cpp`:

```cpp
// Add to NeuralNetwork.cpp
void NeuralNetwork::printStandardizedOutput(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    // Header
    outFile << "==================================================\n";
    outFile << "Neural Rock Paper Scissors - C++ Implementation\n";
    outFile << "==================================================\n";
    outFile << "Version: 1.0\n";
    outFile << "Implementation Type: Neural Network with PPO\n\n";
    
    // Network Architecture
    printModelArchitecture(outFile);
    
    // Training Process
    printTrainingProcess(outFile, _episodes, _finalReward, _trainingTime);
    
    // Model Predictions
    printModelPredictions(outFile);
    
    // Model Parameters
    printModelParameters(outFile);
    
    outFile.close();
}

// Implement the other functions...
```

3. Update `main.cpp` to call the output function after training:

```cpp
// In main() after training
network.printStandardizedOutput("cpp_demo_output.txt");
```

## Golang Implementation

### Files to modify:
- `golang_implementation/cmd/neural_rps/main.go`
- `golang_implementation/pkg/network/network.go`

### Code Changes:

1. Add output utility functions to `network.go`:

```go
// Add to package network

// PrintStandardizedOutput prints the network details in standardized format
func (n *Network) PrintStandardizedOutput(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Print header
    fmt.Fprintf(file, "==================================================\n")
    fmt.Fprintf(file, "Neural Rock Paper Scissors - Go Implementation\n")
    fmt.Fprintf(file, "==================================================\n")
    fmt.Fprintf(file, "Version: 1.0\n")
    fmt.Fprintf(file, "Implementation Type: Neural Network with PPO\n\n")
    
    // Print architecture
    n.printModelArchitecture(file)
    
    // Print training process
    n.printTrainingProcess(file)
    
    // Print model predictions
    n.printModelPredictions(file)
    
    // Print model parameters
    n.printModelParameters(file)
    
    return nil
}

// Implement helper methods...
```

2. Update `main.go` to call the output function:

```go
// After training
err := network.PrintStandardizedOutput("go_demo_output.txt")
if err != nil {
    log.Fatalf("Failed to write output: %v", err)
}
```

## AlphaGo Demo (Tic-Tac-Toe)

### Files to modify:
- `alphago_demo/cmd/tictactoe/main.go`
- `alphago_demo/pkg/network/network.go`

### Code Changes:

1. Add output utility functions to the network package:

```go
// Add to the network package

// PrintStandardizedOutput prints the network details in standardized format
func (n *Network) PrintStandardizedOutput(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Print header
    fmt.Fprintf(file, "==================================================\n")
    fmt.Fprintf(file, "Neural Game AI - Go Implementation (AlphaGo-style)\n")
    fmt.Fprintf(file, "==================================================\n")
    fmt.Fprintf(file, "Version: 1.0\n")
    fmt.Fprintf(file, "Implementation Type: AlphaGo-style MCTS with Neural Networks\n\n")
    
    // Print architecture
    n.printModelArchitecture(file)
    
    // Print training process
    n.printTrainingProcess(file)
    
    // Print model predictions (adapted for Tic-Tac-Toe)
    n.printModelPredictions(file)
    
    // Print model parameters
    n.printModelParameters(file)
    
    return nil
}

// Implement helper methods...
```

2. Update the main function to call the output function:

```go
// After training
err := network.PrintStandardizedOutput("alphago_demo_output.txt")
if err != nil {
    log.Fatalf("Failed to write output: %v", err)
}
```

## Legacy C++ Implementation

### Files to modify:
- `legacy_cpp_implementation/src/main.cpp`
- `legacy_cpp_implementation/include/NeuralNetwork.h`
- `legacy_cpp_implementation/src/NeuralNetwork.cpp`

### Code Changes:

Similar to the C++ implementation:

1. Add output utility functions to the header file
2. Implement them in the source file
3. Call the output function from the main function

## Shared Visualization Functions

For ASCII visualizations of the network architecture, each implementation should implement a similar approach to display the network structure. This can be implemented with similar structure in each language.

## Output Format Validator

Consider creating a simple validator script to check that each implementation's output follows the standardized format:

```python
def validate_output(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Check for required sections
    required_sections = [
        "Neural Rock Paper Scissors",
        "Network Architecture",
        "Training Process",
        "Model Predictions"
    ]
    
    for section in required_sections:
        if section not in content:
            print(f"Missing section: {section} in {filename}")
            return False
    
    print(f"Output format validated for {filename}")
    return True

# Validate all outputs
validate_output("cpp_demo_output.txt")
validate_output("go_demo_output.txt")
validate_output("legacy_cpp_demo_output.txt")
validate_output("alphago_demo_output.txt")
``` 