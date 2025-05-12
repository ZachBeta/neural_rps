# Neural RPS Build System Guide

This document explains the build system structure and Makefile organization for the Neural RPS project.

## 1. Build System Overview

The Neural RPS project uses a hierarchical Makefile structure:
- A top-level Makefile with high-level commands
- Package-specific Makefiles for detailed operations

```
           ┌─────────────────────┐
           │  Top-level Makefile │
           └──────────┬──────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
┌─────────▼────────┐ ┌▼────────────────┐ ┌▼─────────────────┐
│ AlphaGo Makefile │ │ Golang Makefile │ │ C++ Makefile     │
└──────────────────┘ └─────────────────┘ └──────────────────┘
```

### 1.1. Version Compatibility

| Component | Required Version |
|-----------|------------------|
| Go        | 1.19+            |
| CMake     | 3.10+            |
| Make      | 3.8+             |
| Git       | 2.0+             |

### 1.2. Quick Reference

| Operation | Command |
|-----------|---------|
| Build all | `make build` |
| Clean all | `make clean` |
| Run tests | `make test` |
| Train models | `make alphago-train` |
| Run tournament | `make golang-tournament` |

## 2. Top-Level Makefile

The top-level Makefile provides commands that apply across the whole project:

```
neural_rps/
└── Makefile  # Top-level Makefile
```

### 2.1. Main Build Targets

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `build` | Build all implementations | `make build` |
| `build-legacy-cpp` | Build legacy C++ implementation | `make build-legacy-cpp` |
| `build-cpp` | Build C++ implementation | `make build-cpp` |
| `build-go` | Build Golang implementation | `make build-go` |
| `build-alphago` | Build AlphaGo demos | `make build-alphago` |

Example of what the `build` target does:
```makefile
# Example from top-level Makefile
build: build-go build-alphago build-cpp
	@echo "All implementations built successfully"
```

### 2.2. Run Targets

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `run-legacy-cpp` | Run legacy C++ implementation | `make run-legacy-cpp` |
| `run-cpp` | Run C++ simplified demo | `make run-cpp` |
| `run-go` | Run Golang implementation | `make run-go` |
| `run-alphago-ttt` | Run AlphaGo Tic-Tac-Toe demo | `make run-alphago-ttt` |
| `run-alphago-rps` | Run AlphaGo RPS Card Game demo | `make run-alphago-rps` |
| `run-alphago-interactive` | Run interactive AlphaGo game | `make run-alphago-interactive` |

Example of what the `run-alphago-rps` target does:
```makefile
# Example from top-level Makefile
run-alphago-rps: build-alphago
	@echo "Running AlphaGo RPS Card Game demo..."
	cd alphago_demo && make play
```

### 2.3. Test Targets

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `test` | Run all tests | `make test` |
| `test-go` | Run Golang tests | `make test-go` |
| `test-alphago` | Run AlphaGo tests | `make test-alphago` |

Example of what the `test` target does:
```makefile
# Example from top-level Makefile
test: test-go test-alphago
	@echo "All tests passed"
```

### 2.4. Package-Specific Targets

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `alphago-train` | Train AlphaGo models | `make alphago-train` |
| `alphago-tournament` | Run AlphaGo tournament | `make alphago-tournament` |
| `golang-tournament` | Run Golang tournament | `make golang-tournament` |
| `golang-vs-alphago` | Run tournament between Golang and AlphaGo agents | `make golang-vs-alphago` |

Example of what the `golang-vs-alphago` target does:
```makefile
# Example from top-level Makefile
golang-vs-alphago: build-go
	@echo "Running tournament between Golang and AlphaGo agents..."
	@cd golang_implementation && make alphago-vs-alphago-verbose
	@echo "Tournament complete, results in golang_implementation/results/"
```

### 2.5. Utility Targets

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `clean` | Clean build artifacts | `make clean` |
| `install-deps` | Install dependencies | `make install-deps` |
| `fmt` | Format code | `make fmt` |
| `doc` | Generate documentation | `make doc` |

Example of what the `clean` target does:
```makefile
# Example from top-level Makefile
clean:
	@echo "Cleaning all build artifacts..."
	cd alphago_demo && make clean
	cd golang_implementation && make clean
	cd cpp_implementation && make clean
```

## 3. Package-Specific Makefiles

Each implementation has its own Makefile for package-specific operations:

```
neural_rps/
├── alphago_demo/
│   └── Makefile  # AlphaGo-specific Makefile
├── golang_implementation/
│   └── Makefile  # Golang-specific Makefile
└── cpp_implementation/
    └── Makefile  # C++-specific Makefile
```

### 3.1. AlphaGo Demo Makefile

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `build` | Build all AlphaGo binaries | `cd alphago_demo && make build` |
| `test` | Run AlphaGo tests | `cd alphago_demo && make test` |
| `train` | Train AlphaGo models | `cd alphago_demo && make train` |
| `compare` | Compare AlphaGo models | `cd alphago_demo && make compare` |
| `play` | Play against an AlphaGo agent | `cd alphago_demo && make play` |
| `play-interactive` | Play interactively | `cd alphago_demo && make play-interactive` |
| `clean` | Clean build artifacts | `cd alphago_demo && make clean` |

Example of what the `train` target does:
```makefile
# Example from alphago_demo/Makefile
train: build
	@echo "Training AlphaGo models..."
	@mkdir -p output
	@./bin/train_models
	@echo "Models saved to output/ directory"
```

### 3.2. Golang Implementation Makefile

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `build` | Build all Golang binaries | `cd golang_implementation && make build` |
| `test` | Run Golang tests | `cd golang_implementation && make test` |
| `tournament` | Run tournament between different agents | `cd golang_implementation && make tournament` |
| `tournament-verbose` | Run tournament with verbose output | `cd golang_implementation && make tournament-verbose` |
| `alphago-vs-alphago` | Run tournament between AlphaGo models | `cd golang_implementation && make alphago-vs-alphago` |
| `alphago-vs-alphago-verbose` | Run tournament with verbose output | `cd golang_implementation && make alphago-vs-alphago-verbose` |
| `clean` | Clean build artifacts | `cd golang_implementation && make clean` |

Example of what the `tournament` target does:
```makefile
# Example from golang_implementation/Makefile
tournament:
	@mkdir -p bin results
	@go build -o bin/tournament cmd/tournament/main.go
	@echo "Running tournament between AlphaGo and PPO agents..."
	@./bin/tournament --games 50 --alphago-sims 300 --ppo-hidden 128 > results/tournament_results.txt
	@echo "Tournament complete! Results saved to results/tournament_results.txt"
```

### 3.3. C++ Implementation Makefile

| Target | Description | Example Usage |
|--------|-------------|---------------|
| `all` | Build all C++ binaries | `cd cpp_implementation && make all` |
| `neural_rps_demo` | Build the simplified demo | `cd cpp_implementation && make neural_rps_demo` |
| `neural_rps_full` | Build the full implementation | `cd cpp_implementation && make neural_rps_full` |
| `run-demo` | Run the simplified demo | `cd cpp_implementation && make run-demo` |
| `run-full` | Run the full implementation | `cd cpp_implementation && make run-full` |
| `clean` | Clean build artifacts | `cd cpp_implementation && make clean` |

Example of what the `all` target does:
```makefile
# Example from cpp_implementation/Makefile
all: neural_rps_demo neural_rps_full
	@echo "All C++ binaries built successfully"
```

## 4. Build Process Flow

The typical build and execution flow follows this pattern:

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│ Install         │────►│ Build Binaries    │────►│ Train Models     │
│ Dependencies    │     │ (make build)      │     │ (make alphago-   │
│ (make install-  │     │                   │     │ train)           │
│ deps)           │     │                   │     │                  │
└─────────────────┘     └───────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│ View Results    │◄────│ Run Tournament    │◄────│ Setup Tournament │
│ (results/*.txt) │     │ (make golang-     │     │ Parameters       │
│                 │     │ tournament)       │     │                  │
└─────────────────┘     └───────────────────┘     └──────────────────┘
```

## 5. Directory Organization

The build system produces artifacts in these locations:

```
neural_rps/
├── alphago_demo/
│   ├── bin/     # AlphaGo binaries
│   │   ├── train_models     # Model training binary
│   │   ├── rps_card         # RPS demo binary
│   │   ├── play_vs_ai       # Interactive gameplay binary
│   │   ├── tictactoe        # Tic-Tac-Toe demo binary
│   │   └── compare_models   # Model comparison binary
│   └── output/  # Trained models and results
│       ├── rps_policy1.model      # Policy network model
│       ├── rps_value1.model       # Value network model
│       └── training_stats.txt     # Training statistics
├── golang_implementation/
│   ├── bin/     # Golang binaries
│   │   ├── neural_rps           # Main binary
│   │   ├── tournament           # Tournament binary
│   │   └── alphago_vs_alphago   # AlphaGo comparison binary
│   └── results/ # Tournament results
│       ├── tournament_results.txt          # Standard tournament results
│       └── alphago_vs_alphago_results.txt  # AlphaGo comparison results
├── cpp_implementation/
│   └── build/   # C++ build artifacts
│       ├── neural_rps_demo  # Demo binary
│       └── neural_rps_full  # Full implementation binary
└── docs/        # Documentation
    ├── build_system.md           # Build system guide
    ├── integration_guide.md      # Integration guide
    └── interfaces.md             # Interface definitions
```

## 6. Using the Build System

### 6.1. First-Time Setup

When using the project for the first time:

```bash
# Clone the repository
git clone https://github.com/zachbeta/neural_rps.git
cd neural_rps

# Install dependencies
make install-deps

# Build all implementations
make build

# Run tests to verify everything is working
make test
```

### 6.2. Training AlphaGo Models

To train AlphaGo models:

```bash
make alphago-train
```

This will:
1. Build necessary binaries in `alphago_demo/bin/`
2. Run self-play to generate training data
3. Train neural networks on the data
4. Save models to `alphago_demo/output/rps_policy1.model` and `alphago_demo/output/rps_value1.model`

Example output:
```
Building AlphaGo binaries...
AlphaGo binaries built successfully
Training AlphaGo models...
Starting self-play (100 games)...
Self-play complete.
Starting neural network training...
Policy network training: 100%
Value network training: 100%
Models saved to output/ directory
```

### 6.3. Running Tournaments

To run a tournament between different agents:

```bash
make golang-tournament
```

Example output:
```
Building Golang binaries...
Golang binaries built successfully
Running tournament...
Tournament configuration:
- Games: 100
- Agents: AlphaGo, PPO, Random
Tournament in progress...
Tournament complete!
Results saved to results/tournament_results.txt
```

To run a tournament between AlphaGo models with different parameters:

```bash
make golang-vs-alphago
```

Example output:
```
Building Golang binaries...
Golang binaries built successfully
Running tournament between AlphaGo models...
Tournament configuration:
- Games: 100
- AlphaGo-200: 200 MCTS simulations, c_puct=1.0
- AlphaGo-500: 500 MCTS simulations, c_puct=1.5
Tournament in progress...
Tournament complete!
Results saved to results/alphago_vs_alphago_results.txt
```

### 6.4. Building Individual Implementations

To build just the Golang implementation:

```bash
make build-go
```

To build just the AlphaGo demos:

```bash
make build-alphago
```

## 7. Common Build Scenarios

### 7.1. Full Workflow Example

Here's a complete workflow for training and evaluation:

```bash
# Build everything
make build

# Train AlphaGo models
make alphago-train

# Run a tournament to evaluate the models
make golang-tournament

# View the results
cat golang_implementation/results/tournament_results.txt
```

### 7.2. Iterative Development

During development, you might want to quickly rebuild and test:

```bash
# In alphago_demo directory
cd alphago_demo
make clean
make build
make test

# In golang_implementation directory
cd ../golang_implementation
make clean
make build
make test
```

## 8. Troubleshooting

### 8.1. Missing Dependencies

**Problem**: "Command not found" or missing library errors

**Solution**:
```bash
# Install all dependencies
make install-deps

# For specific Go package issues
go get -u github.com/missing/package

# For specific C++ library issues
apt-get install libmissing-dev  # Ubuntu/Debian
brew install missing-lib        # macOS
```

### 8.2. Go Build Failures

**Problem**: "Cannot find package" or "Unknown revision" errors

**Solution**:
```bash
# Go to the specific implementation directory
cd golang_implementation

# Fix module dependencies
go mod tidy
go mod download

# Try building again
make build
```

### 8.3. C++ Build Failures

**Problem**: "CMake configuration" or "Compilation" errors

**Solution**:
```bash
# Go to C++ implementation directory
cd cpp_implementation

# Remove old build artifacts
rm -rf build

# Create a fresh build directory
mkdir build
cd build

# Run CMake with debug output
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build with verbose output
make VERBOSE=1
```

### 8.4. Path Issues

**Problem**: "File not found" errors when running binaries

**Solution**:

Check relative paths in your makefiles. For example, if your `golang_implementation` is trying to load files from `alphago_demo`:

```makefile
# Using incorrect relative path
policyPath := "../alphago_demo/output/rps_policy1.model"

# Using correct relative path with $(CURDIR)
policyPath := "$(CURDIR)/../alphago_demo/output/rps_policy1.model"
```

You can also export environment variables in the makefile:

```makefile
export ALPHAGO_OUTPUT_DIR := $(CURDIR)/../alphago_demo/output
```

### 8.5. Model Loading Issues

**Problem**: "Failed to load model" errors

**Solution**:

1. Verify the model files exist:
```bash
ls -la alphago_demo/output/rps_policy1.model
ls -la alphago_demo/output/rps_value1.model
```

2. Check file permissions:
```bash
chmod 644 alphago_demo/output/*.model
```

3. Make sure the model format is compatible:
```bash
# Re-train models if necessary
make alphago-train
```

### 8.6. Tournament Execution Issues

**Problem**: Tournament hangs or crashes

**Solution**:

1. Run in verbose mode to see detailed logs:
```bash
make golang-tournament-verbose
```

2. Check for memory issues (especially with large neural networks):
```bash
# Limit MCTS simulations to reduce memory usage
export MCTS_SIMULATIONS=100
make golang-vs-alphago
```

## 9. Extending the Build System

### 9.1. Adding New Targets

To add a new target to the top-level Makefile:

1. Define the target and its dependencies
2. Add documentation to the .PHONY line
3. Update this build_system.md document

Example:
```makefile
.PHONY: ... new-target

# New target description
new-target:
    @echo "Running new target..."
    @cd some_directory && do_something
```

### 9.2. Adding a New Implementation

If you add a new implementation or package:

1. Create a directory structure:
```
neural_rps/
└── new_implementation/
    ├── cmd/           # Command-line binaries
    ├── pkg/           # Package code
    ├── tests/         # Tests
    └── Makefile       # Package-specific Makefile
```

2. Create a Makefile in the package directory with standard targets:
```makefile
.PHONY: build test clean

build:
	@echo "Building new implementation..."
	# Build commands here

test:
	@echo "Testing new implementation..."
	# Test commands here

clean:
	@echo "Cleaning new implementation..."
	# Clean commands here
```

3. Add targets in the top-level Makefile:
```makefile
.PHONY: ... build-new run-new test-new

# Build new implementation
build-new:
	@echo "Building new implementation..."
	@cd new_implementation && make build

# Run new implementation
run-new: build-new
	@echo "Running new implementation..."
	@cd new_implementation && ./bin/main

# Test new implementation
test-new:
	@echo "Testing new implementation..."
	@cd new_implementation && make test
```

## 10. Common Makefile Patterns

### 10.1. Consistent Directory Structure

Maintain a consistent directory structure for all implementations:

```makefile
# Create necessary directories
dirs:
	@mkdir -p bin output results
```

### 10.2. Standard Target Names

Use standard target names across all implementations:

```makefile
# Standard targets
.PHONY: build test clean run
```

### 10.3. Verbose Flags

Add verbose options to commands for debugging:

```makefile
# Build with verbose output if VERBOSE is set
build:
ifdef VERBOSE
	go build -v -x -o bin/myapp cmd/main.go
else
	@go build -o bin/myapp cmd/main.go
endif
```

### 10.4. Environment Variables

Use environment variables to configure builds:

```makefile
# Use environment variables for configuration
run:
	@ALPHAGO_SIMS=$${MCTS_SIMULATIONS:-300} ./bin/tournament
```

## 11. Continuous Integration

For CI systems, create a workflow that includes:

```bash
# Install dependencies
make install-deps

# Build all implementations
make build

# Run all tests
make test

# Run specific integration tests
make golang-tournament
```

Example GitHub Actions workflow:
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Go
      uses: actions/setup-go@v2
      with:
        go-version: 1.19
    - name: Install dependencies
      run: make install-deps
    - name: Build
      run: make build
    - name: Test
      run: make test
```

## 12. Build System Testing

The project includes a test script (`test_build_system.sh`) to validate the build system:

```bash
# Run the test script
./test_build_system.sh
```

This script:
- Tests all major build targets
- Validates the directory structure
- Ensures all binaries can be built
- Checks for proper error handling

You should run this script after any significant changes to the build system.

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2023-12-01 | Initial build system |
| 1.1 | 2024-01-15 | Added interactive play mode |
| 1.2 | 2024-03-25 | Standardized all Makefiles |

## 14. Conclusion

The Neural RPS build system provides a structured way to build, test, and run different implementations of neural network-based RPS games. By understanding the hierarchy and organization, you can efficiently work with the codebase and extend it as needed.

For further assistance:
- Check implementation-specific READMEs
- Refer to the [Integration Guide](integration_guide.md)
- See the [Architecture Documentation](../ARCHITECTURE.md) 