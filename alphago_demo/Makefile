.PHONY: build test train compare play play-interactive quick-train clean \
	train-parallel tournament-elo profile-train

# Build all AlphaGo binaries
build:
	@echo "Building AlphaGo binaries..."
	@mkdir -p bin
	@go build -o bin/tictactoe cmd/tictactoe/main.go
	@go build -o bin/rps_card cmd/rps_card/main.go
	@go build -o bin/train_models cmd/train_models/main.go
	@go build -o bin/compare_models cmd/compare_models/main.go
	@go build -o bin/play_vs_ai cmd/play_vs_ai/main.go
	@echo "AlphaGo binaries built successfully"

# Run AlphaGo tests
test:
	@echo "Running AlphaGo tests..."
	@go test -v ./...
	@echo "AlphaGo tests completed"

# Train AlphaGo models
train: build
	@echo "Training AlphaGo models..."
	@mkdir -p output
	@./bin/train_models
	@echo "Models saved to output/ directory"

# Train AlphaGo models using optimized parallel implementation
train-parallel: build
	@echo "Training AlphaGo models with parallel optimization..."
	@mkdir -p output
	@./bin/train_models --parallel
	@echo "Models saved to output/ directory"

# Train AlphaGo models with reduced parameters (for testing)
quick-train: build
	@echo "Quick training AlphaGo models (reduced parameters)..."
	@mkdir -p output
	@./bin/train_models --small-run
	@echo "Models saved to output/ directory"

# Train AlphaGo models with parallel optimization (in small-run mode)
quick-train-parallel: build
	@echo "Quick training AlphaGo models (reduced parameters, parallel)..."
	@mkdir -p output
	@./bin/train_models --small-run --parallel
	@echo "Models saved to output/ directory"

# Compare AlphaGo models
compare: build
	@echo "Comparing AlphaGo models..."
	@./bin/compare_models --games 50
	@echo "Comparison results saved to output/ directory"

# Run a tournament with ELO tracking
tournament-elo: build
	@echo "Running tournament with ELO tracking..."
	@./bin/compare_models --games 100 --tournament-mode --elo
	@echo "Tournament results and ELO ratings saved to output/elo_ratings/"

# Optimize thread count for your hardware
optimize-threads: build
	@echo "Finding optimal thread count for your hardware..."
	@./bin/train_models --optimize-threads
	@echo "Optimization complete, results in output/thread_optimization.txt"

# Run training with CPU profiling
profile-train: build
	@echo "Running training with CPU profiling..."
	@mkdir -p output/profiles
	@./bin/train_models --profile --small-run
	@echo "Profile saved to output/profiles/"

# Play against an AlphaGo agent
play: build
	@echo "Starting AlphaGo RPS game..."
	@./bin/rps_card
	@echo "Game ended"

# Play against an AlphaGo agent (interactive mode)
play-interactive: build
	@echo "Starting interactive game against AlphaGo agent..."
	@./bin/play_vs_ai
	@echo "Game ended"

# Clean build artifacts
clean:
	@echo "Cleaning AlphaGo build artifacts..."
	@rm -rf bin/
	@rm -f tictactoe rps_card
	@echo "AlphaGo build artifacts cleaned" 