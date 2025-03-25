.PHONY: all build run test test-coverage clean build-legacy-cpp build-cpp build-go build-alphago run-demos run-cpp-full install-go-deps install-cpp-deps game-server run-tournament train-and-compete train-large-comparison play-vs-ai compare-models

# Build all implementations
build: build-go build-cpp build-alphago build-legacy-cpp

# Build the legacy C++ implementation
build-legacy-cpp:
	@echo "Building Legacy C++ implementation..."
	cd legacy_cpp_implementation && mkdir -p build && cd build && cmake .. && make

# Build the C++ implementation
build-cpp:
	@echo "Building C++ implementation..."
	cd cpp_implementation && mkdir -p build && cd build && cmake .. && make

# Build the Golang implementation
build-go:
	@echo "Building Golang implementation..."
	cd golang_implementation && go build -o bin/neural_rps cmd/neural_rps/main.go

# Build the AlphaGo demo
build-alphago:
	@echo "Building AlphaGo demo..."
	cd alphago_demo && go build -o tictactoe cmd/tictactoe/main.go

# Run demos for all implementations and generate output files
run-demos: build
	@echo "Running all implementation demos..."
	@echo "---------------------------------"
	@echo "Running Legacy C++ implementation demo (this will take a while)..."
	-cd legacy_cpp_implementation/build && ./src/legacy_neural_rps
	@echo "---------------------------------"
	@echo "Running C++ simplified demo..."
	-cd cpp_implementation/build && ./neural_rps_demo
	@echo "---------------------------------"
	@echo "Running Go implementation demo..."
	cd golang_implementation && ./bin/neural_rps
	@echo "---------------------------------"
	@echo "Running AlphaGo demo..."
	cd alphago_demo && ./run.sh
	@echo "---------------------------------"
	@echo "All demos completed. Output files generated in output directory:"
	@echo "- output/legacy_cpp_demo_output.txt (from legacy C++ implementation)"
	@echo "- output/cpp_demo_output.txt (from simplified C++ demo)"
	@echo "- output/go_demo_output.txt"
	@echo "- output/alphago_demo_output.txt"
	@echo ""
	@echo "Note: The full C++ neural implementation can be run with:"
	@echo "make run-cpp-full"

# Run tests for all implementations
test: test-go

# Run Golang tests
test-go:
	@echo "Running Golang tests..."
	cd golang_implementation && go test -v ./...

# Run test coverage for Golang implementation
test-coverage:
	@echo "Running Golang test coverage..."
	cd golang_implementation && go test -coverprofile=coverage.out ./...
	cd golang_implementation && go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf golang_implementation/bin/
	rm -rf golang_implementation/coverage.out golang_implementation/coverage.html
	rm -rf cpp_implementation/build/
	rm -rf legacy_cpp_implementation/build/
	rm -f alphago_demo/tictactoe
	rm -f output/cpp_demo_output.txt output/legacy_cpp_demo_output.txt output/go_demo_output.txt output/alphago_demo_output.txt output/go_neural_rps_model.gob

# Run the Golang program
run-go: build-go
	@echo "Running Golang implementation..."
	./golang_implementation/bin/neural_rps

# Run the AlphaGo demo
run-alphago: build-alphago
	@echo "Running AlphaGo demo..."
	./alphago_demo/tictactoe

# Run the legacy C++ implementation
run-legacy-cpp: build-legacy-cpp
	@echo "Running Legacy C++ implementation..."
	./legacy_cpp_implementation/build/src/legacy_neural_rps

# Run the C++ implementation (simplified demo version)
run-cpp: build-cpp
	@echo "Running C++ simplified demo..."
	./cpp_implementation/build/neural_rps_demo

# Run the full C++ implementation
run-cpp-full: build-cpp
	@echo "Running C++ full neural implementation (will run for a while)..."
	./cpp_implementation/build/neural_rps_full

# Install dependencies
install-go-deps:
	@echo "Installing Go dependencies..."
	cd golang_implementation && go mod tidy
	cd alphago_demo && go mod tidy
	go install github.com/golangci/golint/cmd/golangci-lint@latest
	@echo "Go dependencies installed!"

install-cpp-deps:
	@echo "Checking for CMake..."
	@which cmake > /dev/null || (echo "CMake not found. Please install CMake." && exit 1)
	@echo "Dependencies checked!"

# Game server and tournament targets
build-game-server:
	@echo "Building RPS game server..."
	@mkdir -p scripts/bin
	@cd scripts && go build -o bin/game_server game_server.go agent_adapters.go
	@echo "Game server build complete!"

run-game-server: build-game-server
	@echo "Starting RPS game server..."
	@cd scripts && ./bin/game_server

# Build the external agent adapters for Go and C++ implementations
build-agent-adapters:
	@echo "Building external agent adapters..."
	@cd golang_implementation && go build -o bin/rps_agent ./cmd/rps_agent
	@mkdir -p cpp_implementation/build
	@cd cpp_implementation/build && cmake .. -DBUILD_AGENT=ON
	@cd cpp_implementation/build && make rps_agent
	@echo "Agent adapters build complete!"

# Run a tournament between all available agents
run-tournament: build-game-server build-agent-adapters
	@echo "Starting RPS tournament..."
	@cd scripts && ./bin/game_server --tournament
	@echo "Tournament complete!"

# Train and compare AlphaGo models
train-and-compete:
	@echo "Building training script..."
	@cd alphago_demo && go build -o bin/train_models ./cmd/train_models
	@echo "Starting AlphaGo model training and competition..."
	@cd alphago_demo && ./bin/train_models
	@echo "Training and competition complete!"
	@echo "Model files saved to alphago_demo/output/"

# Train with larger datasets (100 vs 1000 games)
train-large-comparison:
	@echo "Building training script for large comparison..."
	@cd alphago_demo && go build -o bin/train_models ./cmd/train_models
	@echo "Starting AlphaGo model training with 100 vs 1000 games..."
	@cd alphago_demo && ./bin/train_models
	@echo "Training and competition complete!"
	@echo "Model files saved to alphago_demo/output/"
	@echo "Note: This will take significantly longer to run than the standard training."

# Compare trained models in a tournament
compare-models:
	@echo "Building model comparison tool..."
	@cd alphago_demo && go build -o bin/compare_models ./cmd/compare_models
	@echo "Starting model comparison tournament..."
	@cd alphago_demo && ./bin/compare_models --games 100 --model1-name "100Games" --model2-name "1000Games"
	@echo "Model comparison complete!"
	@echo "Results saved to alphago_demo/results/ directory"

# Experimental: Play against a trained AI model (not a priority for current development)
play-vs-ai:
	@echo "Note: Human vs AI mode is experimental and not a current priority"
	@cd alphago_demo && go build -o bin/play_vs_ai ./cmd/play_vs_ai
	@echo "Starting human vs AI game..."
	@cd alphago_demo && ./bin/play_vs_ai
	@echo "Game completed!"

# Game server dependencies
install-game-server-deps:
	@echo "Installing game server dependencies..."
	@cd scripts && go get github.com/gorilla/mux
	@echo "Game server dependencies installed!"

# Install all dependencies
install-deps: install-go-deps install-cpp-deps install-game-server-deps
	@echo "All dependencies installed!"

# Format code
fmt:
	@echo "Formatting Golang code..."
	cd golang_implementation && go fmt ./...
	cd alphago_demo && go fmt ./...

# Generate documentation
doc:
	@echo "Generating documentation..."
	cd golang_implementation && godoc -http=:6060 