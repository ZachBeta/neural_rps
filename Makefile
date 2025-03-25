.PHONY: all build run test clean \
        build-legacy-cpp build-cpp build-go build-alphago \
        run-legacy-cpp run-cpp run-go run-alphago \
        alphago-train alphago-tournament golang-tournament \
        doc fmt install-deps

# Main targets
all: build

# Build all implementations
build: build-legacy-cpp build-cpp build-go build-alphago

# Build individual implementations
build-legacy-cpp:
	@echo "Building Legacy C++ implementation..."
	cd legacy_cpp_implementation && mkdir -p build && cd build && cmake .. && make

build-cpp:
	@echo "Building C++ implementation..."
	cd cpp_implementation && mkdir -p build && cd build && cmake .. && make

build-go:
	@echo "Building Golang implementation..."
	cd golang_implementation && go build -o bin/neural_rps cmd/neural_rps/main.go

build-alphago:
	@echo "Building AlphaGo demos..."
	cd alphago_demo && go build -o bin/tictactoe cmd/tictactoe/main.go
	cd alphago_demo && go build -o bin/rps_card cmd/rps_card/main.go

# Run individual implementations
run-legacy-cpp: build-legacy-cpp
	@echo "Running Legacy C++ implementation..."
	./legacy_cpp_implementation/build/src/legacy_neural_rps

run-cpp: build-cpp
	@echo "Running C++ simplified demo..."
	./cpp_implementation/build/neural_rps_demo

run-go: build-go
	@echo "Running Golang implementation..."
	./golang_implementation/bin/neural_rps

run-alphago-ttt: build-alphago
	@echo "Running AlphaGo Tic-Tac-Toe demo..."
	./alphago_demo/bin/tictactoe

run-alphago-rps: build-alphago
	@echo "Running AlphaGo RPS Card Game demo..."
	./alphago_demo/bin/rps_card

# Testing
test:
	@echo "Running all tests..."
	cd golang_implementation && go test -v ./...
	cd alphago_demo && go test -v ./...

# AlphaGo specific targets
alphago-train:
	@echo "Training AlphaGo models..."
	cd alphago_demo && go build -o bin/train_models ./cmd/train_models
	cd alphago_demo && ./bin/train_models
	@echo "Training complete, models saved to alphago_demo/output/"

alphago-tournament:
	@echo "Running AlphaGo tournament..."
	cd alphago_demo && go build -o bin/compare_models ./cmd/compare_models
	cd alphago_demo && ./bin/compare_models --games 50
	@echo "Tournament complete, results in alphago_demo/output/"

# Golang implementation specific targets
golang-tournament:
	@echo "Running Golang implementation tournament..."
	cd golang_implementation && make tournament-verbose
	@echo "Tournament complete, results in golang_implementation/results/"

golang-vs-alphago:
	@echo "Running tournament between Golang PPO and AlphaGo agents..."
	cd golang_implementation && make alphago-vs-alphago-verbose
	@echo "Tournament complete, results in golang_implementation/results/"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf golang_implementation/bin/
	rm -rf golang_implementation/coverage.out golang_implementation/coverage.html
	rm -rf cpp_implementation/build/
	rm -rf legacy_cpp_implementation/build/
	rm -rf alphago_demo/bin/
	rm -f alphago_demo/tictactoe alphago_demo/rps_card

# Install dependencies
install-deps:
	@echo "Installing dependencies..."
	cd golang_implementation && go mod tidy
	cd alphago_demo && go mod tidy
	@echo "Dependencies installed!"

# Format code
fmt:
	@echo "Formatting Golang code..."
	cd golang_implementation && go fmt ./...
	cd alphago_demo && go fmt ./...

# Generate documentation
doc:
	@echo "Generating documentation..."
	@echo "For golang_implementation: http://localhost:6060/pkg/github.com/zachbeta/neural_rps/golang_implementation/"
	@echo "For alphago_demo: http://localhost:6061/pkg/github.com/zachbeta/neural_rps/alphago_demo/"
	cd golang_implementation && godoc -http=:6060 &
	cd alphago_demo && godoc -http=:6061 