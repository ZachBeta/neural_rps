.PHONY: all build run test clean \
        build-legacy-cpp build-cpp build-go build-alphago \
        run-legacy-cpp run-cpp run-go run-alphago-ttt run-alphago-rps run-alphago-interactive \
        test-go test-alphago \
        alphago-train alphago-quick-train alphago-tournament golang-tournament golang-vs-alphago \
        doc fmt install-deps

# Main targets
all: build

# Build all implementations
build: build-legacy-cpp build-cpp build-go build-alphago
	@echo "All implementations built successfully"

# Build individual implementations
build-legacy-cpp:
	@echo "Building Legacy C++ implementation..."
	cd legacy_cpp_implementation && mkdir -p build && cd build && cmake .. && make

build-cpp:
	@echo "Building C++ implementation..."
	cd cpp_implementation && make all

build-go:
	@echo "Building Golang implementation..."
	cd golang_implementation && make build

build-alphago:
	@echo "Building AlphaGo demos..."
	cd alphago_demo && make build

# Run individual implementations
run-legacy-cpp: build-legacy-cpp
	@echo "Running Legacy C++ implementation..."
	./legacy_cpp_implementation/build/src/legacy_neural_rps

run-cpp: build-cpp
	@echo "Running C++ simplified demo..."
	cd cpp_implementation && make run-demo

run-go: build-go
	@echo "Running Golang implementation..."
	cd golang_implementation && make run

run-alphago-ttt: build-alphago
	@echo "Running AlphaGo Tic-Tac-Toe demo..."
	cd alphago_demo && ./bin/tictactoe

run-alphago-rps: build-alphago
	@echo "Running AlphaGo RPS Card Game demo..."
	cd alphago_demo && make play

run-alphago-interactive: build-alphago
	@echo "Running interactive game against AlphaGo agent..."
	cd alphago_demo && make play-interactive

# Testing
test: test-go test-alphago
	@echo "All tests passed"

test-go:
	@echo "Running Golang tests..."
	cd golang_implementation && make test

test-alphago:
	@echo "Running AlphaGo tests..."
	cd alphago_demo && make test

# AlphaGo specific targets
alphago-train: build-alphago
	@echo "Training AlphaGo models..."
	cd alphago_demo && make train
	@echo "Training complete, models saved to alphago_demo/output/"

alphago-quick-train: build-alphago
	@echo "Quick training AlphaGo models (reduced parameters)..."
	cd alphago_demo && make quick-train
	@echo "Training complete, models saved to alphago_demo/output/"

alphago-tournament: build-alphago
	@echo "Running AlphaGo tournament..."
	cd alphago_demo && make compare
	@echo "Tournament complete, results in alphago_demo/output/"

# Golang implementation specific targets
golang-tournament: build-go
	@echo "Running Golang implementation tournament..."
	cd golang_implementation && make tournament-verbose
	@echo "Tournament complete, results in golang_implementation/results/"

golang-vs-alphago: build-go
	@echo "Running tournament between Golang and AlphaGo agents..."
	cd golang_implementation && make alphago-vs-alphago-verbose
	@echo "Tournament complete, results in golang_implementation/results/"

# Clean build artifacts
clean:
	@echo "Cleaning all build artifacts..."
	cd golang_implementation && make clean
	cd alphago_demo && make clean
	cd cpp_implementation && make clean
	rm -rf legacy_cpp_implementation/build/

# Install dependencies
install-deps:
	@echo "Installing dependencies..."
	cd golang_implementation && go mod tidy
	cd alphago_demo && go mod tidy
	@echo "Dependencies installed!"

# Format code
fmt:
	@echo "Formatting Golang code..."
	cd golang_implementation && make fmt
	cd alphago_demo && go fmt ./...

# Generate documentation
doc:
	@echo "Generating documentation..."
	mkdir -p docs
	@echo "For golang_implementation: http://localhost:6060/pkg/github.com/zachbeta/neural_rps/golang_implementation/"
	@echo "For alphago_demo: http://localhost:6061/pkg/github.com/zachbeta/neural_rps/alphago_demo/"
	cd golang_implementation && godoc -http=:6060 &
	cd alphago_demo && godoc -http=:6061 