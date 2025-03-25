.PHONY: build run test test-coverage clean build-legacy-cpp build-cpp build-go build-alphago run-demos run-cpp-full

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
	@echo "All demos completed. Output files generated in project root:"
	@echo "- legacy_cpp_demo_output.txt (from legacy C++ implementation)"
	@echo "- cpp_demo_output.txt (from simplified C++ demo)"
	@echo "- go_demo_output.txt"
	@echo "- alphago_demo_output.txt"
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
	rm -f cpp_demo_output.txt legacy_cpp_demo_output.txt go_demo_output.txt alphago_demo_output.txt go_neural_rps_model.gob

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
deps:
	@echo "Installing dependencies..."
	cd golang_implementation && go mod tidy
	cd alphago_demo && go mod tidy
	go install github.com/golangci/golint/cmd/golangci-lint@latest

# Format code
fmt:
	@echo "Formatting Golang code..."
	cd golang_implementation && go fmt ./...
	cd alphago_demo && go fmt ./...

# Generate documentation
doc:
	@echo "Generating documentation..."
	cd golang_implementation && godoc -http=:6060 