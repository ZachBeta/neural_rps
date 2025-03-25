.PHONY: build run test test-coverage clean build-cpp build-go build-alphago run-demos

# Build all implementations
build: build-go build-cpp build-alphago

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
	@echo "Running C++ demo (this might fail if Eigen is not installed)..."
	-cd cpp_implementation/build && ./neural_rps
	@echo "---------------------------------"
	@echo "Running Go implementation demo..."
	cd golang_implementation && ./bin/neural_rps
	@echo "---------------------------------"
	@echo "Running AlphaGo demo..."
	cd alphago_demo && ./run.sh
	@echo "---------------------------------"
	@echo "All demos completed. Output files generated in project root:"
	@echo "- cpp_demo_output.txt (if C++ demo ran successfully)"
	@echo "- go_demo_output.txt"
	@echo "- alphago_demo_output.txt"

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
	rm -f alphago_demo/tictactoe
	rm -f cpp_demo_output.txt go_demo_output.txt alphago_demo_output.txt go_neural_rps_model.gob

# Run the Golang program
run-go: build-go
	@echo "Running Golang implementation..."
	./golang_implementation/bin/neural_rps

# Run the AlphaGo demo
run-alphago: build-alphago
	@echo "Running AlphaGo demo..."
	./alphago_demo/tictactoe

# Run the C++ implementation
run-cpp: build-cpp
	@echo "Running C++ implementation..."
	./cpp_implementation/build/neural_rps

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