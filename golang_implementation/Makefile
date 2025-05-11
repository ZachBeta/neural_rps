.PHONY: all build run test test-coverage clean tournament tournament-verbose alphago-vs-alphago alphago-vs-alphago-verbose benchmark benchmark-batch-test install-tf-mac install-tf-linux help update-tf-deps

# Default target
all: build

# Build the project
build:
	go build -o bin/neural_rps cmd/neural_rps/main.go

# Run tests
test:
	go test -v ./...

# Run test coverage
test-coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f coverage.out coverage.html

# Run the program
run: build
	./bin/neural_rps

# Install dependencies
deps:
	go mod tidy
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Format code
fmt:
	go fmt ./...

# Generate documentation
doc:
	godoc -http=:6060

# Tournament target
tournament:
	@mkdir -p bin results
	@go build -o bin/tournament cmd/tournament/main.go
	@echo "Running tournament between AlphaGo and PPO agents..."
	@./bin/tournament --games 50 --alphago-sims 300 --ppo-hidden 128 > results/tournament_results.txt
	@echo "Tournament complete! Results saved to results/tournament_results.txt"

# Tournament verbose
tournament-verbose:
	@mkdir -p bin results
	@go build -o bin/tournament cmd/tournament/main.go
	@echo "Running tournament (verbose mode)..."
	@./bin/tournament --games 10 --alphago-sims 300 --ppo-hidden 128 --verbose > results/tournament_verbose_results.txt
	@echo "Tournament complete! Results saved to results/tournament_verbose_results.txt"

# AlphaGo vs AlphaGo tournament
alphago-vs-alphago:
	@mkdir -p bin results
	@go build -o bin/alphago_vs_alphago cmd/alphago_vs_alphago/main.go
	@echo "Running tournament between small and large AlphaGo agents..."
	@./bin/alphago_vs_alphago --games 50 --small-sims 100 --large-sims 300 --small-exploration 1.0 --large-exploration 1.5 > results/alphago_vs_alphago_results.txt
	@echo "Tournament complete! Results saved to results/alphago_vs_alphago_results.txt"

# AlphaGo vs AlphaGo tournament (verbose)
alphago-vs-alphago-verbose:
	@mkdir -p bin results
	@go build -o bin/alphago_vs_alphago cmd/alphago_vs_alphago/main.go
	@echo "Running tournament between small and large AlphaGo agents (verbose mode)..."
	@./bin/alphago_vs_alphago --games 10 --small-sims 100 --large-sims 300 --small-exploration 1.0 --large-exploration 1.5 --verbose > results/alphago_vs_alphago_verbose_results.txt
	@echo "Tournament complete! Results saved to results/alphago_vs_alphago_verbose_results.txt"

# Run the benchmark
benchmark:
	go run cmd/benchmark/main.go --batch=128 --sims=800

# Run with different batch sizes
benchmark-batch-test:
	@echo "Testing with batch size 16"
	go run cmd/benchmark/main.go --batch=16 --sims=800
	@echo "Testing with batch size 64"
	go run cmd/benchmark/main.go --batch=64 --sims=800
	@echo "Testing with batch size 128"
	go run cmd/benchmark/main.go --batch=128 --sims=800
	@echo "Testing with batch size 256"
	go run cmd/benchmark/main.go --batch=256 --sims=800

# Install TensorFlow dependencies for macOS with Apple Silicon
install-tf-mac:
	@echo "Installing TensorFlow dependencies for macOS with Apple Silicon"
	export TF_NEED_OPENCL_SYCL=0
	export TF_NEED_ROCM=0
	export TF_NEED_CUDA=0
	export TF_NEED_MPS=1
	brew install tensorflow

# Install TensorFlow dependencies for Linux with NVIDIA GPUs
install-tf-linux:
	@echo "Installing TensorFlow dependencies for Linux with NVIDIA GPUs"
	export TF_NEED_OPENCL_SYCL=0
	export TF_NEED_ROCM=0
	export TF_NEED_CUDA=1
	export TF_NEED_MPS=0
	# For Ubuntu: apt-get install libtensorflow libtensorflow-dev

# Update TensorFlow dependencies
update-tf-deps:
	@echo "Updating TensorFlow dependencies..."
	go get github.com/tensorflow/tensorflow@latest
	go mod tidy

# Run the basic CPU benchmark
benchmark-cpu:
	go run cmd/basic_benchmark/main.go

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build the project"
	@echo "  build         - Build the project"
	@echo "  clean         - Clean build artifacts"
	@echo "  benchmark     - Run the benchmark with default settings"
	@echo "  benchmark-batch-test - Test with different batch sizes"
	@echo "  install-tf-mac   - Install TensorFlow dependencies for macOS with Apple Silicon"
	@echo "  install-tf-linux - Install TensorFlow dependencies for Linux with NVIDIA GPUs"
	@echo "  update-tf-deps   - Update TensorFlow dependencies"
	@echo "  help          - Show this help message" 