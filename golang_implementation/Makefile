.PHONY: build run test test-coverage clean tournament tournament-verbose

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
	@mkdir -p bin
	@go build -o bin/tournament cmd/tournament/main.go
	@echo "Running tournament between AlphaGo and PPO agents..."
	@./bin/tournament --games 50 --alphago-sims 300 --ppo-hidden 128
	@echo "Tournament complete!"

# Tournament verbose
tournament-verbose:
	@mkdir -p bin
	@go build -o bin/tournament cmd/tournament/main.go
	@echo "Running tournament (verbose mode)..."
	@./bin/tournament --games 10 --alphago-sims 300 --ppo-hidden 128 --verbose
	@echo "Tournament complete!" 