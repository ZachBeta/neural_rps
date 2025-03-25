.PHONY: build run test test-coverage clean tournament tournament-verbose alphago-vs-alphago alphago-vs-alphago-verbose

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