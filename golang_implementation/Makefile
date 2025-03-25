.PHONY: build run test test-coverage clean

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