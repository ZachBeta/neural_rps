.PHONY: all neural_rps_demo neural_rps_full clean

# Build all C++ binaries
all: neural_rps_demo neural_rps_full
	@echo "All C++ binaries built successfully"

# Ensure build directory exists
build-dir:
	@mkdir -p build

# Build the simplified demo
neural_rps_demo: build-dir
	@echo "Building simplified neural RPS demo..."
	@cd build && cmake .. -DBUILD_DEMO=ON
	@cd build && make neural_rps_demo
	@echo "Simplified neural RPS demo built successfully"

# Build the full implementation
neural_rps_full: build-dir
	@echo "Building full neural RPS implementation..."
	@cd build && cmake .. -DBUILD_FULL=ON
	@cd build && make neural_rps_full
	@echo "Full neural RPS implementation built successfully"

# Run the simplified demo
run-demo: neural_rps_demo
	@echo "Running simplified neural RPS demo..."
	@./build/neural_rps_demo
	@echo "Demo completed"

# Run the full implementation
run-full: neural_rps_full
	@echo "Running full neural RPS implementation..."
	@./build/neural_rps_full
	@echo "Full implementation completed"

# Clean build artifacts
clean:
	@echo "Cleaning C++ build artifacts..."
	@rm -rf build/
	@echo "C++ build artifacts cleaned" 