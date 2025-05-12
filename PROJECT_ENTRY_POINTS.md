# Project Entry Points

This document outlines the various entry points for building, running, training, and benchmarking the components of the Neural Rock Paper Scissors project.

## Go `main.go` Executables

These are direct Go program entry points, typically invoked via `go run` or after building with `go build`.

### Core Commands
*   `cmd/arena/main.go`: Likely for running agent vs. agent matches.
*   `cmd/training/main.go`: General training command (potentially for the `golang_implementation`).
*   `cmd/tournament/main.go`: General tournament command (potentially for the `golang_implementation`).
*   `cmd/tools/main.go`: Miscellaneous tools.
*   `cmd/demo/main.go`: General demo runner.
*   `cmd/evaluate_checkpoints/main.go`: For evaluating saved model checkpoints.
*   `cmd/training_cpp/main.go`: Possibly a Go entry point related to the C++ training or data processing.

### AlphaGo Demo Specific Commands (primarily in `alphago_demo/cmd/`)
These commands are usually run from the `alphago_demo` directory (e.g., `cd alphago_demo; go run cmd/.../main.go`).
*   `alphago_demo/cmd/train_models/main.go`: **Primary training entry point for AlphaGo-style models with MCTS.** Supports self-play, parallel execution, and different training methods (AlphaGo MCTS, NEAT).
*   `alphago_demo/cmd/elo_tournament/main.go`: Comprehensive ELO-based tournament system for comparing all agent types.
*   `alphago_demo/cmd/tournament_with_minimax/main.go`: Runs tournaments comparing neural networks against minimax search agents.
*   `alphago_demo/cmd/train_top_agents/main.go`: For continuing the training of pre-trained models.
*   `alphago_demo/cmd/train_supervised/main.go`: For training models on existing expert gameplay data.
*   `alphago_demo/cmd/generate_training_data/main.go`: Generates expert gameplay data using minimax search.
*   `cmd/alphago_demo/main.go`: A more general entry point for the `alphago_demo`, though the README emphasizes the more specific `main.go` files within `alphago_demo/cmd/`.


## Shell Scripts (`.sh`)

Various shell scripts provide for building, running, and utility tasks. (Note: Over 20 `.sh` files were found; key ones are listed below).

### Root Directory Scripts
*   `run_benchmark.sh`: **Crucial for CPU vs. GPU performance benchmarking.** Accepts options like `--batch-size`, `--iterations`, `--cpu-only`, `--gpu-only`. (Note: Currently runs, but GPU performance is significantly slower than CPU and needs investigation/improvement as part of the refactoring.)
*   `start_neural_service.sh`: **Starts the Python gRPC service (using TensorFlow) for GPU acceleration.** (Note: Runs successfully, starts the gRPC service, and correctly identifies/utilizes the Apple Silicon Metal GPU. Provides startup/shutdown/status checks.)

### `scripts/` Directory
*   `scripts/build.sh`: Generic build script.
*   `scripts/build_all.sh`: Builds all relevant parts of the project.
*   `scripts/run_demos.sh`: Runs various demonstrations.
*   `scripts/generate_proto.sh`: Likely for generating code from Protocol Buffer definitions (gRPC).

### `alphago_demo/` Directory
*   `alphago_demo/run.sh`: General run script for the AlphaGo demo.
*   `alphago_demo/build.sh`: Build script specific to the AlphaGo demo.
*   `alphago_demo/run_rps.sh`: Runs the RPS card game demo.
*   `alphago_demo/run_tests.sh`: Runs tests specific to the AlphaGo demo.

### `python/` Directory
*   `python/setup_local_env.sh`: Sets up the Python virtual environment, including dependencies for Apple Silicon (Metal) GPU acceleration via TensorFlow.
*   `python/generate_grpc.sh`: Generates gRPC-related Python code.

## Makefiles

Makefiles offer convenient commands for building and running different parts of the project.

### Root Makefile (`Makefile`)
This is the primary Makefile providing high-level commands.
*   `make build`: Builds all implementations. (Note: Currently fails during the build of `legacy_cpp_implementation` due to C/C++ compiler/linker errors. This part is deprecated and planned for removal. The status of Go components build under this target is unconfirmed due to the C++ build halting the process.)
*   `make run-legacy-cpp`: Runs the legacy C++ implementation.
*   `make run-cpp`: Runs the C++ demonstration.
*   `make run-go`: Runs the Golang implementation. (Note: Fails to build. The `golang_implementation` it targets has direct Go dependencies on TensorFlow, which are missing `go.sum` entries. This entry point and potentially the entire `golang_implementation` are likely deprecated/superseded by `alphago_demo` and plans for PyTorch/ONNX, marking it for future removal.)
*   `make run-alphago-ttt`: Runs the AlphaGo Tic-Tac-Toe demo.
*   `make run-alphago-rps`: Runs the AlphaGo RPS card game demo.
*   `make alphago-train`: **Trains AlphaGo RPS models (likely invokes `alphago_demo/cmd/train_models/main.go`).** (Note: Runs successfully. This includes self-play, model training, model saving, and a subsequent tournament to compare different model configurations.)
*   `make alphago-tournament`: **Compares different AlphaGo models (likely invokes `alphago_demo/cmd/elo_tournament/main.go` or similar).** (Note: Runs successfully. It builds AlphaGo binaries, loads existing models from the `output/` directory, compares them, and runs a tournament, saving results.)
*   `make golang-tournament`: Runs tournaments between agents in the Golang implementation. (Note: Fails to build due to the same TensorFlow Go dependency issues as `make run-go`. This entry point is tied to the likely deprecated `golang_implementation` and is marked for future removal.)
*   `make golang-vs-alphago`: Compares Golang and AlphaGo agents. (Note: Fails to build, likely due to dependencies on the deprecated `golang_implementation` and its TensorFlow Go bindings. Marked for future removal.)
*   `make run-alphago-ttt`: Runs the AlphaGo Tic-Tac-Toe demo. (Note: Runs, but appears to get stuck in a gameplay loop. Likely stale and not critical for the RPS MCTS project. Marked for review/potential removal.)
*   `make run-alphago-rps`: Runs the AlphaGo RPS card game demo. (Note: Runs, allowing a human player against an ad-hoc trained model. Considered stale as the project focus has shifted to agent vs. agent tournaments. Marked for review/potential removal.)

### Other Makefiles
These provide more granular control within specific project parts.
*   `alphago_demo/Makefile`
*   `golang_implementation/Makefile`
*   `cpp_implementation/Makefile`

## Primary Entry Points for CPU/GPU Profiling Goals

Based on the project goals (CPU/GPU MCTS versions, removing TensorFlow, profiling benchmarks):

1.  **Benchmarking:**
    *   `./run_benchmark.sh` (Shell script)
    *   `./start_neural_service.sh` (Shell script, for the current GPU path via Python/TensorFlow)

2.  **Training Commands:**
    *   `make alphago-train` (Makefile target)
    *   `alphago_demo/cmd/train_models/main.go` (Go executable)
    *   Possibly `cmd/training/main.go` if it's adapted for the new MCTS.

3.  **Tournament Commands:**
    *   `make alphago-tournament` (Makefile target)
    *   `make golang-tournament` (Makefile target)
    *   `alphago_demo/cmd/elo_tournament/main.go` (Go executable)
    *   Possibly `cmd/tournament/main.go` if it's adapted.

Understanding the interaction between the Go code and the Python gRPC service (`start_neural_service.sh`) will be key for the TensorFlow removal and GPU refactoring. 