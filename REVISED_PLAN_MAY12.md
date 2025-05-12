# Revised Project Plan (May 12, 2024)

## Background

The initial goal was to refactor the Go MCTS codebase, remove TensorFlow dependencies, and benchmark CPU vs. GPU performance using ONNX. We explored converting Go models to ONNX, benchmarking ONNX inference speed in Go (CPU) and via a Python gRPC service (intended for GPU, but fell back to CPU). We then shifted focus to comparing end-to-end training pipelines (pure Go vs. Go example generation + Python training) and started integrating NEAT benchmarking.

## Key Achievements & Observations

1.  **ONNX Conversion:** Successfully converted native Go JSON models and PyTorch `.pt` models to ONNX format.
2.  **Go ONNX Inference (CPU):** Integrated `onnxruntime_go` into the `cmd/benchmark` tool. Showed significant inference speedup (3-27x faster) compared to the ad-hoc Go implementation for the models tested.
3.  **Python ONNX Service:** Created a Python gRPC service using ONNX Runtime. GPU acceleration via CoreML was problematic and fell back to CPU. gRPC overhead was substantial compared to direct Go ONNX inference.
4.  **Training Pipeline Experiment:** Demonstrated a ~1.3x speedup for end-to-end training time (H64 model) by separating Go self-play example generation (`alphago_demo/cmd/generate_examples`) from Python training (`python/train_from_go_examples.py`), compared to the original monolithic Go training loop (`alphago_demo/cmd/train_models`).
5.  **NEAT Training:** Successfully trained a NEAT model.
6.  **MCTS Optimization:** Identified MCTS node update contention (`sync.RWMutex`) as a bottleneck in Go training and made an initial optimization using `atomic.Int64`.

## Current Roadblock: Go Module Resolution

We are currently blocked by a persistent Go module resolution error:

```
go: finding module for package github.com/zachbeta/neural_rps/pkg/neural
go: github.com/zachbeta/neural_rps/cmd/benchmark imports
        github.com/zachbeta/neural_rps/pkg/neural: no matching versions for query "latest"
```

This occurs when running `go mod tidy` from the workspace root, despite having a single root `go.mod` defining `module github.com/zachbeta/neural_rps`, deleting the nested `alphago_demo/go.mod`, and confirming that import paths within the codebase *appear* correct (using the full `github.com/zachbeta/neural_rps/...` path). Attempts to use `replace` directives or clean the module cache have not resolved the issue. This prevents building/running tools like `cmd/benchmark`.

## Revised Plan

Our priority is to fix the module resolution issue and then leverage the fast Go ONNX inference capabilities within the training loop itself.

**Phase 1: Fix Go Module Resolution**

1.  **Investigate Root Cause:** Systematically review import dependencies across *all* key packages (`cmd/benchmark`, `pkg/neural/*`, `alphago_demo/cmd/*`, `alphago_demo/pkg/*`). Look for potential circular dependencies, conflicts between `pkg/neural` and `alphago_demo/pkg/neural`, or other structural issues that might confuse the Go module tooling.
2.  **Attempt Explicit Local Get:** Remove any `replace` directive from `go.mod`. Run `go get ./...` from the root directory, followed by `go mod tidy -v`. This explicitly tells Go to find local packages first.
3.  **Consider Restructuring:** If the issue persists, consider a minor code structure change. For example, consolidate all core, reusable logic (including `alphago_demo/pkg/*`) under the root `pkg/` directory, leaving `alphago_demo/` primarily for `cmd/` binaries and potentially game-specific logic (if any isn't already in `pkg/game`).
4.  **Explore Go Workspaces:** If restructuring is undesirable or doesn't fix the issue, investigate using Go Workspaces (`go work init`, `go work use ./ ./alphago_demo`). This is the modern approach for managing multiple modules within a single repository and might resolve the pathing conflicts more cleanly than `replace`.

**Phase 2: Complete NEAT Benchmarking**

1.  **Integrate NEAT:** Once the module issues are resolved (allowing `cmd/benchmark` to build), complete the integration of NEAT model loading and benchmarking (`runCPUNEATBenchmark`) in `cmd/benchmark/main.go`.
2.  **Run Benchmarks:** Compare NEAT CPU inference performance against the MCTS baselines (ad-hoc Go, ONNX Go).

**Phase 3: Integrate ONNX into Go MCTS Training/Self-Play**

1.  **Goal:** Utilize the fast `onnxruntime_go` inference *during* MCTS self-play to potentially speed up the Go training process (`alphago_demo/cmd/train_models`) and example generation (`alphago_demo/cmd/generate_examples`).
2.  **Define Interface:** Create a Go interface (e.g., `ModelEvaluator` in `pkg/neural`) with `Predict(input []float32) ([]float32, error)` and `BatchPredict(batch [][]float32) ([][]float32, error)` methods.
3.  **Implement Interface:**
    *   Create an implementation for the existing ad-hoc Go networks (e.g., `cpu.RPSCPUPolicyNetwork` wrapper).
    *   Create a new implementation using `onnxruntime_go` (e.g., in a new `pkg/neural/onnx` package) that loads an ONNX model and uses the session to fulfill the interface.
4.  **Refactor MCTS:** Modify `alphago_demo/pkg/mcts/rps_search.go` (specifically `searchParallel` and potentially node expansion logic) to accept and use the `ModelEvaluator` interface instead of concrete `neural.RPS...Network` types.
5.  **Update Training Cmds:** Add flags to `alphago_demo/cmd/train_models/main.go` and `alphago_demo/cmd/generate_examples/main.go` to:
    *   Specify the evaluator type (`--evaluator-type go` or `--evaluator-type onnx`).
    *   Provide the ONNX model path if `--evaluator-type onnx` is used.
6.  **Profile:** Benchmark the self-play/generation speed using the ONNX evaluator versus the native Go evaluator within the actual training/generation loops.

**Phase 4: Refine Training Pipeline Comparison**

1.  **Re-evaluate:** Based on the results of Phase 3 (whether ONNX significantly speeds up self-play), re-run the comparison between:
    *   Optimized end-to-end Go training (using the fastest evaluator identified in Phase 3).
    *   Go Example Generation (using fastest evaluator) + Python Training.
2.  **Model Quality:** Compare not just wall-clock time but also the *quality* of the models produced by different pipelines. Use the tournament (`alphago_demo/cmd/tournament_with_minimax`) against a fixed opponent (like Minimax or a baseline MCTS model) as a metric.

**Phase 5: Revisit GPU Acceleration (Lower Priority)**

1.  **Decision Point:** If ONNX inference in Go (Phase 3) provides sufficient performance gains for training, GPU acceleration might become less critical.
2.  **If Needed:** Explore options:
    *   Re-investigate `onnxruntime_go` providers (CoreML, DirectML, CUDA) - check library updates, examples.
    *   Explore direct Go bindings for GPU libraries (cuDNN, Metal Performance Shaders) - likely complex.
    *   Optimize the Python gRPC service (faster serialization, check ONNX Runtime Python optimizations) if it must be used.

**Phase 6: Documentation & Cleanup**

1.  Update `README.md` files to reflect the final architecture and usage.
2.  Update `PROJECT_ENTRY_POINTS.md`.
3.  Remove any obsolete scripts or code artifacts.
4.  Ensure dependencies (`go.mod`, `python/requirements.txt`) are clean.

This revised plan prioritizes fixing the build-blocking issue and then leveraging our findings about ONNX performance directly within the Go ecosystem. 