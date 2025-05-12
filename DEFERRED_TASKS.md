# Deferred Tasks and Future Improvements

This document tracks tasks and potential improvements that have been deferred during the MCTS refactoring project. These are items to revisit after the primary goals of establishing ONNX-based CPU and GPU benchmarking paths are complete.

## Performance Optimization

*   **Advanced GPU Performance Optimization:**
    *   Investigate and implement techniques beyond the initial ONNX/CoreML setup for maximizing GPU throughput.
    *   Explore advanced batching strategies specific to GPU execution.
    *   Consider parallelism options within the GPU service or model execution if beneficial.

## Codebase Cleanup and Maintenance

*   **Remove Old TensorFlow Go Dependencies:**
    *   Identify and remove all Go packages and code related to the previous TensorFlow Go bindings.
    *   Update `go.mod` and `go.sum` accordingly.
*   **Remove Deprecated C++ Parts:**
    *   Remove the C++ components and related `Makefile` targets (e.g., affecting `make build`).
*   **Address Stale `make` Targets and Scripts:**
    *   Review and remove or update `make` targets like `run-alphago-ttt` and `run-alphago-rps` that were marked as stale or problematic.
    *   Clean up any other unused or non-functional shell scripts.

## Development Environment and Workflow

*   **Streamline Python Environment Management:**
    *   Improve the process for setting up and activating Python virtual environments (e.g., `uv`).
    *   Consider adding `Makefile` targets or more explicit script steps to simplify environment setup for developers.

## Minor Pending Items (from previous phases)

*   **Investigate Original `./run_benchmark.sh` GPU Slowdown:**
    *   Although superseded by new benchmark structure, a quick look might offer insights if time permits (low priority).
    *   This was when the benchmark used ad-hoc Go models, not trained neural networks.

## General Code Quality

*   **Comprehensive Review of Comments and Documentation:**
    *   Once major refactoring is stable, review all code comments, READMEs, and tutorial Markdown files for accuracy, clarity, and completeness.

*(This list can be updated as new items are identified or priorities change.)* 