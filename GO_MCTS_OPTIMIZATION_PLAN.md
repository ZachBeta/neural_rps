# Go MCTS Self-Play and Training Optimization Plan

This document outlines the current plan to optimize the Go-based Monte Carlo Tree Search (MCTS) self-play and training loop, primarily focusing on reducing synchronization overhead identified through CPU profiling.

## 1. Initial Profiling and Bottleneck Identification (Done)

*   Successfully ran `go run alphago_demo/cmd/train_models/main.go --m1-hidden=64 --m1-games=100 --m1-epochs=1 --profile`.
*   Analyzed the CPU profile (`output/profiles/cpu_*.prof`) using `go tool pprof -top`.
*   **Key finding:** Significant time spent in runtime synchronization primitives (`runtime.usleep`, `runtime.lock2`, `runtime.pthread_cond_wait`), indicating lock contention, likely within the parallel MCTS (`searchParallel`) and specifically the `backpropagateThreadSafe` method due to its recursive nature under a write lock.

## 2. Optimize MCTS Node Updates (In Progress)

### Step 2.1: Atomic `Visits` Count (Done)

*   **Action:** Modified `alphago_demo/pkg/mcts/rps_node.go`.
    *   Changed `RPSMCTSNode.Visits` from `int` to `atomic.Int64`.
    *   Updated `NewRPSMCTSNode` for initialization.
    *   Modified `RPSMCTSNode.Update` to use `n.Visits.Add(1)`.
    *   Updated all read accesses to `Visits` (e.g., in `UCB`, `MostVisitedChild`, `BestChild`) to use `n.Visits.Load()`.
*   **Goal:** Reduce lock contention during the `Update` phase of backpropagation by making visit count updates atomic.

### Step 2.2: Re-run Profiling to Assess Impact of Atomic `Visits`

*   **Action:** Execute the training with profiling again:
    ```bash
    cd alphago_demo/cmd/train_models && go run . --m1-hidden=64 --m1-games=100 --m1-epochs=1 --profile
    ```
*   **Action:** Analyze the newly generated CPU profile (e.g., `cpu_*.prof` in `output/profiles/` at the workspace root) using:
    ```bash
    go tool pprof -top output/profiles/<new_profile_file.prof>
    ```
*   **Goal:** Observe changes in the `top` functions. Specifically, look for reductions in time spent in `runtime.lock2`, `runtime.usleep`, and other synchronization overhead. Determine if the contention around `backpropagateThreadSafe` has decreased.

### Step 2.3: Further Optimize `TotalValue` Updates (If Necessary)

*   **Condition:** If profiling from Step 2.2 still shows significant lock contention related to `TotalValue` updates within `backpropagateThreadSafe` (since `TotalValue` is still updated under the `treeMutex.Lock()`).
*   **Potential Action:** Modify `RPSMCTSNode.TotalValue` (currently `float64`) to use atomic operations.
    *   Change type to `atomic.Uint64`.
    *   In `Update` method, use `math.Float64bits` to convert the `float64` value to `uint64`, and then use a compare-and-swap (CAS) loop with `atomic.CompareAndSwapUint64` to update `n.TotalValue`.
    *   Reads of `TotalValue` would then involve `n.TotalValue.Load()` and `math.Float64frombits()`.
*   **Goal:** Make `TotalValue` updates atomic, potentially allowing the `treeMutex.Lock()` in `backpropagateThreadSafe` to be removed or its scope dramatically reduced for the update operations.

## 3. Iterative Refinement

*   Based on the results of each profiling run, identify the next most significant bottleneck.
*   Propose and implement targeted optimizations (e.g., further reducing lock scope, exploring alternative synchronization primitives, optimizing data structures, reducing memory allocations if GC pressure is high).
*   Repeat the profiling and analysis cycle.

## 4. Long-Term Context

*   The overarching goal is to significantly improve the wall-clock time required for training more complex neural network models using the Go-based self-play and MCTS engine.
*   Achieving better parallelism and reducing synchronization overhead are key to scaling the training process effectively. 