# Tutorial: Phased Realistic Benchmark (ONNX CPU vs. TensorFlow GPU Service)

**Goal:** To refactor the existing benchmark (`cmd/benchmark/main.go`) to use actual trained models (from Go training) for CPU benchmarking via ONNX Runtime in Go. The GPU benchmark will continue to use the existing Python gRPC service (which internally uses TensorFlow). This provides a phased approach to introducing ONNX and improving benchmark realism.

**Target Audience:** Mid-level Software Engineer.

**Overall Flow:**
1.  Go training program saves models as JSON (existing functionality).
2.  A new Python script converts these JSON models to `.onnx` format (using PyTorch as an intermediary for robust ONNX export).
3.  The Go benchmark program (`cmd/benchmark/main.go`) is modified:
    *   For **CPU tests**: Loads the `.onnx` model and uses ONNX Runtime in Go.
    *   For **GPU tests**: Continues to use the existing Python gRPC service (`python/neural_service.py`) which runs TensorFlow models.

**Assumptions:**
*   You have a working checkout of the `neural_rps` project.
*   The Go training (`make alphago-train`) successfully produces `.model` (JSON) files in `alphago_demo/output/`.
*   The Python gRPC service (`./start_neural_service.sh` launching `python/neural_service.py`) is functional.
*   You are familiar with basic Go and Python development.
*   Your Python environment has PyTorch and the `onnx` library installed.
*   Your Go environment is set up for CGO (for ONNX Runtime bindings).

---

## Phase 1: Understanding Current Go Model Saving (JSON Format)

*   **Recap:** As explored, `alphago_demo/cmd/train_models/main.go` uses `RPSPolicyNetwork` and `RPSValueNetwork` from `alphago_demo/pkg/neural/`. These networks have `SaveToFile` methods that write their architecture (sizes) and parameters (weights, biases) to JSON files (e.g., `output/rps_policy1.model`).
*   **Key Information in JSON:** The JSON files contain:
    *   `inputSize`, `hiddenSize`, `outputSize` (for policy)
    *   `weightsInputHidden` (2D array)
    *   `biasesHidden` (1D array)
    *   `weightsHiddenOutput` (2D array)
    *   `biasesOutput` (1D array for policy, single float for value as `biasOutput`)
*   This JSON structure is what our Python conversion script will parse.

---

## Phase 2: Python Script to Convert Go JSON Models to ONNX

This script reads a JSON model file, reconstructs the network in PyTorch, loads the weights, and then exports it to ONNX.

### Step 2.1: Create `scripts/convert_go_json_to_onnx.py`

### Step 2.2: Script Content

```python
import json
import torch
import torch.nn as nn
import os
import numpy as np

# Define PyTorch Model Architectures (matching Go implementation)
# These need to be flexible enough to match what's in the JSON.

class PyTorchPolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Softmax is often applied outside the model during inference or in the loss function
        # For ONNX export, it's generally better to include it if it's part of the core model logic.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x) # Apply softmax for policy probabilities
        return x

class PyTorchValueNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PyTorchValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1) # Output size is 1 for value
        self.sigmoid = nn.Sigmoid() # Sigmoid for value output (0-1 range)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_weights_from_json_to_pytorch_model(model, json_data, model_type):
    """ Loads weights from parsed JSON data into the PyTorch model. """
    # Input to Hidden
    w_ih_json = json_data['weightsInputHidden']
    b_h_json = json_data['biasesHidden']
    model.fc1.weight.data = torch.tensor(np.array(w_ih_json), dtype=torch.float32)
    model.fc1.bias.data = torch.tensor(np.array(b_h_json), dtype=torch.float32)

    # Hidden to Output
    w_ho_json = json_data['weightsHiddenOutput']
    model.fc2.weight.data = torch.tensor(np.array(w_ho_json), dtype=torch.float32)
    
    if model_type == "policy":
        b_o_json = json_data['biasesOutput'] 
        model.fc2.bias.data = torch.tensor(np.array(b_o_json), dtype=torch.float32)
    elif model_type == "value":
        b_o_json = json_data['biasOutput'] # Single float for value net in Go JSON
        model.fc2.bias.data = torch.tensor(np.array([b_o_json]), dtype=torch.float32) # Bias is 1D
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def convert_json_to_onnx(json_model_path, onnx_model_path, model_type):
    """ Converts a Go JSON model to ONNX via PyTorch. """
    print(f"Converting {model_type} model: {json_model_path} -> {onnx_model_path}")

    # 1. Load JSON data
    try:
        with open(json_model_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_model_path}: {e}")
        return

    # 2. Extract architecture parameters
    try:
        input_size = int(json_data['inputSize'])
        hidden_size = int(json_data['hiddenSize'])
        if model_type == "policy":
            output_size = int(json_data['outputSize'])
        elif model_type == "value":
            output_size = 1 # Value net always has output size 1
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except KeyError as e:
        print(f"Error: Missing architecture parameter in JSON ({e}).")
        return
    except ValueError as e:
        print(f"Error: Invalid architecture parameter in JSON ({e}).")
        return

    # 3. Instantiate PyTorch model
    if model_type == "policy":
        pytorch_model = PyTorchPolicyNet(input_size, hidden_size, output_size)
    elif model_type == "value":
        pytorch_model = PyTorchValueNet(input_size, hidden_size)
    else:
        # Should have been caught already, but as a safeguard:
        print(f"Cannot instantiate unknown model_type: {model_type}") 
        return
        
    # 4. Load weights into PyTorch model
    try:
        load_weights_from_json_to_pytorch_model(pytorch_model, json_data, model_type)
    except Exception as e:
        print(f"Error loading weights into PyTorch model: {e}")
        print("Check if JSON structure matches expected fields (weightsInputHidden, etc.)")
        return

    pytorch_model.eval() # Set to evaluation mode

    # 5. Create dummy input for ONNX export
    # Batch size 1 for export, use dynamic axes for variable batch size later
    dummy_input = torch.randn(1, input_size, requires_grad=False)

    # 6. Export to ONNX
    try:
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_model_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11 # Or a later stable version
        )
        print(f"Successfully exported to {onnx_model_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # Example: Convert a policy model
    # Update this path to an actual Go-trained JSON model file
    go_json_policy_model_file = "alphago_demo/output/rps_policy1.model" 
    onnx_policy_output_file = "output/rps_policy1_converted.onnx"

    # Example: Convert a value model
    # Update this path to an actual Go-trained JSON model file
    go_json_value_model_file = "alphago_demo/output/rps_value1.model"
    onnx_value_output_file = "output/rps_value1_converted.onnx"
    # --- End Configuration ---

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_policy_output_file), exist_ok=True)

    print("--- Starting Go JSON to ONNX Conversion ---")
    
    # Convert Policy Model
    if os.path.exists(go_json_policy_model_file):
        convert_json_to_onnx(go_json_policy_model_file, onnx_policy_output_file, "policy")
    else:
        print(f"Policy model JSON not found: {go_json_policy_model_file}")

    # Convert Value Model
    if os.path.exists(go_json_value_model_file):
        convert_json_to_onnx(go_json_value_model_file, onnx_value_output_file, "value")
    else:
        print(f"Value model JSON not found: {go_json_value_model_file}")

    print("--- Conversion Finished ---")
```

### Step 2.3: Run the Conversion Script
1.  **Update Paths:** In `scripts/convert_go_json_to_onnx.py`, ensure `go_json_policy_model_file` and `go_json_value_model_file` point to actual trained model files from your `alphago_demo/output/` directory.
2.  **Execute:**
    ```bash
    python scripts/convert_go_json_to_onnx.py
    ```
3.  **Verify:** Check that `.onnx` files (e.g., `output/rps_policy1_converted.onnx`) are created.
    *   (Optional) Inspect with Netron (`https://netron.app`) to see the graph structure.

---

## Phase 3: Setting up ONNX Runtime in Go (for CPU path)

Refer to **Phase 2** of the previous `REALISTIC_BENCHMARK_TUTORIAL.md` for detailed steps on:
*   Step 2.1: Install ONNX Runtime Go Bindings and Shared Library (`github.com/yalue/onnxruntime_go` is a good choice).
*   Step 2.2: Basic ONNX Model Loading and Inference in Go (Conceptual understanding).

---

## Phase 4: Refactoring `cmd/benchmark/main.go`

### Step 4.1: Modify Command-Line Flags
In `cmd/benchmark/main.go`:
1.  **Add/Modify:**
    *   `onnxModelPath := flag.String("onnx-model-path", "output/rps_policy1_converted.onnx", "Path to the ONNX model file for CPU benchmarks")`
    *   Keep `inputSize`, `outputSize` for input generation and validation against the ONNX model.
    *   Keep `addr` for the Python gRPC service (for GPU path).
    *   Keep `cpuOnly`, `gpuOnly` flags.
2.  **Remove:**
    *   `hiddenSize` flag (architecture comes from ONNX model for CPU path).

### Step 4.2: Update `main()` Function Logic
1.  **Load ONNX Model (for CPU path):**
    *   Near the beginning of `main()`, if not `*gpuOnly`, load the ONNX model specified by `*onnxModelPath` using ONNX Runtime Go bindings.
    *   Store the ONNX session, input name(s), and output name(s).
    *   Validate the `*inputSize` flag against the loaded ONNX model's actual input dimension.
    *   Refer to **Step 3.2 (Update Model Loading Logic)** in `REALISTIC_BENCHMARK_TUTORIAL.md` for example Go code, but use `*onnxModelPath`.

2.  **CPU Benchmark Functions (`benchmarkCPUSingle`, `benchmarkCPUBatch`):**
    *   These will now take the ONNX session, input/output names as arguments (instead of the old `*cpu.RPSCPUPolicyNetwork`).
    *   Inside, they will prepare input tensors, run `session.Run(...)` with the ONNX model, and process results.
    *   Refer to **Step 3.3 (Refactor CPU Benchmarking Functions)** in `REALISTIC_BENCHMARK_TUTORIAL.md`.

3.  **GPU Benchmark Functions (`benchmarkGPUSingle`, `benchmarkGPUBatch`):**
    *   **NO CHANGE NEEDED for these functions or `gpu.NewRPSGPUPolicyNetwork` initially.** They will continue to connect to the existing Python gRPC service which uses TensorFlow.
    *   The `gpuNetwork, err := gpu.NewRPSGPUPolicyNetwork(*addr)` call remains as is for the GPU path.

4.  **Input Data Generation:**
    *   Ensure `generateRandomInput` and `generateRandomBatch` use the `*inputSize` that has been validated against the ONNX model for CPU tests. For GPU tests (calling the Python service), the Python service itself dictates its expected input size (currently 64, as seen in its source).
    *   **Important Consideration:** If the ONNX model (from your Go training) has a different input size (e.g., 81) than what the Python TensorFlow service expects (e.g., 64), the GPU benchmark will be using a different effective model architecture/input than the CPU ONNX benchmark. This is a known discrepancy in this phased approach.

5.  **Performance Comparison Output:**
    *   Update the print statements to clearly state what is being compared: "CPU (ONNX Runtime Go)" vs. "GPU (Python TF Service)".
    *   Implement the actual calculation of speedup as in **Step 3.6** of `REALISTIC_BENCHMARK_TUTORIAL.md`.

---

## Phase 5: Updating `run_benchmark.sh`

1.  **Add ONNX Model Path:**
    *   The script might need a new variable or argument for `ONNX_MODEL_PATH`.
    *   Pass `--onnx-model-path=$ONNX_MODEL_PATH` to the `go run cmd/benchmark/main.go ...` command.
2.  **Python Service Management:**
    *   **Keep the existing logic** for checking and starting `python/neural_service.py`, as it's still needed for the GPU benchmark path.

    ```bash
    # Example additions/modifications in run_benchmark.sh
    ONNX_POLICY_MODEL="output/rps_policy1_converted.onnx" # Default or get from arg
    # ... existing PORT, BATCH_SIZE, ITERATIONS, ARGS parsing ...

    # Ensure Python service is running (existing logic remains)
    # ... (if ! nc -z localhost $PORT ...)

    # Run the benchmark
    echo "Running benchmark: CPU (ONNX Go) vs GPU (Python TF Service)..."
    ARGS_FOR_GO="$ARGS --onnx-model-path=$ONNX_POLICY_MODEL"
    # Add other relevant ONNX model paths if benchmarking value nets too

    go run cmd/benchmark/main.go --batch-size=$BATCH_SIZE --iterations=$ITERATIONS $ARGS_FOR_GO 
    ```

---

## Phase 6: Testing and Validation

1.  **Generate `.model` files:** Run `make alphago-train`.
2.  **Convert to ONNX:** Run `python scripts/convert_go_json_to_onnx.py`.
3.  **Install ONNX Runtime (Go bindings & core library):** If not already done.
4.  **Run Benchmark:** Execute the modified `./run_benchmark.sh`.
5.  **Check Logs & Results:**
    *   Verify the Go benchmark program logs loading the ONNX model for CPU tests.
    *   Verify it still connects to the Python service for GPU tests.
    *   Analyze the CPU (ONNX Go) vs. GPU (Python TF Service) performance figures.

---

## Further Considerations & Next Steps

*   **Model Discrepancy:** Be mindful that this setup benchmarks your *actual trained model* (via ONNX) on CPU against an *ad-hoc TensorFlow model* (in the Python service) on GPU. The architectures/training levels might differ.
*   **Next Step 1: Align GPU Model:** Modify `python/neural_service.py` to load and serve the same `.onnx` files (converted from your Go training) using ONNX Runtime in Python (with CoreML/CUDA execution providers). This would make the CPU vs. GPU comparison truly apples-to-apples in terms of the model being tested.
*   **Next Step 2: Full Go ONNX GPU:** Explore using ONNX Runtime Go bindings directly for GPU inference in `cmd/benchmark/main.go`, removing the Python service dependency from the benchmark entirely. This would fulfill the original vision of the `REALISTIC_BENCHMARK_TUTORIAL.md`.
*   **Policy and Value Models:** This tutorial primarily discusses one model type (e.g., policy). Extend the Python conversion script and Go benchmark to handle both policy and value `.onnx` models if needed.

This phased tutorial allows you to make incremental progress, leverage existing components, and gradually introduce ONNX into your benchmarking pipeline. Good luck! 