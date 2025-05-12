# Tutorial: Implementing a Realistic Neural Network Benchmark

**Goal:** To refactor the existing benchmark (`cmd/benchmark/main.go`) to use actual trained PyTorch models. This will be achieved by converting the PyTorch models to the ONNX (Open Neural Network Exchange) format and then using ONNX Runtime in Go to perform inference for both CPU and GPU benchmarking. This approach will provide more accurate performance insights and remove the dependency on the Python gRPC service for GPU tasks.

**Target Audience:** Mid-level Software Engineer.

**Assumptions:**
*   You have a working checkout of the `neural_rps` project.
*   You are familiar with basic Go and Python development.
*   You have a conceptual understanding of neural networks (layers, input/output tensors).
*   Your Python environment has PyTorch and the `onnx` library installed.
*   Your Go environment is set up.

---

## Phase 1: Understanding and Exporting Trained Models to ONNX

The first step is to take our existing trained PyTorch models and convert them into the ONNX format.

### Step 1.1: Identify Trained Model Files and Architecture
1.  **Locate Trained Models:**
    *   Run `make alphago-train` if you haven't already. This command saves trained models, typically in a subdirectory within `alphago_demo/output/` (e.g., `output/rps_h128_g1000_e10_2025MMDD-HHMMSS_policy.model`).
    *   Note the filenames, especially distinguishing between policy and value network models if both are present and relevant for benchmarking.
2.  **Determine Model Architecture:**
    *   You'll need to know the exact architecture of the PyTorch model you intend to export. This includes:
        *   Input tensor shape (e.g., what does the `81` in "81-64-9" from tournament logs correspond to?)
        *   Output tensor shape(s).
        *   The structure of layers.
    *   This information is found in the Python script where the PyTorch `nn.Module` class for your policy (and/or value) network is defined. This is likely located within the `alphago_demo` Python codebase or the Python code responsible for the neural network aspects of MCTS.
    *   For example, if the input is a flattened game board of 9x9, the input size would be 81.

### Step 1.2: Write a Python Script to Export Models to ONNX
1.  **Create `scripts/export_to_onnx.py`**.
2.  **Script Content:** This script will load your trained PyTorch model and use `torch.onnx.export()` to save it as an `.onnx` file.

    ```python
    import torch
    import os # Or wherever your model definition is
    # Make sure to import your PyTorch model class definition
    # from its location within the project, e.g.:
    # from alphago_demo.your_python_module import YourPolicyNetModel

    # Example (replace with your actual model class and path):
    # Assuming YourPolicyNetModel is defined elsewhere and can be imported
    # class YourPolicyNetModel(torch.nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(YourPolicyNetModel, self).__init__()
    #         self.fc1 = torch.nn.Linear(input_size, hidden_size)
    #         self.relu = torch.nn.ReLU()
    #         self.fc2 = torch.nn.Linear(hidden_size, output_size)
    #         # self.softmax = torch.nn.Softmax(dim=-1) # Often not included in export if loss handles it

    #     def forward(self, x):
    #         x = self.fc1(x)
    #         x = self.relu(x)
    #         x = self.fc2(x)
    #         # x = self.softmax(x)
    #         return x

    def export_model_to_onnx(pytorch_model_path, onnx_model_path, input_size, hidden_size, output_size):
        """
        Loads a trained PyTorch model, and exports it to ONNX format.

        Args:
            pytorch_model_path (str): Path to the saved PyTorch .model file (state_dict).
            onnx_model_path (str): Path where the .onnx model will be saved.
            input_size (int): The input dimension for the model.
            hidden_size (int): The hidden dimension for the model.
            output_size (int): The output dimension for the model.
        """
        # 1. Instantiate your model architecture
        #    This MUST match the architecture of the model whose state_dict you are loading.
        #    Replace YourPolicyNetModel with your actual model class.
        model = YourPolicyNetModel(input_size, hidden_size, output_size) # Ensure class is defined/imported

        # 2. Load the trained weights (state_dict)
        try:
            # Adjust map_location if you trained on GPU and are exporting on CPU
            state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading state_dict from {pytorch_model_path}: {e}")
            print("Ensure the model class instantiation matches the saved model's architecture.")
            return

        # 3. Set the model to evaluation mode (important for layers like dropout, batchnorm)
        model.eval()

        # 4. Create a dummy input tensor with the correct shape and type
        #    Batch size is typically 1 for export, or use dynamic axes for variable batch size.
        #    The shape must match what your model's forward() method expects.
        dummy_input = torch.randn(1, input_size, requires_grad=False) # e.g., (1, 81) for an input_size of 81

        # 5. Define dynamic axes if you want to support variable batch sizes or sequence lengths
        #    This is highly recommended for flexibility.
        dynamic_axes = {
            'input': {0: 'batch_size'},  # batch_size is dynamic for input
            'output': {0: 'batch_size'} # batch_size is dynamic for output
        }

        # 6. Export the model
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_model_path,
                input_names=['input'],         # Names for input tensors in the ONNX graph
                output_names=['output'],       # Names for output tensors in the ONNX graph
                dynamic_axes=dynamic_axes,
                opset_version=11,              # Choose a suitable opset version
                verbose=False
            )
            print(f"Model successfully exported to {onnx_model_path}")
        except Exception as e:
            print(f"Error during ONNX export: {e}")

    if __name__ == '__main__':
        # --- CONFIGURATION ---
        # Replace with the actual path to your trained PyTorch model
        trained_model_file = "path/to/your/trained/output/rps_policy.model"
        # Define the output path for the ONNX model
        onnx_output_file = "output/rps_policy_trained.onnx"
        
        # These MUST match the architecture of the `trained_model_file`
        # Example: if your trained model was 81 input, 128 hidden, 9 output
        MODEL_INPUT_SIZE = 81  # Update this
        MODEL_HIDDEN_SIZE = 128 # Update this
        MODEL_OUTPUT_SIZE = 9   # Update this
        # --- END CONFIGURATION ---

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(onnx_output_file), exist_ok=True)

        if not os.path.exists(trained_model_file):
            print(f"Error: Trained model file not found at {trained_model_file}")
            print("Please ensure you have run training and updated the path.")
        else:
            export_model_to_onnx(
                trained_model_file, 
                onnx_output_file,
                MODEL_INPUT_SIZE,
                MODEL_HIDDEN_SIZE,
                MODEL_OUTPUT_SIZE
            )
        
        # Example for a different model (e.g., value network)
        # trained_value_model_file = "path/to/your/trained/output/rps_value.model"
        # onnx_value_output_file = "output/rps_value_trained.onnx"
        # VALUE_MODEL_INPUT_SIZE = 81
        # VALUE_MODEL_HIDDEN_SIZE = 64
        # VALUE_MODEL_OUTPUT_SIZE = 1 
        # export_model_to_onnx(
        #     trained_value_model_file,
        #     onnx_value_output_file,
        #     VALUE_MODEL_INPUT_SIZE,
        #     VALUE_MODEL_HIDDEN_SIZE,
        #     VALUE_MODEL_OUTPUT_SIZE
        # )
    ```
    *   **Important:** You **must** replace `YourPolicyNetModel` with the actual class definition of your PyTorch model. You'll also need to update `MODEL_INPUT_SIZE`, `MODEL_HIDDEN_SIZE`, `MODEL_OUTPUT_SIZE` to match the architecture of the specific trained model file you are exporting.

### Step 1.3: Generate ONNX Models
1.  **Configure Paths:** In `scripts/export_to_onnx.py`, update `trained_model_file` to point to one of your actual `.model` files from training (e.g., from `alphago_demo/output/`). Update `MODEL_INPUT_SIZE`, etc., to match that model's architecture.
2.  **Run the Export Script:**
    ```bash
    python scripts/export_to_onnx.py
    ```
3.  **Verify:** Check that an `.onnx` file (e.g., `output/rps_policy_trained.onnx`) has been created.
    *   (Optional) You can inspect the ONNX model using tools like Netron (`https://netron.app`) to visualize its structure and verify input/output names and shapes.

---

## Phase 2: Setting up ONNX Runtime in Go

Now, we'll integrate ONNX Runtime into our Go environment to load and run these `.onnx` models.

### Step 2.1: Install ONNX Runtime Go Bindings and Shared Library
1.  **Install ONNX Runtime Core:** ONNX Runtime Go bindings are wrappers around the ONNX Runtime C/C++ shared library. You need to have this shared library installed on your system.
    *   **macOS (Homebrew):** `brew install onnxruntime`
    *   **Linux/Windows:** Download pre-built binaries from the [ONNX Runtime GitHub Releases](https://github.com/microsoft/onnxruntime/releases) (choose a version compatible with the Go bindings, often the latest stable C/C++ API version). Ensure the shared library (`libonnxruntime.so` on Linux, `onnxruntime.dll` on Windows, `libonnxruntime.dylib` on macOS) is in your system's library path or accessible by your Go application.
2.  **Choose and Install Go Bindings:** A common Go binding is `github.com/yalue/onnxruntime_go`.
    ```bash
    go get github.com/yalue/onnxruntime_go
    ```
    *   Ensure your `CGO_CFLAGS` and `CGO_LDFLAGS` are set up correctly if the ONNX Runtime headers and library are in non-standard locations. For Homebrew installs on macOS, it's often automatic. For manual installs:
        ```bash
        # Example for Linux if installed in /opt/onnxruntime
        # export CGO_CFLAGS="-I/opt/onnxruntime/include"
        # export CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime"
        # For macOS with Homebrew, it might be:
        # export CGO_CFLAGS="-I$(brew --prefix onnxruntime)/include"
        # export CGO_LDFLAGS="-L$(brew --prefix onnxruntime)/lib -lonnxruntime"
        ```

### Step 2.2: Basic ONNX Model Loading and Inference in Go (Conceptual)
Before modifying the benchmark, let's understand the basics:

```go
// (This is a conceptual example, refer to chosen binding's documentation)
// import "github.com/yalue/onnxruntime_go" // Or your chosen binding

// // Initialize ONNX Runtime (once per application)
// onnxruntime_go.Initialize() // Or specific init for your binding
// defer onnxruntime_go.Destroy()

// // Load the model
// model, err := onnxruntime_go.NewSession("path/to/your/output/rps_policy_trained.onnx")
// if err != nil { /* handle error */ }
// defer model.Destroy()

// // Assume inputTensor is a []float32 slice prepared correctly
// // The shape here [1, inputSizeFromModel] must match the model's expected input shape.
// // The "1" is the batch size.
// inputShape := []int64{1, int64(inputSizeFromModel)} 
// inputOrtTensor, err := onnxruntime_go.NewTensor(inputShape, inputTensor)
// if err != nil { /* handle error */ }
// defer inputOrtTensor.Destroy()

// // Run inference (names 'input' and 'output' must match those defined during ONNX export)
// results, err := model.Run([]onnxruntime_go.Tensor{inputOrtTensor}, []string{"input"}, []string{"output"})
// if err != nil { /* handle error */ }
// defer results[0].Destroy()

// // Process output
// outputData := results[0].GetData().([]float32) 
// // outputData now contains the raw float32 predictions
```
*   **Note:** The exact API calls might differ slightly based on the Go binding you choose. Always refer to its documentation. The `github.com/yalue/onnxruntime_go` library is a popular choice.

---

## Phase 3: Refactoring `cmd/benchmark/main.go`

Now, let's modify the actual benchmark Go program.

### Step 3.1: Modify Command-Line Flags
In `cmd/benchmark/main.go`:
1.  **Remove:**
    *   `hiddenSize := flag.Int("hidden-size", ...)` (The architecture is now defined by the ONNX model).
2.  **Add:**
    *   `modelPath := flag.String("model-path", "output/rps_policy_trained.onnx", "Path to the ONNX model file")`
3.  **Keep (for validation/input generation):**
    *   `inputSize := flag.Int("input-size", defaultInputSize, ...)` (This should now match the loaded ONNX model's input size).
    *   `outputSize := flag.Int("output-size", defaultOutputSize, ...)` (This should match the loaded ONNX model's output size).
    *   Remove `defaultAddr` and the `addr` flag as the gRPC service will no longer be used.

### Step 3.2: Update Model Loading Logic
In `main()` function of `cmd/benchmark/main.go`:
1.  **Remove CPU/GPU network creation:**
    *   Delete `cpu.NewRPSCPUPolicyNetwork(...)`.
    *   Delete `gpu.NewRPSGPUPolicyNetwork(...)` and its `defer gpuNetwork.Close()`.
2.  **Load ONNX Model:**
    *   Add code to initialize ONNX Runtime (globally, once).
    *   Load the ONNX model specified by `*modelPath` using your chosen Go ONNX Runtime binding.
    *   Get the expected input and output shapes/names from the loaded ONNX model. This is crucial. Many ONNX Runtime bindings provide functions to inspect model metadata.
    *   **Validate:** Compare the ONNX model's actual input/output dimensions with the `*inputSize` and `*outputSize` flags. Log a warning or error if they don't match. The ONNX model's dimensions are the source of truth.

    ```go
    // Example (using yalue/onnxruntime_go syntax, place near start of main)
    // Needs to be adapted based on actual library used.
    // This is a simplified version. You'll need to handle errors and resource management.
    
    // Initialize ONNX Runtime (if required by your binding, typically once)
    // Check your binding's documentation for initialization and destruction.
    // e.g., onnxruntime_go.Initialize() might be needed.
    // Consider deferring a corresponding onnxruntime_go.Destroy() or similar.

    onnxSession, err := onnxruntime_go.NewSession(*modelPath) // This path comes from the new flag
    if err != nil {
        log.Fatalf("Failed to create ONNX session from %s: %v", *modelPath, err)
    }
    defer onnxSession.Destroy()

    // --- Get model metadata (IMPORTANT!) ---
    // This part is crucial and API-dependent. Example for yalue/onnxruntime_go:
    modelInputs := onnxSession.GetInputInfos()
    if len(modelInputs) == 0 {
        log.Fatalf("ONNX model has no inputs!")
    }
    onnxInputName := modelInputs[0].Name // e.g., "input"
    onnxInputShape := modelInputs[0].Dimensions // e.g., [-1, 81] where -1 is batch_size

    modelOutputs := onnxSession.GetOutputInfos()
    if len(modelOutputs) == 0 {
        log.Fatalf("ONNX model has no outputs!")
    }
    onnxOutputName := modelOutputs[0].Name // e.g., "output"
    // onnxOutputShape := modelOutputs[0].Dimensions

    // Validate *inputSize against onnxInputShape (e.g., onnxInputShape[1])
    // The first dimension of onnxInputShape is often batch size (e.g., -1 or 1 if not dynamic)
    // The second dimension is the feature size.
    actualModelInputSize := onnxInputShape[len(onnxInputShape)-1] // Last dim is usually feature count
    if int64(*inputSize) != actualModelInputSize {
        log.Printf("Warning: CLI --input-size (%d) does not match ONNX model's input size (%d). Using model's size.", *inputSize, actualModelInputSize)
        *inputSize = int(actualModelInputSize) // Use the model's actual size
    }
    // Similarly for output size if needed for validation.
    // --- End Get model metadata ---
    ```

### Step 3.3: Refactor CPU Benchmarking Functions
Modify `benchmarkCPUSingle` and `benchmarkCPUBatch`:
1.  **Change Signature:** They will now take the `onnxSession` (or equivalent from your binding) and the `onnxInputName`, `onnxOutputName` as arguments instead of `*cpu.RPSCPUPolicyNetwork`.
    ```go
    // func benchmarkCPUSingle(session *onnxruntime_go.Session, inputName string, outputName string, inputSize int, iterations int) time.Duration { ... }
    ```
2.  **Prediction Logic:**
    *   Inside the loop, prepare the input `[]float32` data.
    *   Convert it to an ONNX Runtime tensor (e.g., `onnxruntime_go.NewTensor`). Ensure the shape is correct (e.g., `[]int64{1, int64(inputSize)}` for single, `[]int64{int64(batchSize), int64(inputSize)}` for batch).
    *   Call `session.Run(...)` using the CPU execution provider (usually default).
    *   Get the output tensor and (optionally) convert its data back to a Go slice.
    *   Remember to `Destroy()` tensors after use to free memory.

### Step 3.4: Refactor GPU Benchmarking Functions
Modify `benchmarkGPUSingle` and `benchmarkGPUBatch`:
1.  **Change Signature:** Similar to CPU, they take `onnxSession`, `onnxInputName`, `onnxOutputName`.
2.  **GPU Execution Provider:**
    *   When you create the ONNX session (or perhaps as an option to `Run`), you'll need to specify a GPU execution provider. This is API-specific to your Go ONNX Runtime binding.
    *   Common providers: "CUDAExecutionProvider", "CoreMLExecutionProvider" (for macOS), "ROCmExecutionProvider", "DirectMLExecutionProvider".
    *   This may require building/installing ONNX Runtime with support for that specific provider.
    *   **Example (conceptual, check your binding's docs):**
        ```go
        // gpuSessionOptions := onnxruntime_go.NewSessionOptions()
        // err := gpuSessionOptions.AppendExecutionProvider_CUDA(0) // 0 for device ID
        // if err != nil { log.Fatalf("Failed to set CUDA provider: %v", err) }
        // onnxGPUSession, err := onnxruntime_go.NewSessionWithONNXModel(*modelPath, gpuSessionOptions)
        // defer onnxGPUSession.Destroy()
        // Use onnxGPUSession in benchmarkGPU... functions
        ```
        Alternatively, some bindings might allow specifying the provider per `Run` call, or the session might try to use GPU if available by default if ONNX Runtime was built with GPU support. This needs careful checking with the chosen binding.
3.  **Prediction Logic:** Similar to the CPU functions, but ensure the ONNX Runtime session is configured to use the GPU.

### Step 3.5: Update Input Data Generation
*   The `generateRandomInput(size int)` and `generateRandomBatch(batchSize, inputSize int)` functions should now use the `*inputSize` that has been validated against (or set by) the ONNX model's actual input dimension.
*   Ensure the generated `[]float64` is correctly shaped into a `[]float32` slice if your ONNX model expects float32 (common). ONNX Runtime bindings usually handle this conversion when creating the tensor.

### Step 3.6: Update Performance Comparison
In the `main` function, after CPU and GPU benchmarks:
1.  **Capture Timings:** Ensure `cpuSingleTime`, `cpuBatchTime`, `gpuSingleTime`, `gpuBatchTime` are correctly calculated and stored.
2.  **Calculate Speedup:**
    ```go
    if !*cpuOnly && !*gpuOnly {
        fmt.Println("Performance Comparison:")
        if cpuSingleTime.Microseconds() > 0 && gpuSingleTime.Microseconds() > 0 {
            singleSpeedup := float64(cpuSingleTime.Microseconds()) / float64(gpuSingleTime.Microseconds())
            fmt.Printf("  Single prediction speedup: GPU is %.2f times %s than CPU\n", 
                singleSpeedup, if singleSpeedup >= 1.0 {"faster"} else {"slower"})
        }
        // ... similar for batchSpeedup ...
    }
    ```

---

## Phase 4: Updating `run_benchmark.sh`

The shell script also needs minor changes.

### Step 4.1: Remove Python Service Management
In `run_benchmark.sh`:
*   Delete the entire `if ! nc -z localhost $PORT ... else ... fi` block that checks, starts, and manages `python/neural_service.py`. This is no longer needed.
*   Remove the `PORT` variable and its parsing.

### Step 4.2: Update `go run` Command
Modify the `go run` line:
*   Pass the new `--model-path` flag. You might want to make this configurable in the shell script or assume a default location (e.g., `output/rps_policy_trained.onnx`).
*   Remove the `--addr` argument as it's no longer used.
*   Example:
    ```bash
    # run_benchmark.sh
    MODEL_FILE="output/rps_policy_trained.onnx" # Or pass as script arg
    # ... other args parsing ...
    echo "Running benchmark with ONNX model $MODEL_FILE, batch size $BATCH_SIZE and $ITERATIONS iterations..."
    go run cmd/benchmark/main.go --model-path=$MODEL_FILE --batch-size=$BATCH_SIZE --iterations=$ITERATIONS $ARGS 
    ```

---

## Phase 5: Testing and Validation

1.  **Build ONNX Runtime (if needed):** If you're targeting a specific GPU provider (like CUDA), you might need to build ONNX Runtime from source with that provider enabled if pre-built binaries don't suit your needs or Go binding.
2.  **Run:** Execute the modified `./run_benchmark.sh`.
3.  **Check Logs:** Look for any errors during ONNX model loading, tensor creation, or inference.
4.  **Analyze Results:**
    *   Are the CPU results sensible for the loaded model?
    *   Are the GPU results improved? (This is the key outcome!)
    *   Does the performance scale with batch size as expected?
5.  **Troubleshooting:**
    *   **ONNX Export Issues:** If `torch.onnx.export` fails, check model compatibility with ONNX, input shapes, and opset versions.
    *   **Go ONNX Runtime Errors:** Consult the Go binding's documentation for error messages. Common issues are incorrect library paths, version mismatches between binding and shared library, or incorrect tensor shapes/types.
    *   **GPU Not Used:** If GPU performance is still bad, ensure ONNX Runtime is actually using the GPU. Check for logs from ONNX Runtime or use GPU monitoring tools. Ensure the execution provider is correctly specified and available.

---

## Further Considerations

*   **Policy vs. Value Networks:** This tutorial focused on a single model (e.g., policy). You might want to benchmark both policy and value networks if they have different characteristics or are used differently. This would involve exporting both to ONNX and adding logic to the benchmark to test them.
*   **Multiple GPU Providers:** If you need to support different GPU types (NVIDIA, AMD, Apple Silicon), you'll need to handle the selection of the correct ONNX Runtime execution provider.
*   **Warm-up Runs:** For more stable benchmark results, consider adding a few "warm-up" inference calls before starting the timed measurements, especially for GPU.
*   **Input Data:** Currently, random data is used. For more application-specific benchmarks, you might consider using representative game state data as input.

This tutorial provides a comprehensive roadmap. Remember to consult the documentation for PyTorch (ONNX export) and your chosen Go ONNX Runtime binding for specific API details and troubleshooting. Good luck! 