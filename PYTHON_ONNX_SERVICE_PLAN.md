# Plan: Python gRPC Service with ONNX Runtime (for GPU Benchmarking)

This plan outlines the steps to create a new Python gRPC service that uses ONNX Runtime for model inference, intended to be benchmarked for GPU performance. This service will run in parallel with the existing TensorFlow-based gRPC service.

## Phase 1: Setup New Service Files & Scripts

1.  **Duplicate Python Service Script:**
    *   Copy `python/neural_service.py` to `python/neural_service_onnx.py`.
    *   This new script will be modified to use ONNX Runtime.

2.  **Duplicate Startup Shell Script:**
    *   Copy `start_neural_service.sh` to `start_neural_service_onnx.sh`.
    *   This new script will launch `neural_service_onnx.py`.

## Phase 2: Modify `start_neural_service_onnx.sh`

1.  **Target Script:**
    *   Change the script to execute `python -u neural_service_onnx.py` (or similar, ensuring it points to the new Python script).

2.  **Command-Line Arguments for `neural_service_onnx.py`:**
    *   The script should accept command-line arguments to pass to `neural_service_onnx.py`:
        *   `--port`: The gRPC port number for the service to listen on.
            *   Default: `50053` (to differentiate from the TF service's default `50052`).
        *   `--model_path`: The file path to the `.onnx` model.
            *   Example default: `python/output/rps_value1.onnx` or make it a required argument.

3.  **Pass Arguments:**
    *   Ensure these arguments are correctly passed when invoking `neural_service_onnx.py`.

## Phase 3: Modify `python/neural_service_onnx.py`

1.  **Add Imports:**
    *   `import onnxruntime as ort`
    *   `import numpy as np`
    *   `import argparse`
    *   Remove TensorFlow-related imports.

2.  **Argument Parsing:**
    *   Implement `argparse` at the beginning of the script to:
        *   Accept `--port` (integer, with default, e.g., 50053).
        *   Accept `--model_path` (string, required or with a default).

3.  **NeuralServicer Class (`__init__`):**
    *   Remove TensorFlow model loading.
    *   Load the ONNX model using `ort.InferenceSession`:
        *   `self.ort_session = ort.InferenceSession(args.model_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])`
        *   Store the session.
    *   Get and store input and output names from `self.ort_session.get_inputs()[0].name` and `self.ort_session.get_outputs()[0].name`.
        *   (For `rps_value1.onnx`, these are likely "input" and "output").

4.  **NeuralServicer Class (`Predict` method):**
    *   Convert the input `request.state` (which is `repeated double`) into a 1D NumPy array of `np.float32`.
    *   Reshape it to `(1, num_features)`. The `num_features` should match the ONNX model's expected input dimension (e.g., 81 for `rps_value1.onnx`). This can be obtained from `self.ort_session.get_inputs()[0].shape`.
    *   Prepare the input dictionary: `inputs = {self.input_name: numpy_input_data}`.
    *   Run inference: `results = self.ort_session.run([self.output_name], inputs)`.
    *   `results[0]` will be a NumPy array (e.g., `[[value]]`). Extract the scalar value.
    *   Populate and return the gRPC `PredictionResponse` with this value.

5.  **NeuralServicer Class (`BatchPredict` method):**
    *   Iterate through `request.batch_state` (list of `StateProto`).
    *   For each `StateProto`, convert its `state` (`repeated double`) to a 1D NumPy array of `np.float32`.
    *   Collect these 1D arrays into a list and then stack them into a 2D NumPy array of shape `(batch_size, num_features)` and `np.float32` dtype.
    *   Prepare the input dictionary: `inputs = {self.input_name: numpy_batch_input_data}`.
    *   Run inference: `results = self.ort_session.run([self.output_name], inputs)`.
    *   `results[0]` will be a NumPy array of shape `(batch_size, 1)`.
    *   Flatten or iterate through this result array to get individual prediction values.
    *   Populate and return the gRPC `BatchPredictionResponse` with these values.

6.  **gRPC Server Setup (in `serve()` function):**
    *   Ensure the server uses the port specified by `args.port`.

## Phase 4: Modify Go Benchmark (`cmd/benchmark/main.go`)

1.  **Add New Flag:**
    *   `--onnx-gpu-addr`: String flag for the address of the new ONNX Python gRPC service.
        *   Default: `localhost:50053`.

2.  **New Benchmark Function `runGPUONNXBenchmark()`:**
    *   This function will be similar in structure to how the current GPU benchmark is called, but will use the `--onnx-gpu-addr`.
    *   It will connect to the `neural_service_onnx.py` instance.
    *   It can reuse the existing `benchmarkGPUSingle` and `benchmarkGPUBatch` helper functions, as they are generic enough to work with any service that conforms to the gRPC interface, provided the `gpu.NewRPSGPUPolicyNetwork` is pointed to the correct address.

3.  **Update `main()` Logic:**
    *   Add a new condition or modify the existing GPU benchmark call to optionally use `runGPUONNXBenchmark()` (perhaps controlled by a new flag like `--gpu-type=tf` vs `--gpu-type=onnx`, or by simply running both if different addresses are provided). For now, we might just add it to run alongside the existing GPU benchmark if `--onnx-gpu-addr` is set.

## Phase 5: Testing & Iteration

1.  Start the new `start_neural_service_onnx.sh` script.
2.  Run the Go benchmark (`cmd/benchmark/main.go`) with the appropriate flags to target the new ONNX GPU service.
3.  Verify functionality and performance.
4.  Compare with CPU ONNX and the original TensorFlow GPU service. 