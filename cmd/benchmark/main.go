package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	// "github.com/zachbeta/neural_rps/pkg/neural" // Removed unused import
	"github.com/zachbeta/neural_rps/pkg/neural/cpu"
	"github.com/zachbeta/neural_rps/pkg/neural/gpu"
)

const (
	defaultInputSize   = 64
	defaultHiddenSize  = 128
	defaultOutputSize  = 8
	defaultTfGpuAddr   = "localhost:50052" // For the original TensorFlow Python service
	defaultOnnxGpuPort = 50054             // Default for the new ONNX Python gRPC service
)

func generateRandomInput(size int) []float64 {
	input := make([]float64, size)
	for i := range input {
		input[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
	}
	return input
}

func generateRandomBatch(batchSize, inputSize int) [][]float64 {
	batch := make([][]float64, batchSize)
	for i := range batch {
		batch[i] = generateRandomInput(inputSize)
	}
	return batch
}

func benchmarkCPUSingle(network *cpu.RPSCPUPolicyNetwork, inputSize, iterations int) time.Duration {
	input := generateRandomInput(inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.Predict(input)
		if err != nil {
			log.Fatalf("Error during CPU prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkCPUBatch(network *cpu.RPSCPUPolicyNetwork, inputSize, batchSize, iterations int) time.Duration {
	batch := generateRandomBatch(batchSize, inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.PredictBatch(batch)
		if err != nil {
			log.Fatalf("Error during CPU batch prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkGPUSingle(network *gpu.RPSGPUPolicyNetwork, inputSize, iterations int) time.Duration {
	input := generateRandomInput(inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.Predict(input)
		if err != nil {
			log.Fatalf("Error during GPU prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

func benchmarkGPUBatch(network *gpu.RPSGPUPolicyNetwork, inputSize, batchSize, iterations int) time.Duration {
	batch := generateRandomBatch(batchSize, inputSize)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := network.BatchPredict(batch)
		if err != nil {
			log.Fatalf("Error during GPU batch prediction: %v", err)
		}
	}
	elapsed := time.Since(start)

	return elapsed
}

// runCPUAdHocBenchmark_Old runs the original CPU benchmarks using the ad-hoc Go neural network.
func runCPUAdHocBenchmark_Old(inputSize, hiddenSize, outputSize, iterations, batchSize int) {
	fmt.Println("CPU Benchmarks (Ad-hoc Go Network):")
	cpuNetwork, err := cpu.NewRPSCPUPolicyNetwork(inputSize, hiddenSize, outputSize)
	if err != nil {
		log.Fatalf("Failed to create CPU network: %v", err)
	}

	// Single prediction benchmark
	cpuSingleTime := benchmarkCPUSingle(cpuNetwork, inputSize, iterations)
	cpuSingleAvg := float64(cpuSingleTime.Microseconds()) / float64(iterations)
	fmt.Printf("  Single prediction: %v (avg %.2f µs/prediction)\n", cpuSingleTime, cpuSingleAvg)

	// Batch prediction benchmark
	cpuBatchTime := benchmarkCPUBatch(cpuNetwork, inputSize, batchSize, iterations)
	cpuBatchAvg := float64(cpuBatchTime.Microseconds()) / float64(iterations*batchSize)
	fmt.Printf("  Batch prediction:  %v (avg %.2f µs/prediction)\n", cpuBatchTime, cpuBatchAvg)
	fmt.Println()
}

// runCPUONNXBenchmark will run CPU benchmarks using a loaded ONNX model.
// flagInputSize is the input_size passed from the command line, used for ad-hoc or as a reference.
func runCPUONNXBenchmark(onnxModelPath string, flagInputSize, iterations, batchSize int) {
	fmt.Println("CPU Benchmarks (ONNX Model):")
	if onnxModelPath == "" {
		fmt.Println("  ONNX model path not provided, skipping ONNX CPU benchmark.")
		fmt.Println()
		return
	}
	fmt.Printf("  Attempting to load ONNX Model Path: %s\n", onnxModelPath)

	// Set shared library path (assuming it's needed)
	sharedLibraryPath := "/Users/zmorek/go/pkg/mod/github.com/yalue/onnxruntime_go@v1.19.0/test_data/onnxruntime_arm64.dylib"
	ort.SetSharedLibraryPath(sharedLibraryPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// --- Infer Model Input Size ---
	inputsInfo, _, err := ort.GetInputOutputInfo(onnxModelPath)
	if err != nil {
		log.Fatalf("Failed to get ONNX model input/output info for '%s': %v", onnxModelPath, err)
		return
	}
	if len(inputsInfo) == 0 {
		log.Fatalf("ONNX model '%s' has no inputs defined according to GetInputOutputInfo.", onnxModelPath)
		return
	}
	firstInputDims := inputsInfo[0].Dimensions
	if len(firstInputDims) < 2 {
		log.Fatalf("First input of ONNX model '%s' has unexpected dimensions %v (expected at least 2).", onnxModelPath, firstInputDims)
		return
	}
	modelFeatureInputSize := int(firstInputDims[1]) // Assuming shape [batch_size, features]
	if modelFeatureInputSize <= 0 {
		log.Fatalf("Inferred ONNX model input feature size (%d) must be positive. Dimensions from model: %v", modelFeatureInputSize, firstInputDims)
		return
	}
	fmt.Printf("  Inferred ONNX Model Input Feature Size: %d (from model dimensions: %v)\n", modelFeatureInputSize, firstInputDims)
	if flagInputSize != modelFeatureInputSize {
		fmt.Printf("  Note: Command-line --input-size (%d) differs from inferred ONNX model input feature size (%d). Using inferred size for ONNX benchmarks.\n", flagInputSize, modelFeatureInputSize)
	}

	inputNames := []string{"input"}   // Matches the name used during ONNX export from Python
	outputNames := []string{"output"} // Matches the name used during ONNX export from Python

	// Create a new dynamic ONNX session, explicitly specifying float32 for input and output generic types
	session, err := ort.NewDynamicSession[float32, float32](onnxModelPath, inputNames, outputNames)
	if err != nil {
		log.Fatalf("Failed to create dynamic ONNX session for model '%s': %v", onnxModelPath, err)
		return
	}
	defer session.Destroy() // Ensure resources are released

	fmt.Printf("  Successfully loaded ONNX model and created dynamic session.\n")

	// ----- DEBUG: Re-check output info just before running -----
	_, checkOutputInfo, checkErr := ort.GetInputOutputInfo(onnxModelPath)
	if checkErr != nil {
		log.Printf("DEBUG: Failed to re-check output info: %v\n", checkErr)
	} else {
		log.Printf("DEBUG: Re-checked Output Info Count: %d\n", len(checkOutputInfo))
		for i, info := range checkOutputInfo {
			// Assuming float32 type for printing dimensions based on session creation
			log.Printf("DEBUG: Re-checked Output %d - Name: %s, Dimensions: %v\n", i, info.Name, info.Dimensions)
		}
	}
	// ----- END DEBUG -----

	// --- Actual ONNX benchmarking logic starts here ---

	// 1. Prepare a single input tensor
	randomInputFloat64 := generateRandomInput(modelFeatureInputSize) // Returns []float64
	randomInputFloat32 := make([]float32, modelFeatureInputSize)
	for i, val := range randomInputFloat64 {
		randomInputFloat32[i] = float32(val)
	}
	inputShape := ort.NewShape(1, int64(modelFeatureInputSize))
	inputTensor, err := ort.NewTensor(inputShape, randomInputFloat32)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
		return
	}
	defer inputTensor.Destroy()
	fmt.Printf("  Successfully created input tensor with shape %v.\n", inputShape)

	// 2. Run inference for a single input
	// Input tensors slice must be of type []*ort.Tensor[InputT]
	inputTensors := []*ort.Tensor[float32]{inputTensor} // inputTensor is *ort.Tensor[float32]

	// Pre-allocate output tensor(s). For DynamicSession[InputT, OutputT],
	// the Run method expects a slice of output tensors, []*Tensor[OutputT], to be filled.
	// We need to know the expected output shape. Assuming policy model -> [1, 9]
	// NOTE: This assumes the output is for a policy model!
	outputShape := ort.NewShape(1, 9) // Batch size 1, 9 outputs for policy

	outputPlaceholder, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Failed to create empty output tensor: %v", err)
		return
	}
	defer outputPlaceholder.Destroy()
	outputsToFill := []*ort.Tensor[float32]{outputPlaceholder}

	// --- Single Prediction Benchmark Loop ---
	fmt.Println("  Starting single prediction benchmark...")
	start := time.Now()

	for i := 0; i < iterations; i++ {
		err = session.Run(inputTensors, outputsToFill)
		if err != nil {
			// Simple error handling for now
			log.Fatalf("Failed to run ONNX inference during single prediction benchmark (iteration %d): %v", i, err)
			return
		}
	}
	elapsedSingle := time.Since(start)
	avgSingleTime := float64(elapsedSingle.Microseconds()) / float64(iterations)
	fmt.Printf("  Single prediction (ONNX): %v total, (avg %.2f µs/prediction) for %d iterations\n", elapsedSingle, avgSingleTime, iterations)

	// We can verify the last output as a sanity check
	outputTensor := outputsToFill[0] // This is our outputPlaceholder, now filled with data from the last iteration.
	outputData := outputTensor.GetData()
	if len(outputData) == 0 {
		log.Printf("Warning: Output tensor data is empty after single prediction benchmark loop")
	} else {
		fmt.Printf("  Sample predicted value from last iteration (first element): %f\n", outputData[0])
	}

	// --- Batch Prediction Benchmark Loop ---
	fmt.Println("  Starting batch prediction benchmark...")

	// 1. Prepare a batch input tensor using modelFeatureInputSize
	randomBatchFloat64 := generateRandomBatch(batchSize, modelFeatureInputSize) // Returns [][]float64
	randomBatchFloat32 := make([]float32, 0, batchSize*modelFeatureInputSize)
	for _, singleInputFloat64 := range randomBatchFloat64 {
		for _, val := range singleInputFloat64 {
			randomBatchFloat32 = append(randomBatchFloat32, float32(val))
		}
	}
	batchInputShape := ort.NewShape(int64(batchSize), int64(modelFeatureInputSize))
	batchInputTensor, err := ort.NewTensor(batchInputShape, randomBatchFloat32)
	if err != nil {
		log.Fatalf("Failed to create batch input tensor: %v", err)
		return
	}
	defer batchInputTensor.Destroy()
	fmt.Printf("  Successfully created batch input tensor with shape %v.\n", batchInputShape)
	batchInputTensors := []*ort.Tensor[float32]{batchInputTensor}

	// 2. Prepare a batch output placeholder tensor
	// NOTE: Assuming policy model output shape [batchSize, 9]
	batchOutputShape := ort.NewShape(int64(batchSize), 9)
	batchOutputPlaceholder, err := ort.NewEmptyTensor[float32](batchOutputShape)
	if err != nil {
		log.Fatalf("Failed to create batch empty output tensor: %v", err)
		return
	}
	defer batchOutputPlaceholder.Destroy()
	batchOutputsToFill := []*ort.Tensor[float32]{batchOutputPlaceholder}

	// 3. Run batch inference loop
	startBatch := time.Now()
	// Calculate number of batches needed
	numBatches := iterations / batchSize
	if iterations%batchSize != 0 {
		numBatches++
	}
	var totalActualPredictions int64 = 0

	for i := 0; i < numBatches; i++ {
		err = session.Run(batchInputTensors, batchOutputsToFill)
		if err != nil {
			// Simple error handling for now
			log.Fatalf("Failed to run ONNX inference during batch prediction benchmark (batch %d): %v", i, err)
			return
		}
		totalActualPredictions += int64(batchSize) // Count predictions made
	}
	elapsedBatch := time.Since(startBatch)
	avgBatchTime := float64(elapsedBatch.Microseconds()) / float64(totalActualPredictions)
	fmt.Printf("  Batch prediction (ONNX): %v total, (avg %.2f µs/prediction/item) over %d batches (%d total predictions)\n", elapsedBatch, avgBatchTime, numBatches, totalActualPredictions)

	// Sanity check the last batch output
	batchOutputData := batchOutputPlaceholder.GetData()
	if len(batchOutputData) == 0 {
		log.Printf("Warning: Batch output tensor data is empty after benchmark loop")
	} else {
		// Print first element of the first prediction in the batch
		fmt.Printf("  Sample predicted value from last batch iteration (first item, first element): %f\n", batchOutputData[0])
	}

	fmt.Println()
}

// runGPUBenchmark will run GPU benchmarks using a gRPC connection to a Python service.
// If targeting the ONNX Python service, ensure inputSize matches the ONNX model's expected input.
func runGPUBenchmark(addr string, inputSize, hiddenSize, outputSize, iterations, batchSize int, isONNXService bool) {
	serviceType := "TensorFlow Python Service"
	if isONNXService {
		serviceType = "ONNX Python Service"
	}
	fmt.Printf("GPU Benchmarks (%s):\n", serviceType)

	fmt.Printf("  Connecting to GPU service at %s...\n", addr)
	// inputSize, hiddenSize, outputSize are passed here for consistency with CPU network creation
	// and for data generation. For ONNX, the Python service's model defines the true architecture.
	gpuNetwork, err := gpu.NewRPSGPUPolicyNetwork(addr)
	if err != nil {
		log.Fatalf("Failed to create GPU network: %v", err)
	}
	defer gpuNetwork.Close()

	// Single prediction benchmark
	gpuSingleTime := benchmarkGPUSingle(gpuNetwork, inputSize, iterations)
	gpuSingleAvg := float64(gpuSingleTime.Microseconds()) / float64(iterations)
	fmt.Printf("  Single prediction: %v (avg %.2f µs/prediction)\n", gpuSingleTime, gpuSingleAvg)

	// Batch prediction benchmark
	gpuBatchTime := benchmarkGPUBatch(gpuNetwork, inputSize, batchSize, iterations)
	gpuBatchAvg := float64(gpuBatchTime.Microseconds()) / float64(iterations*batchSize)
	fmt.Printf("  Batch prediction:  %v (avg %.2f µs/prediction)\n", gpuBatchTime, gpuBatchAvg)

	// Print network stats
	stats := gpuNetwork.GetStats()
	fmt.Printf("  Total calls: %d, Total positions: %d\n", stats.TotalCalls, stats.TotalBatchSize)
	fmt.Printf("  Avg latency: %.2f µs, Avg batch size: %.2f\n", stats.AvgLatencyUs, stats.AvgBatchSize)
	fmt.Println()
}

// --- NEAT Benchmark Functions ---

// runCPUNEATBenchmark runs CPU benchmarks using a loaded NEAT policy model.
func runCPUNEATBenchmark(neatPolicyModelPath string, iterations, batchSize int) {
	fmt.Println("CPU Benchmarks (NEAT Go Network):")
	if neatPolicyModelPath == "" {
		fmt.Println("  NEAT policy model path not provided, skipping NEAT CPU benchmark.")
		fmt.Println()
		return
	}
	fmt.Printf("  Attempting to load NEAT Policy Model Path: %s\n", neatPolicyModelPath)

	// Load the NEAT policy network - THIS IS THE PROBLEMATIC PART
	// neatNetwork, err := neural.LoadPolicyNetwork(neatPolicyModelPath) // TODO: This is likely the wrong loader for NEAT models
	// if err != nil {
	// 	log.Fatalf("Failed to load NEAT policy network from '%s': %v", neatPolicyModelPath, err)
	// }
	// fmt.Println("  Successfully loaded NEAT policy network.")
	// We assume NEAT models don't need explicit Close(), but add if necessary
	// defer neatNetwork.Close()

	// Infer input size from the loaded network
	// inputSize := neatNetwork.GetInputSize() // Depends on neatNetwork
	// if inputSize <= 0 {
	// 	log.Fatalf("Loaded NEAT network has invalid input size: %d", inputSize)
	// }
	// fmt.Printf("  Inferred NEAT Model Input Size: %d\n", inputSize)

	// TODO: Implement benchmarkNEATSingle and benchmarkNEATBatch helpers
	// TODO: Call helpers and print results

	fmt.Println("  NEAT benchmark implementation pending...") // Placeholder

	fmt.Println()
}

func main() {
	// Define command line flags
	inputSize := flag.Int("input-size", defaultInputSize, "Input layer size for neural networks (used by AdHoc, ignored by ONNX/NEAT)")
	hiddenSize := flag.Int("hidden-size", defaultHiddenSize, "Hidden layer size for neural networks (used by AdHoc)")
	outputSize := flag.Int("output-size", defaultOutputSize, "Output layer size for neural networks (used by AdHoc)")
	iterations := flag.Int("iterations", 1000, "Number of iterations for each benchmark")
	batchSize := flag.Int("batch-size", 32, "Batch size for batch predictions")
	tfGpuAddr := flag.String("gpu-addr", defaultTfGpuAddr, "Address of the TensorFlow Python gRPC service (legacy GPU benchmark)")
	onnxGpuPort := flag.Int("onnx-gpu-port", defaultOnnxGpuPort, "Port for the ONNX Python gRPC service (new GPU benchmark)")
	onnxModelPath := flag.String("onnx-model", "", "Path to the ONNX model for CPU benchmarks (e.g., ./output/rps_value1.onnx)")
	neatPolicyModelPath := flag.String("neat-policy-model", "", "Path to the NEAT policy model (.model) for CPU benchmarks")
	runCPUAdHoc := flag.Bool("run-cpu-adhoc", true, "Run CPU benchmarks with ad-hoc Go network")
	runCPUONNX := flag.Bool("run-cpu-onnx", true, "Run CPU benchmarks with ONNX model")
	runCPUNEAT := flag.Bool("run-cpu-neat", true, "Run CPU benchmarks with NEAT model")
	runGpuTF := flag.Bool("run-gpu-tf", false, "Run GPU benchmarks with the (legacy) TensorFlow Python service")
	runGpuONNX := flag.Bool("run-gpu-onnx", true, "Run GPU benchmarks with the ONNX Python service")

	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Printf("Benchmark Configuration:\n")
	fmt.Printf("  Iterations: %d\n", *iterations)
	fmt.Printf("  Batch Size: %d\n", *batchSize)
	if *runCPUAdHoc {
		fmt.Printf("  AdHoc Input Size: %d\n", *inputSize)
		fmt.Printf("  AdHoc Hidden Size: %d\n", *hiddenSize)
		fmt.Printf("  AdHoc Output Size: %d\n", *outputSize)
	}
	if *runCPUONNX {
		if *onnxModelPath == "" {
			fmt.Println("  ONNX Model Path for CPU: Not provided (CPU ONNX benchmarks will be skipped)")
		} else {
			fmt.Println("  ONNX Model Path for CPU:", *onnxModelPath)
		}
	}
	if *runCPUNEAT {
		if *neatPolicyModelPath == "" {
			fmt.Println("  NEAT Policy Model Path for CPU: Not provided (CPU NEAT benchmarks will be skipped)")
		} else {
			fmt.Println("  NEAT Policy Model Path for CPU:", *neatPolicyModelPath)
		}
	}
	if *runGpuTF {
		fmt.Println("  GPU Service Address (TensorFlow):", *tfGpuAddr)
	}
	if *runGpuONNX {
		fmt.Printf("  GPU Service Port (ONNX Python): %d\n", *onnxGpuPort)
	}
	fmt.Println()

	if *runCPUAdHoc {
		runCPUAdHocBenchmark_Old(*inputSize, *hiddenSize, *outputSize, *iterations, *batchSize)
	}

	if *runCPUONNX {
		runCPUONNXBenchmark(*onnxModelPath, *inputSize, *iterations, *batchSize)
	}

	if *runCPUNEAT {
		runCPUNEATBenchmark(*neatPolicyModelPath, *iterations, *batchSize)
	}

	// GPU Benchmarks with TensorFlow service (legacy)
	if *runGpuTF {
		runGPUBenchmark(*tfGpuAddr, *inputSize, *hiddenSize, *outputSize, *iterations, *batchSize, false)
	}

	// GPU Benchmarks with ONNX Python service
	if *runGpuONNX {
		onnxGpuServiceAddr := fmt.Sprintf("localhost:%d", *onnxGpuPort)
		runGPUBenchmark(onnxGpuServiceAddr, *inputSize, *hiddenSize, *outputSize, *iterations, *batchSize, true)
	}
}

// Helper min function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
