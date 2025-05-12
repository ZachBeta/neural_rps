package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/zachbeta/neural_rps/pkg/neural/cpu"
	"github.com/zachbeta/neural_rps/pkg/neural/gpu"
)

const (
	defaultInputSize  = 64
	defaultHiddenSize = 128
	defaultOutputSize = 8
	defaultAddr       = "localhost:50052"
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
func runCPUONNXBenchmark(onnxModelPath string, inputSize, iterations, batchSize int) {
	fmt.Println("CPU Benchmarks (ONNX Model):")
	if onnxModelPath == "" {
		fmt.Println("  ONNX model path not provided, skipping ONNX CPU benchmark.")
		fmt.Println()
		return
	}
	fmt.Printf("  Attempting to load ONNX Model Path: %s\n", onnxModelPath)

	// Explicitly set the path to the ONNX Runtime shared library.
	// This path was found by inspecting the contents of the Go module directory for yalue/onnxruntime_go.
	// For macOS ARM64, the library is typically named onnxruntime_arm64.dylib.
	// The version v1.19.0 corresponds to the version of the Go wrapper module being used.
	sharedLibraryPath := "/Users/zmorek/go/pkg/mod/github.com/yalue/onnxruntime_go@v1.19.0/test_data/onnxruntime_arm64.dylib"
	ort.SetSharedLibraryPath(sharedLibraryPath)

	// Initialize ONNX runtime; this is a good place to do it once.
	// Note: For multiple models or dynamic library paths, more complex initialization might be needed.
	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime environment: %v", err)
	}
	// Defer finalization of the ONNX Runtime environment.
	// According to docs, this should be called when no more ORT functions are needed.
	defer ort.DestroyEnvironment()

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

	// --- Actual ONNX benchmarking logic starts here ---

	// 1. Prepare a single input tensor
	// Note: The inputSize for rps_value1.model is 81.
	// The 'inputSize' parameter to this function should match the model's requirement.
	if inputSize != 81 { // Temporary check for our specific model
		log.Printf("Warning: inputSize %d does not match rps_value1.model expected inputSize of 81. Using %d.", inputSize, inputSize)
	}

	randomInputFloat64 := generateRandomInput(inputSize) // Returns []float64
	randomInputFloat32 := make([]float32, inputSize)
	for i, val := range randomInputFloat64 {
		randomInputFloat32[i] = float32(val)
	}

	// Shape for a single input: [1, inputSize]
	// The batch dimension is 1 because we are creating a tensor for a single prediction.
	inputShape := ort.NewShape(1, int64(inputSize))
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
	// For rps_value1.model (a ValueNet), the output is expected to be a single float32 value.
	// The shape for a single output item is typically [1, 1] for a scalar.
	outputShape := ort.NewShape(1, 1) // Assuming one output, batch size 1, output size 1.

	// Create an empty tensor to serve as a placeholder for the output.
	// The ONNX Runtime will write the inference result into this tensor's memory.
	// The type float32 matches the OutputT of NewDynamicSession[float32, float32].
	outputPlaceholder, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Failed to create empty output tensor: %v", err)
		return
	}
	// Crucially, defer the destruction of this tensor.
	defer outputPlaceholder.Destroy()

	// This slice will be passed to session.Run. It contains the tensor(s)
	// where the ONNX runtime will store the results. The number of tensors
	// in this slice must match the number of output names defined for the session.
	// In our case, we have one outputName ("output").
	outputsToFill := []*ort.Tensor[float32]{outputPlaceholder}

	// --- Single Prediction Benchmark Loop ---
	fmt.Println("  Starting single prediction benchmark...")
	start := time.Now()

	for i := 0; i < iterations; i++ {
		err = session.Run(inputTensors, outputsToFill)
		if err != nil {
			log.Fatalf("Failed to run ONNX inference during single prediction benchmark (iteration %d): %v", i, err)
			return // Redundant due to log.Fatalf, but good practice
		}
		// Optional: could inspect outputsToFill[0].GetData() here if needed for debugging,
		// but for a benchmark, we focus on timing the Run call.
	}
	elapsedSingle := time.Since(start)
	avgSingleTime := float64(elapsedSingle.Microseconds()) / float64(iterations)
	fmt.Printf("  Single prediction (ONNX): %v total, (avg %.2f µs/prediction) for %d iterations\n", elapsedSingle, avgSingleTime, iterations)

	// We can verify the last output as a sanity check
	outputTensor := outputsToFill[0] // This is our outputPlaceholder, now filled with data from the last iteration.
	outputData := outputTensor.GetData()

	if len(outputData) == 0 {
		log.Fatalf("Output tensor data is empty after benchmark loop")
		return
	}
	fmt.Printf("  Sample predicted value from last iteration (first element): %f\n", outputData[0])

	// --- Batch Prediction Benchmark Loop ---
	fmt.Println("  Starting batch prediction benchmark...")

	// 1. Prepare a batch input tensor
	// The inputSize here should correctly correspond to the model's expected single item input dimension (e.g., 81)
	randomBatchFloat64 := generateRandomBatch(batchSize, inputSize) // Returns [][]float64
	randomBatchFloat32 := make([]float32, 0, batchSize*inputSize)
	for _, singleInputFloat64 := range randomBatchFloat64 {
		for _, val := range singleInputFloat64 {
			randomBatchFloat32 = append(randomBatchFloat32, float32(val))
		}
	}

	// Shape for a batched input: [batchSize, inputSize]
	batchInputShape := ort.NewShape(int64(batchSize), int64(inputSize))
	batchInputTensor, err := ort.NewTensor(batchInputShape, randomBatchFloat32)
	if err != nil {
		log.Fatalf("Failed to create batch input tensor: %v", err)
		return
	}
	defer batchInputTensor.Destroy()
	fmt.Printf("  Successfully created batch input tensor with shape %v.\n", batchInputShape)

	batchInputTensors := []*ort.Tensor[float32]{batchInputTensor}

	// 2. Prepare a batch output placeholder tensor
	// For rps_value1.model, output is [batchSize, 1]
	batchOutputShape := ort.NewShape(int64(batchSize), 1)
	batchOutputPlaceholder, err := ort.NewEmptyTensor[float32](batchOutputShape)
	if err != nil {
		log.Fatalf("Failed to create batch empty output tensor: %v", err)
		return
	}
	defer batchOutputPlaceholder.Destroy()

	batchOutputsToFill := []*ort.Tensor[float32]{batchOutputPlaceholder}

	// 3. Run batch inference loop
	startBatch := time.Now()
	for i := 0; i < iterations; i++ {
		err = session.Run(batchInputTensors, batchOutputsToFill)
		if err != nil {
			log.Fatalf("Failed to run ONNX inference during batch prediction benchmark (iteration %d): %v", i, err)
			return
		}
	}
	elapsedBatch := time.Since(startBatch)
	avgBatchTime := float64(elapsedBatch.Microseconds()) / float64(iterations*batchSize)
	fmt.Printf("  Batch prediction (ONNX): %v total, (avg %.2f µs/prediction/item) for %d iterations of batch size %d\n", elapsedBatch, avgBatchTime, iterations, batchSize)

	// Sanity check the last batch output
	batchOutputData := batchOutputPlaceholder.GetData()
	if len(batchOutputData) == 0 {
		log.Fatalf("Batch output tensor data is empty after benchmark loop")
	}
	fmt.Printf("  Sample predicted value from last batch iteration (first item, first element): %f\n", batchOutputData[0])

	fmt.Println()
}

func main() {
	// Define command line flags
	inputSize := flag.Int("input-size", defaultInputSize, "Size of the input layer")
	hiddenSize := flag.Int("hidden-size", defaultHiddenSize, "Size of the hidden layer")
	outputSize := flag.Int("output-size", defaultOutputSize, "Size of the output layer")
	iterations := flag.Int("iterations", 100, "Number of iterations for each benchmark")
	batchSize := flag.Int("batch-size", 32, "Batch size for batch processing")
	addr := flag.String("addr", defaultAddr, "The address of the gRPC server")
	cpuOnly := flag.Bool("cpu-only", false, "Run only CPU benchmarks")
	gpuOnly := flag.Bool("gpu-only", false, "Run only GPU benchmarks")
	onnxModel := flag.String("onnx-model", "", "Path to the ONNX model for CPU benchmarking")

	flag.Parse()

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Printf("Benchmarking neural networks with:\n")
	fmt.Printf("  Input size: %d\n", *inputSize)
	fmt.Printf("  Hidden size: %d\n", *hiddenSize)
	fmt.Printf("  Output size: %d\n", *outputSize)
	fmt.Printf("  Iterations: %d\n", *iterations)
	fmt.Printf("  Batch size: %d\n", *batchSize)
	fmt.Println()

	// Run CPU benchmarks
	if !*gpuOnly {
		runCPUAdHocBenchmark_Old(*inputSize, *hiddenSize, *outputSize, *iterations, *batchSize)
		// Also run the ONNX CPU benchmark if a model path is provided
		runCPUONNXBenchmark(*onnxModel, *inputSize, *iterations, *batchSize)
	}

	// Run GPU benchmarks
	if !*cpuOnly {
		fmt.Println("GPU Benchmarks:")
		gpuNetwork, err := gpu.NewRPSGPUPolicyNetwork(*addr)
		if err != nil {
			log.Fatalf("Failed to create GPU network: %v", err)
		}
		defer gpuNetwork.Close()

		// Single prediction benchmark
		gpuSingleTime := benchmarkGPUSingle(gpuNetwork, *inputSize, *iterations)
		gpuSingleAvg := float64(gpuSingleTime.Microseconds()) / float64(*iterations)
		fmt.Printf("  Single prediction: %v (avg %.2f µs/prediction)\n", gpuSingleTime, gpuSingleAvg)

		// Batch prediction benchmark
		gpuBatchTime := benchmarkGPUBatch(gpuNetwork, *inputSize, *batchSize, *iterations)
		gpuBatchAvg := float64(gpuBatchTime.Microseconds()) / float64(*iterations*(*batchSize))
		fmt.Printf("  Batch prediction:  %v (avg %.2f µs/prediction)\n", gpuBatchTime, gpuBatchAvg)

		// Print network stats
		stats := gpuNetwork.GetStats()
		fmt.Printf("  Total calls: %d, Total positions: %d\n", stats.TotalCalls, stats.TotalBatchSize)
		fmt.Printf("  Avg latency: %.2f µs, Avg batch size: %.2f\n", stats.AvgLatencyUs, stats.AvgBatchSize)
		fmt.Println()
	}

	// Print comparison if both CPU and GPU were benchmarked
	if !*cpuOnly && !*gpuOnly {
		// Note: These values are calculated in the blocks above, but would need to be returned
		// or made accessible to actually compare here. This is just placeholder code.
		fmt.Println("Performance Comparison:")
		fmt.Println("  Single prediction speedup: GPU is X times faster than CPU")
		fmt.Println("  Batch prediction speedup: GPU is Y times faster than CPU")
	}
}
