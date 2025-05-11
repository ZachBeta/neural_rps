package gpu

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/zachbeta/neural_rps/pkg/common"
	pb "github.com/zachbeta/neural_rps/pkg/neural/proto"
)

// RPSGPUPolicyNetwork is a policy network that uses the gRPC service for GPU-accelerated inference
type RPSGPUPolicyNetwork struct {
	conn   *grpc.ClientConn
	client pb.NeuralServiceClient

	// Network dimensions
	InputSize  int
	HiddenSize int
	OutputSize int

	// Performance metrics
	totalTime      time.Duration
	totalCalls     int
	totalBatchSize int
}

// NewRPSGPUPolicyNetwork creates a new policy network client that uses GPU acceleration
func NewRPSGPUPolicyNetwork(addr string) (*RPSGPUPolicyNetwork, error) {
	// Set up a connection to the server
	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to neural service: %v", err)
	}

	client := pb.NewNeuralServiceClient(conn)

	// Get model info
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := client.GetModelInfo(ctx, &pb.ModelInfoRequest{
		ModelType: "policy",
	})

	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to get model info: %v", err)
	}

	return &RPSGPUPolicyNetwork{
		conn:           conn,
		client:         client,
		InputSize:      int(info.InputSize),
		HiddenSize:     int(info.HiddenSize),
		OutputSize:     int(info.OutputSize),
		totalTime:      0,
		totalCalls:     0,
		totalBatchSize: 0,
	}, nil
}

// Forward runs a forward pass through the policy network
func (n *RPSGPUPolicyNetwork) Forward(input []float64) ([]float64, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize++

	// Convert float64 to float32 for gRPC
	features := make([]float32, len(input))
	for i, v := range input {
		features[i] = float32(v)
	}

	// Create the request
	req := &pb.PredictRequest{
		Features:  features,
		ModelType: "policy",
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %v", err)
	}

	// Convert float32 to float64 for output
	output := make([]float64, len(resp.Probabilities))
	for i, v := range resp.Probabilities {
		output[i] = float64(v)
	}

	n.totalTime += time.Since(start)

	return output, nil
}

// Predict returns the best move based on the input
func (n *RPSGPUPolicyNetwork) Predict(input []float64) (int, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize++

	// Convert float64 to float32 for gRPC
	features := make([]float32, len(input))
	for i, v := range input {
		features[i] = float32(v)
	}

	// Create the request
	req := &pb.PredictRequest{
		Features:  features,
		ModelType: "policy",
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.Predict(ctx, req)
	if err != nil {
		return -1, fmt.Errorf("prediction failed: %v", err)
	}

	n.totalTime += time.Since(start)

	return int(resp.BestMove), nil
}

// BatchPredict performs inference on multiple inputs at once
func (n *RPSGPUPolicyNetwork) BatchPredict(inputs [][]float64) ([]int, error) {
	if len(inputs) == 0 {
		return []int{}, nil
	}

	start := time.Now()
	n.totalCalls++
	n.totalBatchSize += len(inputs)

	// Create batch request
	req := &pb.BatchPredictRequest{
		ModelType: "policy",
		Inputs:    make([]*pb.InputFeatures, len(inputs)),
	}

	// Convert all inputs to float32
	for i, input := range inputs {
		features := make([]float32, len(input))
		for j, v := range input {
			features[j] = float32(v)
		}
		req.Inputs[i] = &pb.InputFeatures{
			Features: features,
		}
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.BatchPredict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("batch prediction failed: %v", err)
	}

	// Extract predictions
	predictions := make([]int, len(resp.Outputs))
	for i, output := range resp.Outputs {
		predictions[i] = int(output.BestMove)
	}

	n.totalTime += time.Since(start)

	return predictions, nil
}

// BatchForward performs forward passes on multiple inputs at once
func (n *RPSGPUPolicyNetwork) BatchForward(inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return [][]float64{}, nil
	}

	start := time.Now()
	n.totalCalls++
	n.totalBatchSize += len(inputs)

	// Create batch request
	req := &pb.BatchPredictRequest{
		ModelType: "policy",
		Inputs:    make([]*pb.InputFeatures, len(inputs)),
	}

	// Convert all inputs to float32
	for i, input := range inputs {
		features := make([]float32, len(input))
		for j, v := range input {
			features[j] = float32(v)
		}
		req.Inputs[i] = &pb.InputFeatures{
			Features: features,
		}
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.BatchPredict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("batch prediction failed: %v", err)
	}

	// Extract probabilities
	outputs := make([][]float64, len(resp.Outputs))
	for i, output := range resp.Outputs {
		probs := make([]float64, len(output.Probabilities))
		for j, p := range output.Probabilities {
			probs[j] = float64(p)
		}
		outputs[i] = probs
	}

	n.totalTime += time.Since(start)

	return outputs, nil
}

// Close closes the gRPC connection
func (n *RPSGPUPolicyNetwork) Close() error {
	if n.conn != nil {
		return n.conn.Close()
	}
	return nil
}

// GetStats returns performance statistics
func (n *RPSGPUPolicyNetwork) GetStats() common.NetworkStats {
	var avgLatency float64
	if n.totalCalls > 0 {
		avgLatency = float64(n.totalTime.Microseconds()) / float64(n.totalCalls)
	}

	var avgBatchSize float64
	if n.totalCalls > 0 {
		avgBatchSize = float64(n.totalBatchSize) / float64(n.totalCalls)
	}

	return common.NetworkStats{
		TotalCalls:     n.totalCalls,
		TotalBatchSize: n.totalBatchSize,
		AvgLatencyUs:   avgLatency,
		AvgBatchSize:   avgBatchSize,
	}
}

// RPSGPUValueNetwork is a value network that uses the gRPC service for GPU-accelerated inference
type RPSGPUValueNetwork struct {
	conn   *grpc.ClientConn
	client pb.NeuralServiceClient

	// Network dimensions
	InputSize  int
	HiddenSize int
	OutputSize int

	// Performance metrics
	totalTime      time.Duration
	totalCalls     int
	totalBatchSize int
}

// NewRPSGPUValueNetwork creates a new value network client that uses GPU acceleration
func NewRPSGPUValueNetwork(addr string) (*RPSGPUValueNetwork, error) {
	// Set up a connection to the server
	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to neural service: %v", err)
	}

	client := pb.NewNeuralServiceClient(conn)

	// Get model info
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := client.GetModelInfo(ctx, &pb.ModelInfoRequest{
		ModelType: "value",
	})

	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to get model info: %v", err)
	}

	return &RPSGPUValueNetwork{
		conn:           conn,
		client:         client,
		InputSize:      int(info.InputSize),
		HiddenSize:     int(info.HiddenSize),
		OutputSize:     int(info.OutputSize),
		totalTime:      0,
		totalCalls:     0,
		totalBatchSize: 0,
	}, nil
}

// Evaluate returns a value estimation for the given input
func (n *RPSGPUValueNetwork) Evaluate(input []float64) (float64, error) {
	start := time.Now()
	n.totalCalls++
	n.totalBatchSize++

	// Convert float64 to float32 for gRPC
	features := make([]float32, len(input))
	for i, v := range input {
		features[i] = float32(v)
	}

	// Create the request
	req := &pb.PredictRequest{
		Features:  features,
		ModelType: "value",
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.Predict(ctx, req)
	if err != nil {
		return 0, fmt.Errorf("evaluation failed: %v", err)
	}

	n.totalTime += time.Since(start)

	return float64(resp.Value), nil
}

// BatchEvaluate performs evaluations on multiple inputs at once
func (n *RPSGPUValueNetwork) BatchEvaluate(inputs [][]float64) ([]float64, error) {
	if len(inputs) == 0 {
		return []float64{}, nil
	}

	start := time.Now()
	n.totalCalls++
	n.totalBatchSize += len(inputs)

	// Create batch request
	req := &pb.BatchPredictRequest{
		ModelType: "value",
		Inputs:    make([]*pb.InputFeatures, len(inputs)),
	}

	// Convert all inputs to float32
	for i, input := range inputs {
		features := make([]float32, len(input))
		for j, v := range input {
			features[j] = float32(v)
		}
		req.Inputs[i] = &pb.InputFeatures{
			Features: features,
		}
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Make the gRPC call
	resp, err := n.client.BatchPredict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("batch evaluation failed: %v", err)
	}

	// Extract values
	values := make([]float64, len(resp.Outputs))
	for i, output := range resp.Outputs {
		values[i] = float64(output.Value)
	}

	n.totalTime += time.Since(start)

	return values, nil
}

// Close closes the gRPC connection
func (n *RPSGPUValueNetwork) Close() error {
	if n.conn != nil {
		return n.conn.Close()
	}
	return nil
}

// GetStats returns performance statistics
func (n *RPSGPUValueNetwork) GetStats() common.NetworkStats {
	var avgLatency float64
	if n.totalCalls > 0 {
		avgLatency = float64(n.totalTime.Microseconds()) / float64(n.totalCalls)
	}

	var avgBatchSize float64
	if n.totalCalls > 0 {
		avgBatchSize = float64(n.totalBatchSize) / float64(n.totalCalls)
	}

	return common.NetworkStats{
		TotalCalls:     n.totalCalls,
		TotalBatchSize: n.totalBatchSize,
		AvgLatencyUs:   avgLatency,
		AvgBatchSize:   avgBatchSize,
	}
}
