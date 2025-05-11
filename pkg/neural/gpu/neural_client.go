package gpu

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/zachbeta/neural_rps/pkg/neural/proto"
)

// NeuralResponse contains the results from a neural network prediction
type NeuralResponse struct {
	Probabilities []float32
	Value         float32
	BestMove      int32
}

// NetworkStats contains performance metrics for the neural network
type NetworkStats struct {
	TotalCalls     int
	TotalPositions int
	AvgLatencyUs   float64
}

// NeuralClient is a client for the neural service gRPC API
type NeuralClient struct {
	conn       *grpc.ClientConn
	client     proto.NeuralServiceClient
	modelType  string
	inputSize  int
	outputSize int

	// Performance metrics
	totalTime      time.Duration
	totalCalls     int
	totalPositions int
	mu             sync.Mutex
}

// NewNeuralClient creates a new client for the neural service
func NewNeuralClient(addr string, modelType string) (*NeuralClient, error) {
	// Set up connection to the gRPC server
	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to neural service: %v", err)
	}

	// Create gRPC client
	client := proto.NewNeuralServiceClient(conn)

	// Get model info
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := client.GetModelInfo(ctx, &proto.ModelInfoRequest{
		ModelType: modelType,
	})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to get model info: %v", err)
	}

	return &NeuralClient{
		conn:           conn,
		client:         client,
		modelType:      modelType,
		inputSize:      int(info.InputSize),
		outputSize:     int(info.OutputSize),
		totalTime:      0,
		totalCalls:     0,
		totalPositions: 0,
	}, nil
}

// Predict runs inference on a single input
func (c *NeuralClient) Predict(ctx context.Context, features []float32) ([]float32, float32, error) {
	start := time.Time{}
	c.mu.Lock()
	c.totalCalls++
	c.totalPositions++
	c.mu.Unlock()

	start = time.Now()

	// Create request
	req := &proto.PredictRequest{
		Features:  features,
		ModelType: c.modelType,
	}

	// Make gRPC call
	resp, err := c.client.Predict(ctx, req)
	if err != nil {
		return nil, 0, fmt.Errorf("prediction failed: %v", err)
	}

	c.mu.Lock()
	c.totalTime += time.Since(start)
	c.mu.Unlock()

	return resp.Probabilities, resp.Value, nil
}

// PredictBatch runs inference on a batch of inputs
func (c *NeuralClient) PredictBatch(ctx context.Context, batch [][]float32) ([]*NeuralResponse, error) {
	if len(batch) == 0 {
		return []*NeuralResponse{}, nil
	}

	start := time.Time{}
	c.mu.Lock()
	c.totalCalls++
	c.totalPositions += len(batch)
	c.mu.Unlock()

	start = time.Now()

	// Create batch request
	req := &proto.BatchPredictRequest{
		ModelType: c.modelType,
		Inputs:    make([]*proto.InputFeatures, len(batch)),
	}

	// Add each input to the batch
	for i, features := range batch {
		req.Inputs[i] = &proto.InputFeatures{
			Features: features,
		}
	}

	// Make gRPC call
	resp, err := c.client.BatchPredict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("batch prediction failed: %v", err)
	}

	c.mu.Lock()
	c.totalTime += time.Since(start)
	c.mu.Unlock()

	// Convert gRPC response to our domain model
	results := make([]*NeuralResponse, len(resp.Outputs))
	for i, output := range resp.Outputs {
		results[i] = &NeuralResponse{
			Probabilities: output.Probabilities,
			Value:         output.Value,
			BestMove:      output.BestMove,
		}
	}

	return results, nil
}

// GetInputSize returns the neural network input size
func (c *NeuralClient) GetInputSize() int {
	return c.inputSize
}

// GetOutputSize returns the neural network output size
func (c *NeuralClient) GetOutputSize() int {
	return c.outputSize
}

// GetStats returns performance statistics for the neural network
func (c *NeuralClient) GetStats() NetworkStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	var avgLatency float64
	if c.totalCalls > 0 {
		avgLatency = float64(c.totalTime.Microseconds()) / float64(c.totalCalls)
	}

	return NetworkStats{
		TotalCalls:     c.totalCalls,
		TotalPositions: c.totalPositions,
		AvgLatencyUs:   avgLatency,
	}
}

// Close closes the connection to the neural service
func (c *NeuralClient) Close() error {
	return c.conn.Close()
}
